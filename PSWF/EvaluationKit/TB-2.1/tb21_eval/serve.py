"""Serve LAR-27B for the evaluation: 8 x TP1 SGLang replicas + a cache-aware router + the loopback model guard.

    python -m tb21_eval.serve up     --model M --work-dir W [--gpus 0,1,2,3,4,5,6,7]
    python -m tb21_eval.serve status --work-dir W
    python -m tb21_eval.serve down   --work-dir W

Run it in the serving environment (SGLang 0.5.13.post1). `up` starts every process detached (they keep running
after this command returns), waits until all replicas are healthy and registered with the router, checks that
prefix caching and reasoning separation work, starts the model guard on 127.0.0.1:8795 (what harbor talks to) and
exits. Logs and pids go to W/serve/.

The engine and router settings are defined in tb21_eval.ENGINE_*_ARGS and
ROUTER_*_ARGS. Each selected GPU runs one TP1 replica. SGLang's configured
request cap applies independently to every replica.
Keep the work directory, logs and caches outside the source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import tb21_eval as T


def log(msg: str) -> None:
    print(f"[serve] {msg}", flush=True)


def die(msg: str) -> None:
    raise SystemExit(f"[serve] FATAL: {msg}")


def http(method: str, url: str, body: dict | None = None, timeout: float = 10.0) -> tuple[int, dict | str]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never route loopback via a proxy
    try:
        with opener.open(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(errors="replace"), e.code
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def serve_dir(work_dir: Path) -> Path:
    d = work_dir / "serve"
    d.mkdir(parents=True, exist_ok=True)
    return d


def spawn(name: str, cmd: list[str], sd: Path, env: dict) -> subprocess.Popen:
    """Start a detached process (own session) logging to W/serve/<name>.log and record its pid."""
    with open(sd / f"{name}.log", "ab") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=out, stdin=subprocess.DEVNULL, env=env,
                                start_new_session=True)
    (sd / f"{name}.cmd.json").write_text(json.dumps(cmd, indent=1))
    pids = json.loads((sd / "pids.json").read_text()) if (sd / "pids.json").exists() else {}
    pids[name] = proc.pid
    (sd / "pids.json").write_text(json.dumps(pids, indent=1))
    return proc


def wait_until(pred, timeout: float, what: str, poll: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(poll)
    die(f"timed out waiting for {what}")


# -------------------------------------------------------------------------------------------------------- checks
def preflight(args: argparse.Namespace, gpus: list[int]) -> None:
    import importlib.metadata as md

    for pkg, want in (("sglang", T.SGLANG_VERSION), ("transformers", T.TRANSFORMERS_VERSION)):
        have = md.version(pkg)
        if have != want:
            die(f"{pkg} {have} is installed; the supported stack is {pkg}=={want} (see requirements/serve.txt)")
    if not md.version("torch").startswith(T.TORCH_VERSION_PREFIX):
        die(f"torch {md.version('torch')} is installed; the supported stack is torch {T.TORCH_VERSION_PREFIX}+cu130")
    if md.version("sglang-router") != T.SGLANG_ROUTER_VERSION:
        die(f"sglang-router {md.version('sglang-router')} is installed; the supported stack is {T.SGLANG_ROUTER_VERSION}")
    cfg = json.loads((args.model / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    rope = text_cfg.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or text_cfg.get("rope_scaling") or cfg.get("rope_scaling"):
        die("the model config carries a RoPE override (e.g. YaRN); serve the original checkpoint")
    for name in ("tokenizer.json", "chat_template.jinja"):
        if sha256(args.model / name) != T.EXPECTED_SHA256[name]:
            if not args.allow_different_checkpoint:
                die(f"{name} differs from the supported LAR-27B checkpoint (the prompt tokens would change); "
                    "pass --allow-different-checkpoint to serve it anyway")
            log(f"WARNING: {name} differs from the supported LAR-27B checkpoint")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
    used = {int(a): int(b) for a, b in (line.split(",") for line in query.strip().splitlines())}
    busy = [g for g in gpus if used.get(g, 0) > 1024]
    if busy and not args.allow_busy_gpus:
        die(f"GPU(s) {busy} already hold memory; stop what runs there first (the kit never kills other processes)")


def prefix_cache_ok(url: str) -> bool:
    """Check cache reuse by extending a synthetic prompt with one additional turn."""
    filler = "Shared evaluation context.\n" + "The quick brown fox jumps over the lazy dog. " * 800
    msgs = [{"role": "user", "content": filler + "\nReply with OK."}]
    body = {"model": T.SERVED_MODEL, "messages": msgs, "max_tokens": 48, "temperature": 1.0, "top_p": 0.95,
            "chat_template_kwargs": T.CHAT_TEMPLATE_KWARGS}
    s1, r1 = http("POST", f"{url}/v1/chat/completions", body, timeout=600)
    if s1 != 200:
        die(f"prefix-cache probe on {url} failed: {s1} {str(r1)[:300]}")
    answer = r1["choices"][0]["message"].get("content") or ""
    body["messages"] = msgs + [{"role": "assistant", "content": answer}, {"role": "user", "content": "Now reply with DONE."}]
    s2, r2 = http("POST", f"{url}/v1/chat/completions", body, timeout=600)
    if s2 != 200:
        die(f"prefix-cache probe on {url} failed: {s2} {str(r2)[:300]}")
    cached = ((r2.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    log(f"prefix cache {url}: prompt {r1['usage']['prompt_tokens']} -> {r2['usage']['prompt_tokens']}, cached {cached}")
    return cached >= 4000


def reasoning_ok(base: str) -> bool:
    body = {"model": T.SERVED_MODEL, "messages": [{"role": "user", "content": 'Return the JSON object {"ok": true}.'}],
            "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95,
            "chat_template_kwargs": T.CHAT_TEMPLATE_KWARGS, "top_k": 20}
    status, r = http("POST", f"{base}/v1/chat/completions", body, timeout=900)
    if status != 200:
        die(f"chat probe via {base} failed: {status} {str(r)[:300]}")
    msg = r["choices"][0]["message"]
    return bool((msg.get("reasoning_content") or "").strip()) and bool((msg.get("content") or "").strip())


# ------------------------------------------------------------------------------------------------------ commands
def url_of(i: int) -> str:
    return f"http://{T.HOST}:{T.ENGINE_BASE_PORT + i}"


def guard_is_current(guard: str, router: str) -> bool:
    """Whether the model guard answering on `guard` runs the installed model_guard.py + policy.py against `router`."""
    status, stats = http("GET", f"{guard}/guard/stats")
    if status != 200 or not isinstance(stats, dict):
        return False
    return (stats.get("guard_sha256") == sha256(T.GUARD_DIR / "model_guard.py")
            and stats.get("policy_sha256") == sha256(T.GUARD_DIR / "policy.py")
            and stats.get("upstream") == f"{router}/v1")


def engine_ready(url: str) -> bool:
    if http("GET", f"{url}/health", timeout=30)[0] != 200:
        return False
    status, models = http("GET", f"{url}/v1/models")
    return status == 200 and any(m.get("id") == T.SERVED_MODEL for m in models.get("data", []))


def cmd_up(args: argparse.Namespace) -> None:
    """Start (or complete) the serving stack. Healthy components from an earlier `up` are reused, so re-running
    after an interruption only starts what is missing."""
    sd = serve_dir(args.work_dir)
    gpus = [int(g) for g in args.gpus.split(",")]
    py = sys.executable
    # Engines get the operator's environment minus anything that could change their behaviour behind our back.
    drop = ("SGLANG_", "SGL_", "FLASHINFER_", "PYTORCH_", "VLLM_", "TORCHINDUCTOR_", "TRITON_")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(drop) and k not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")}
    # The environment's bin/ goes first on PATH (SGLang's kernel JIT runs `ninja` from it).
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               TORCH_EXTENSIONS_DIR=str(sd / "cache" / "torch_extensions"),
               FLASHINFER_WORKSPACE_BASE=str(sd / "cache" / "flashinfer"))
    router = f"http://{T.HOST}:{T.ROUTER_PORT}"
    engines = [url_of(i) for i in range(len(gpus))]
    running = [url for url in engines if engine_ready(url)]
    missing = [i for i, url in enumerate(engines) if url not in running]
    if running and missing and not args.repair:
        die(f"{len(running)}/{len(engines)} replicas are running; `serve up --repair` relaunches the missing ones, "
            "`serve down` stops everything")
    if missing:
        preflight(args, gpus=[gpus[i] for i in missing])

    if http("GET", f"{router}/health")[0] == 200:
        log(f"router already up on {router}")
    else:
        spawn("router", [py, "-m", "sglang_router.launch_router", "--host", T.HOST, "--port", str(T.ROUTER_PORT),
                         *T.ROUTER_POLICY_ARGS, "--tokenizer-path", str(args.model), *T.ROUTER_BALANCE_ARGS,
                         "--prometheus-host", T.HOST, "--prometheus-port", str(T.ROUTER_METRICS_PORT)], sd, env)
        wait_until(lambda: http("GET", f"{router}/health")[0] == 200, 300, "the router")
        log(f"router up on {router} (cache_aware)")

    procs: dict[str, subprocess.Popen] = {}
    if running:
        log(f"reusing {len(running)} healthy replicas")
    if missing:
        for i in missing:
            gpu = gpus[i]
            cmd = [py, "-m", "sglang.launch_server", "--model-path", str(args.model),
                   "--served-model-name", T.SERVED_MODEL, *T.ENGINE_PARALLEL_ARGS,
                   "--nccl-port", str(25000 + i), "--dist-init-addr", f"127.0.0.1:{26000 + i}",
                   "--host", T.HOST, "--port", str(T.ENGINE_BASE_PORT + i), *T.ENGINE_ARGS]
            procs[url_of(i)] = spawn(f"engine_{T.ENGINE_BASE_PORT + i}", cmd, sd,
                                     dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
            time.sleep(1)
        log(f"launched {len(missing)} TP1 replicas on GPUs {[gpus[i] for i in missing]}; waiting for them "
            "(the first start compiles kernels)")

    def registered() -> set[str]:
        status, w = http("GET", f"{router}/workers")
        return {x.get("url") for x in w.get("workers", [])} if status == 200 else set()

    for url in engines:
        def healthy(url=url):
            if url in procs and procs[url].poll() is not None:
                die(f"replica {url} exited; see {sd}/engine_{url.rsplit(':', 1)[1]}.log")
            return engine_ready(url)
        wait_until(healthy, 2400, f"replica {url}", poll=5)
        if url not in registered():
            status, resp = http("POST", f"{router}/workers", {"url": url})
            if status >= 300:
                die(f"router refused worker {url}: {status} {resp}")
        log(f"replica {url} healthy and registered")
    wait_until(lambda: set(engines) <= registered(), 300, "router worker registration")
    foreign = registered() - set(engines)
    if foreign:
        die(f"the router also routes to {sorted(foreign)}; run `serve down` and start a clean stack")
    for url in engines:
        if not prefix_cache_ok(url):
            die(f"replica {url} shows no prefix-cache reuse")
    if not reasoning_ok(router):
        die("router chat probe: reasoning_content and content were not both returned")
    log("router chat probe OK (reasoning separated from content)")

    guard = f"http://{T.HOST}:{T.MODEL_GUARD_PORT}"
    if http("GET", f"{guard}/health")[0] == 200:
        if not guard_is_current(guard, router):
            die(f"the model guard on {guard} is not running the installed model_guard.py/policy.py against {router}; "
                "stop it (serve down) and run `serve up` again")
        log(f"model guard already up on {guard} (verified: installed code and policy, upstream {router})")
    else:
        rec = sd / "model_guard"
        rec.mkdir(exist_ok=True)
        proc = spawn("model_guard", [py, "-B", str(T.GUARD_DIR / "model_guard.py"), "--host", T.HOST,
                                     "--port", str(T.MODEL_GUARD_PORT), "--upstream", router, "--record-dir", str(rec),
                                     "--timeout", str(int(T.API_TIMEOUT_SEC))], sd, dict(env, PYTHONPATH=str(T.GUARD_DIR)))

        def guard_up():
            if proc.poll() is not None:
                die(f"the model guard exited; see {sd}/model_guard.log")
            return http("GET", f"{guard}/health")[0] == 200
        wait_until(guard_up, 120, "the model guard", poll=1)
    if not reasoning_ok(guard):
        die("model guard chat probe failed")
    log(f"model guard up on {guard}/v1 -> {router}")
    print(f"SERVE_READY api_base={guard}/v1 model={T.SERVED_MODEL}")


def cmd_status(args: argparse.Namespace) -> None:
    sd = serve_dir(args.work_dir)
    router = f"http://{T.HOST}:{T.ROUTER_PORT}"
    st = {"router": http("GET", f"{router}/health")[0] == 200,
          "model_guard": http("GET", f"http://{T.HOST}:{T.MODEL_GUARD_PORT}/health")[0] == 200}
    status, w = http("GET", f"{router}/workers")
    st["workers"] = len(w.get("workers", [])) if status == 200 else 0
    pids = json.loads((sd / "pids.json").read_text()) if (sd / "pids.json").exists() else {}
    for name in sorted(pids):
        if name.startswith("engine_"):
            # SGLang's /health runs a tiny generation: under full load it can take a few seconds.
            st[name] = http("GET", f"http://{T.HOST}:{name.split('_')[1]}/health", timeout=30)[0] == 200
    print(json.dumps(st, indent=2))
    ok = st["router"] and st["model_guard"] and st["workers"] > 0 and all(v for k, v in st.items() if k.startswith("engine_"))
    raise SystemExit(0 if ok else 1)


def cmd_down(args: argparse.Namespace) -> None:
    sd = serve_dir(args.work_dir)
    pids = json.loads((sd / "pids.json").read_text()) if (sd / "pids.json").exists() else {}
    for name, pid in pids.items():
        try:
            os.killpg(pid, signal.SIGTERM)       # each process was started in its own session/process group
            log(f"stopped {name} (pid {pid})")
        except ProcessLookupError:
            pass
    deadline = time.time() + 60
    for pid in pids.values():
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    (sd / "pids.json").unlink(missing_ok=True)
    print("SERVE_DOWN")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["up", "status", "down"])
    ap.add_argument("--work-dir", type=Path, required=True, help="kit work directory (logs go to W/serve)")
    ap.add_argument("--model", type=Path, help="local LAR-27B checkpoint directory (required for `up`)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="one TP1 replica per listed GPU (default: all 8)")
    ap.add_argument("--repair", action="store_true",
                    help="relaunch only the replicas that are not running (e.g. one died during the run)")
    ap.add_argument("--allow-different-checkpoint", action="store_true",
                    help="serve a checkpoint whose tokenizer/chat template differ from the supported model")
    ap.add_argument("--allow-busy-gpus", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.work_dir = args.work_dir.resolve()
    if args.command == "up":
        if args.model is None or not (args.model / "config.json").is_file():
            die("--model must point at the LAR-27B checkpoint directory")
        args.model = args.model.resolve()
    {"up": cmd_up, "status": cmd_status, "down": cmd_down}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

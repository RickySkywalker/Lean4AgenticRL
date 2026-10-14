"""Network guard for the TB-2.1 task containers.

    sudo $(which python) -m tb21_eval.netguard up     --work-dir W
    python -m tb21_eval.netguard probe  --work-dir W     # end-to-end containment check from inside a container
    python -m tb21_eval.netguard status --work-dir W
    sudo $(which python) -m tb21_eval.netguard down   --work-dir W

Task containers run on an `--internal` docker bridge with no route to the internet. The only network-capable path
out is an allowlist HTTP/CONNECT proxy (guard/egress_proxy.py) and an allowlist DNS responder (guard/dns_proxy.py),
both bound to the bridge gateway, plus an nftables table that rejects every other host surface and forwards
nothing. The compose overlay written by `tb21_eval.make_job` attaches each task container to this network and
points HTTP(S)_PROXY and DNS at the gateway. Agents can therefore install packages and download task data from the
allowlisted domains (guard/allowed_domains.txt) and cannot reach Tor, open
proxies, cloud metadata or anything on the host. `up` needs root (port 53 and nftables); the processes it starts
are recorded under W/netguard/ so `down` can stop them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import tb21_eval as T

PROXY_NAMES = ("egress", "dns")


def state_dir(work_dir: Path) -> Path:
    return work_dir / "netguard"


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True, **kw)


def die(msg: str) -> None:
    raise SystemExit(f"[netguard] FATAL: {msg}")


def log(msg: str) -> None:
    print(f"[netguard] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ docker network
def network_info() -> dict | None:
    done = run(["docker", "network", "inspect", T.GUARD_NETWORK], check=False)
    return json.loads(done.stdout)[0] if done.returncode == 0 else None


def check_network(info: dict) -> None:
    ipam = (info.get("IPAM", {}).get("Config") or [{}])[0]
    opts = info.get("Options") or {}
    problems = []
    if info.get("Internal") is not True:
        problems.append("not --internal")
    if ipam.get("Subnet") != T.GUARD_SUBNET or ipam.get("Gateway") != T.GUARD_GATEWAY:
        problems.append(f"subnet/gateway {ipam.get('Subnet')}/{ipam.get('Gateway')}")
    if opts.get("com.docker.network.bridge.name") != T.GUARD_BRIDGE:
        problems.append("bridge name")
    if opts.get("com.docker.network.bridge.enable_icc") != "false":
        problems.append("inter-container communication enabled")
    if problems:
        die(f"docker network {T.GUARD_NETWORK} exists with the wrong shape: {', '.join(problems)}")


def create_network() -> None:
    info = network_info()
    if info is None:
        run(["docker", "network", "create", "--driver", "bridge", "--internal",
             "--subnet", T.GUARD_SUBNET, "--gateway", T.GUARD_GATEWAY,
             "--opt", f"com.docker.network.bridge.name={T.GUARD_BRIDGE}",
             "--opt", "com.docker.network.bridge.enable_icc=false", T.GUARD_NETWORK])
        info = network_info()
    check_network(info)
    log(f"docker network {T.GUARD_NETWORK}: internal, {T.GUARD_SUBNET}, gateway {T.GUARD_GATEWAY}, icc off")


# ------------------------------------------------------------------------------------------------------ nftables
def render_nft() -> str:
    text = (T.GUARD_DIR / "nftables.conf.in").read_text()
    for key, value in {"__TABLE__": T.GUARD_NFT_TABLE, "__BRIDGE__": T.GUARD_BRIDGE, "__GATEWAY__": T.GUARD_GATEWAY,
                       "__DNS_PORT__": str(T.DNS_PORT), "__PROXY_PORT__": str(T.EGRESS_PROXY_PORT)}.items():
        text = text.replace(key, value)
    return text


def load_nft(sd: Path) -> None:
    # One nft transaction: create the table if missing, delete it, define it again. `nft -f` applies the whole file
    # atomically, so re-running `up` during a run never leaves the bridge unfiltered, even for a moment.
    rules = f"table inet {T.GUARD_NFT_TABLE}\ndelete table inet {T.GUARD_NFT_TABLE}\n" + render_nft()
    with tempfile.NamedTemporaryFile("w", suffix=".nft", delete=False) as fh:
        fh.write(rules)
        path = fh.name
    try:
        run(["nft", "-c", "-f", path])
        run(["nft", "-f", path])
    finally:
        os.unlink(path)
    loaded = run(["nft", "list", "table", "inet", T.GUARD_NFT_TABLE]).stdout
    (sd / "nft.loaded").write_text(loaded)
    for marker in ("pswf_host_surface_reject", "pswf_direct_egress_reject", "pswf_proxy_allow", "pswf_imds_drop"):
        if marker not in loaded:
            die(f"kernel readback of table {T.GUARD_NFT_TABLE} lacks {marker}")
    log(f"nftables table inet {T.GUARD_NFT_TABLE} loaded (DNS + proxy on the gateway only; no forwarding)")


# ------------------------------------------------------------------------------------------------------ proxies
def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid(sd: Path, name: str) -> int | None:
    try:
        return int((sd / f"{name}.pid").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def heartbeat_age(sd: Path, name: str) -> float | None:
    try:
        return time.time() - (sd / f"{name}-heartbeat.json").stat().st_mtime
    except OSError:
        return None


def is_proxy(pid: int, name: str) -> bool:
    """Whether `pid` still runs the kit's <name>_proxy.py (pids are recycled; never signal a stranger)."""
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return any(a.endswith(f"{name}_proxy.py".encode()) for a in argv)


def stop_proxies(sd: Path) -> None:
    for name in PROXY_NAMES:
        pid = read_pid(sd, name)
        if pid and pid_alive(pid) and is_proxy(pid, name):
            os.kill(pid, 15)
            for _ in range(50):
                if not pid_alive(pid):
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, 9)
            log(f"stopped {name} proxy (pid {pid})")


def start_proxies(sd: Path) -> None:
    stop_proxies(sd)
    allow, deny = str(T.GUARD_DIR / "allowed_domains.txt"), str(T.GUARD_DIR / "denied_domains.txt")
    lim = T.EGRESS_LIMITS
    commands = {
        "egress": [sys.executable, "-B", str(T.GUARD_DIR / "egress_proxy.py"),
                   "--host", T.GUARD_GATEWAY, "--port", str(T.EGRESS_PROXY_PORT),
                   "--allowlist", allow, "--denylist", deny, "--state-dir", str(sd),
                   "--max-active", str(lim["max_active"]), "--max-new-10s", str(lim["max_new_10s"]),
                   "--max-domains", str(lim["max_domains"])],
        "dns": [sys.executable, "-B", str(T.GUARD_DIR / "dns_proxy.py"),
                "--host", T.GUARD_GATEWAY, "--port", str(T.DNS_PORT),
                "--allowlist", allow, "--denylist", deny, "--state-dir", str(sd)],
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("Q38_")}   # no ban-set / probe overrides
    for name, cmd in commands.items():
        (sd / f"{name}-heartbeat.json").unlink(missing_ok=True)
        with open(sd / f"{name}.log", "ab") as out:
            subprocess.Popen(cmd, stdout=out, stderr=out, stdin=subprocess.DEVNULL, env=env, cwd=str(T.GUARD_DIR),
                             start_new_session=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        ages = [heartbeat_age(sd, n) for n in PROXY_NAMES]
        if all(a is not None and a < 10 for a in ages):
            log(f"egress proxy {T.GUARD_GATEWAY}:{T.EGRESS_PROXY_PORT} and DNS {T.GUARD_GATEWAY}:{T.DNS_PORT} are up")
            return
        time.sleep(0.2)
    for name in PROXY_NAMES:
        sys.stderr.write((sd / f"{name}.log").read_text()[-3000:])
    die("network proxies did not become healthy")


# ---------------------------------------------------------------------------------------------------- commands
def cmd_up(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        die("`up` needs root (DNS on port 53 and nftables): sudo $(which python) -m tb21_eval.netguard up ...")
    for tool in ("docker", "nft"):
        if shutil.which(tool) is None:
            die(f"{tool} not found")
    sd = state_dir(args.work_dir)
    sd.mkdir(parents=True, exist_ok=True)
    create_network()
    load_nft(sd)
    start_proxies(sd)
    owner = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if all(owner):                       # leave the state readable/removable by the invoking user
        for p in [sd, *sd.iterdir()]:
            os.chown(p, int(owner[0]), int(owner[1]))
    print("NETGUARD_UP")


def cmd_down(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        die("`down` needs root: sudo $(which python) -m tb21_eval.netguard down ...")
    sd = state_dir(args.work_dir)
    info = network_info()
    if info is not None and info.get("Containers"):     # checked first: a running trial must keep its guard
        die(f"refusing to tear down: {len(info['Containers'])} container(s) still attached to {T.GUARD_NETWORK}")
    stop_proxies(sd)
    run(["nft", "delete", "table", "inet", T.GUARD_NFT_TABLE], check=False)
    if info is not None:
        run(["docker", "network", "rm", T.GUARD_NETWORK])
    print("NETGUARD_DOWN")


def status(work_dir: Path) -> dict:
    sd = state_dir(work_dir)
    info = network_info()
    # Reading the ruleset needs root; without it (and without passwordless sudo) the table state is unknown (None).
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    nft = run(prefix + ["nft", "list", "table", "inet", T.GUARD_NFT_TABLE], check=False)
    if nft.returncode == 0:
        nft_ok: bool | None = "pswf_host_surface_reject" in nft.stdout
    elif "password" in nft.stderr.lower():
        nft_ok = None
    else:
        nft_ok = False
    st = {"network": bool(info and info.get("Internal")), "nft_table": nft_ok}
    for name in PROXY_NAMES:
        pid = read_pid(sd, name)
        age = heartbeat_age(sd, name)
        st[name] = bool(pid and pid_alive(pid) and is_proxy(pid, name) and age is not None and age < 10)
    st["ok"] = bool(st["network"] and st["egress"] and st["dns"] and st["nft_table"] is not False)
    return st


def cmd_status(args: argparse.Namespace) -> None:
    st = status(args.work_dir)
    print(json.dumps(st, indent=2))
    raise SystemExit(0 if st["ok"] else 1)


PROBE = r'''
import json, os, socket, urllib.request
gw = os.environ["GW"]
def tcp(host, port, t=3):
    try:
        socket.create_connection((host, port), timeout=t).close(); return True
    except OSError:
        return False
def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status
    except Exception as e:
        return type(e).__name__ + ":" + str(e)[:80]
def resolves(name):
    try:
        socket.getaddrinfo(name, 443); return True
    except OSError:
        return False
r = {
  "direct_tcp_1.1.1.1:443": tcp("1.1.1.1", 443),
  "direct_tcp_8.8.8.8:53": tcp("8.8.8.8", 53),
  "imds_169.254.169.254:80": tcp("169.254.169.254", 80),
  "host_ssh_gateway:22": tcp(gw, 22),
  "host_sglang_gateway:26000": tcp(gw, 26000),
  "proxy_allowlisted_pypi": fetch("https://pypi.org/simple/pip/"),
  "proxy_denied_torproject": fetch("https://www.torproject.org/"),
  "proxy_unlisted_example": fetch("https://example.com/"),
  "dns_allowlisted_pypi.org": resolves("pypi.org"),
  "dns_unlisted_example.com": resolves("example.com"),
}
print(json.dumps(r))
'''

EXPECT = {
    "direct_tcp_1.1.1.1:443": False, "direct_tcp_8.8.8.8:53": False, "imds_169.254.169.254:80": False,
    "host_ssh_gateway:22": False, "host_sglang_gateway:26000": False,
    "proxy_allowlisted_pypi": 200, "dns_allowlisted_pypi.org": True, "dns_unlisted_example.com": False,
}


def cmd_probe(args: argparse.Namespace) -> None:
    proxy = f"http://{T.GUARD_GATEWAY}:{T.EGRESS_PROXY_PORT}"
    cmd = ["docker", "run", "--rm", "--network", T.GUARD_NETWORK, "--dns", T.GUARD_GATEWAY,
           "--cap-drop", "NET_RAW", "--security-opt", "no-new-privileges:true",
           "-e", f"GW={T.GUARD_GATEWAY}", "-e", f"HTTP_PROXY={proxy}", "-e", f"HTTPS_PROXY={proxy}",
           "-e", f"http_proxy={proxy}", "-e", f"https_proxy={proxy}",
           args.image, "python3", "-c", PROBE]
    done = run(cmd, check=False, timeout=300)
    if done.returncode != 0:
        die(f"probe container failed: {done.stderr[-2000:]}")
    result = json.loads(done.stdout.strip().splitlines()[-1])
    bad = {k: v for k, v in EXPECT.items() if result.get(k) != v}
    for key in ("proxy_denied_torproject", "proxy_unlisted_example"):
        if result.get(key) == 200:
            bad[key] = result[key]
    print(json.dumps(result, indent=2))
    if bad:
        die(f"containment probe FAILED: {bad}")
    print("NETGUARD_PROBE_OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["up", "down", "status", "probe"])
    ap.add_argument("--work-dir", type=Path, required=True, help="kit work directory (state goes to W/netguard)")
    ap.add_argument("--image", default="python:3.12-slim", help="probe image (default: %(default)s)")
    args = ap.parse_args()
    args.work_dir = args.work_dir.resolve()
    {"up": cmd_up, "down": cmd_down, "status": cmd_status, "probe": cmd_probe}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

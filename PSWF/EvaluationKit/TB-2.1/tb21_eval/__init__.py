"""tb21_eval -- run TerminalBench-2.1 with the PSWF (dynamic-workflow) Terminus-2 harness on one 8xH200 box.

    python -m tb21_eval.prepare   --work-dir W            # TB-2.1 @ pinned commit -> 20,000 s task tree + 89 images
    python -m tb21_eval.netguard  up --work-dir W         # (root) guarded docker network + allowlist egress/DNS
    python -m tb21_eval.serve     up --model M --work-dir W   # 8 x TP1 SGLang + cache-aware router + model guard
    python -m tb21_eval.make_job  --model M --work-dir W --run-dir R   # harbor job config (+ overlay, env file)
    harbor run -c R/job.json --env-file R/harbor.env
    python -m tb21_eval.score     R                       # Mean@5 / pass@5 / per-task table

The constants below define the kit's task, serving, sampling and container protocol.
Changing them changes the evaluation setup.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

PKG_DIR = Path(__file__).resolve().parent
ASSETS = PKG_DIR / "assets"
GUARD_DIR = PKG_DIR / "guard"

# ---------------------------------------------------------------------------------------------------- benchmark
TB21_REPO = "https://github.com/harbor-framework/terminal-bench-2-1.git"
TB21_REV = "7131e4375048a0e408a8fb404b5f499d726b695b"
TB21_SOURCE_TREE_SHA256 = "03d89fb902b06f8bf94fb320c13b19a4fe140fd79e9a70d32e59abf4a32dc888"
# Every task's [agent].timeout_sec raised to 20,000 s (verifier timeouts untouched) -> this tree digest.
AGENT_TIMEOUT_SEC = 20000
TASKS_AA_TREE_SHA256 = "19e1c96f74892c62410b0041f193b42d36b985134bc674f7c05b344a69221ceb"
N_TASKS = 89
N_ATTEMPTS = 5
TASK_NAMES = [t for t in (ASSETS / "tb21_tasks.txt").read_text().split() if t]
assert len(TASK_NAMES) == N_TASKS, len(TASK_NAMES)

# ------------------------------------------------------------------------------------------------------ serving
SERVED_MODEL = "LAR-27B"            # the name on the wire only; it never reaches the prompt
SGLANG_VERSION = "0.5.13.post1"
SGLANG_ROUTER_VERSION = "0.3.2"
TRANSFORMERS_VERSION = "5.8.1"
TORCH_VERSION_PREFIX = "2.11.0"
HOST = "127.0.0.1"                  # engines, router and model guard listen on loopback only
ENGINE_BASE_PORT = 24000            # engine i: --port 24000+i, --nccl-port 25000+i, --dist-init-addr :26000+i
ROUTER_PORT = 31000
MODEL_GUARD_PORT = 8795
CONTEXT_LENGTH = 262144
MAX_OUTPUT_TOKENS = 32768
COMPACTION_MARGIN_TOKENS = 16384
MAX_INPUT_TOKENS = CONTEXT_LENGTH - MAX_OUTPUT_TOKENS - COMPACTION_MARGIN_TOKENS   # 212,992
API_TIMEOUT_SEC = 20000.0

# One TP1 replica per GPU. Arguments are grouped in the order used by serve.py:
# <model/name> ENGINE_PARALLEL_ARGS --nccl-port --dist-init-addr --host --port ENGINE_ARGS
ENGINE_PARALLEL_ARGS = ["--trust-remote-code", "--tp-size", "1", "--dp-size", "1", "--base-gpu-id", "0", "--gpu-id-step", "1"]
ENGINE_ARGS = [
    "--mem-fraction-static", "0.88", "--context-length", str(CONTEXT_LENGTH),
    "--max-running-requests", "12", "--max-mamba-cache-size", "32",
    "--chunked-prefill-size", "32768", "--max-prefill-tokens", "32768",
    "--attention-backend", "flashinfer", "--sampling-backend", "pytorch",
    "--mamba-scheduler-strategy", "extra_buffer", "--enable-cache-report",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder", "--skip-server-warmup",
]
# Router argument order:  --host --port ROUTER_POLICY_ARGS --tokenizer-path <model> ROUTER_BALANCE_ARGS
#                            --prometheus-host --prometheus-port ROUTER_METRICS_PORT
ROUTER_POLICY_ARGS = ["--log-level", "warn", "--policy", "cache_aware"]
ROUTER_BALANCE_ARGS = ["--cache-threshold", "0.5", "--balance-abs-threshold", "3", "--balance-rel-threshold", "1.3"]
ROUTER_METRICS_PORT = 29100

# --------------------------------------------------------------------------------------------- sampling / agent
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}
LLM_CALL_KWARGS = {
    "max_tokens": MAX_OUTPUT_TOKENS,
    "top_p": 0.95,
    "presence_penalty": 0.0,
    "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS},
}
AGENT_IMPORT_PATH = "terminus2_dynwf.dynamic_agent:Terminus2Dynamic"


def agent_kwargs(api_base: str, trusted_root: str) -> dict:
    """Build the agent configuration with local API and trusted-output paths."""
    return {
        "agent_budget_s": float(AGENT_TIMEOUT_SEC),
        "api_base": api_base,
        "compact_reclaim_telemetry": True,
        "fill_profile_cap_tokens": 4000,
        "interleaved_thinking": True,
        "llm_call_kwargs": LLM_CALL_KWARGS,
        "llm_kwargs": {"api_key": "local", "num_retries": 10, "timeout": API_TIMEOUT_SEC},
        "max_turns": 400,
        "model_info": {
            "input_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "mode": "chat",
            "output_cost_per_token": 0.0,
            "supports_reasoning": True,
        },
        "parser_name": "json",
        "phase1_budget_hint": True,
        "phase1_context_max_frac": 0.25,
        "phase1_deadline_grace_s": 420.0,
        "phase1_forced_min_s": 90.0,
        "phase1_frac": 0.27,
        "phase1_hard_stop_frac": 0.7,
        "phase1_hard_stop_turn_frac": 0.75,
        "phase1_max_s": 1944.0,
        "phase1_max_turns": 30,
        "phase1_min_s": 60.0,
        "phase1_single_confirm": True,
        "phase1_spec_reserve_s": 600.0,
        "phase1_spec_reserve_turns": 2,
        "phase1_truncation_stop": True,
        "phase1_wrapup_frac": 0.6,
        "phase1_wrapup_turn_frac": 0.6,
        "phase2_endgame_s": 1800.0,
        "phase2_min_frac": 0.4,
        "phase2_reinject_on_summarize": True,
        "phase2_replan_before_compact": True,
        "phase2_replan_free_tokens": 32768.0,
        "phase2_replan_max_tokens": 8192,
        "phase2_replan_min_free_tokens": 16384.0,
        "phase2_replan_prefer_structured": True,
        "phase2_restate_format": True,
        "phase2_same_chat": False,
        "replan_trusted_dump": True,
        "request_input_budget_guard": True,
        "spec_max_chars": 12000,
        "spec_min_chars": 200,
        "stop_on_policy_violation": True,
        "suppress_max_turns_warning": True,
        "tb_mode": "dynamic",
        "temperature": 1.0,
        "trusted_root": trusted_root,
    }


# ------------------------------------------------------------------------------------------- container / harbor
N_CONCURRENT_TRIALS = 40
CPUS_PER_TRIAL = 4
MEMORY_MB_PER_TRIAL = 40960
TRIAL_RETRY = {"max_retries": 3, "max_wait_sec": 60.0, "min_wait_sec": 1.0, "wait_multiplier": 1.0}

# ------------------------------------------------------------------------------------------------ network guard
GUARD_NETWORK = "pswf_tb21_guarded"
GUARD_BRIDGE = "pswftb21br0"
GUARD_SUBNET = "10.83.0.0/24"
GUARD_GATEWAY = "10.83.0.1"       # egress proxy :8888 and DNS :53 listen here; nothing else on the host is reachable
EGRESS_PROXY_PORT = 8888
DNS_PORT = 53
GUARD_NFT_TABLE = "pswf_tb21_guard"
EGRESS_LIMITS = {"max_active": 64, "max_new_10s": 120, "max_domains": 64}
TOR_MASKS = [
    "/usr/bin/tor", "/usr/sbin/tor", "/usr/bin/torsocks", "/usr/bin/torify", "/usr/bin/torbrowser-launcher",
    "/usr/local/bin/tor", "/usr/local/sbin/tor", "/usr/local/bin/torsocks", "/usr/local/bin/torify",
]
APT_POLICY_MOUNTS = [
    ("99-q38-deny-tor.pref", "/etc/apt/preferences.d/99-q38-deny-tor"),
    ("99-q38-deny-tor-dpkg.conf", "/etc/dpkg/dpkg.cfg.d/99-q38-deny-tor"),
    ("99-q38-prefer-main-suite.pref", "/etc/apt/preferences.d/99-q38-prefer-main-suite"),
]

# ------------------------------------------------------------------------------- identities of the packaged assets and supported model
EXPECTED_SHA256 = {
    "network_policy.md": "8c7bb04ea2891b0cc1d4c5a7360b96f0a6b16a06974c998351f1654649cb3d57",
    "99-q38-deny-tor.pref": "f718ff8770167bf890ad47b06c90b977b5103c246a87f66c81efd5ed834a7c5c",
    "99-q38-deny-tor-dpkg.conf": "2ad14026da9005279bde3065e4d56ee8769cd2668cf629a76af3d8866469235e",
    "99-q38-prefer-main-suite.pref": "73a43b865e923a53bc74a13d63ed505c4aef840a65a16f2024f1abb7170adbe7",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}

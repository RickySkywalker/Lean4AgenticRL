"""Apply the declared harbor patches to the installed harbor, then verify the patched tree and the harness.

    python -m pswf_harbor_patches --tokenizer-dir /path/to/LAR-27B     # apply (idempotent) + verify
    python -m pswf_harbor_patches --check                              # verify only

Run it with the interpreter of the environment that has harbor installed. The patch scripts are executed through
their own CLIs in the release patch order; each one refuses to touch a file whose shape
differs from the pinned harbor revision, and re-running is a no-op. `--tokenizer-dir` is any directory holding the
served model's tokenizer.json (it is hashed into the tokenizer patch's attestation; at run time the agent reads the
path from the Q38_TOKENIZER_JSON environment variable instead).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from pswf_harbor_patches import HARBOR_REV, HARBOR_VERSION, HARNESS_IDENTITY, PATCHED_TREE, PRISTINE_TREE

HERE = Path(__file__).resolve().parent

# (script, arguments) in dependency order. {hp} = installed harbor package dir, {m} = manifest dir.
PATCHES = [
    ("patch_harbor_cleanup.py",
     ["--target", "{hp}/environments/docker/docker.py", "--manifest", "{m}/harbor-cleanup-patch.json"]),
    ("patch_harbor_native_no_network.py",
     ["--target", "{hp}/environments/docker/docker.py", "--manifest", "{m}/harbor-native-no-network-patch.json"]),
    ("patch_harbor_verifier_retry.py",
     ["--target", "{hp}/verifier/verifier.py", "--manifest", "{m}/harbor-verifier-retry-patch.json"]),
    ("patch_harbor_qwen_tokenizer.py",
     ["--harbor-root", "{hp}", "--tokenizer", "{tok}", "--output", "{m}/harbor-qwen-tokenizer-patch.json"]),
    ("patch_harbor_precontainer.py",
     ["--target", "{hp}/environments/docker/docker.py", "--manifest", "{m}/harbor-precontainer-patch.json"]),
    ("patch_harbor_summarizer_fit.py",
     ["--target", "{hp}/agents/terminus_2/terminus_2.py", "--manifest", "{m}/harbor-summarizer-fit-patch.json"]),
]


def tree_digest(root: Path) -> tuple[int, str]:
    """sha256 over the sorted [{path, sha256}] rows of every harbor/**/*.py file (the canonical source identity)."""
    rows = sorted(
        (p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob("*.py")
    )
    canon = json.dumps([{"path": a, "sha256": b} for a, b in rows], sort_keys=True, separators=(",", ":"))
    return len(rows), hashlib.sha256(canon.encode()).hexdigest()


def harness_identity(package_dir: Path) -> str:
    """sha256 over the sorted [relpath, sha256] rows of the agent package's .py/.yaml/.md files."""
    rows = sorted(
        [p.relative_to(package_dir).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()]
        for p in package_dir.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".md") and "__pycache__" not in p.parts
    )
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def installed_harbor() -> Path:
    import harbor

    version = importlib.metadata.version("harbor")
    if version != HARBOR_VERSION:
        raise SystemExit(f"FAIL harbor {version} is installed; the harness needs harbor {HARBOR_VERSION} @ {HARBOR_REV}")
    direct_url = importlib.metadata.distribution("harbor").read_text("direct_url.json")
    commit = (json.loads(direct_url or "{}").get("vcs_info") or {}).get("commit_id")
    if commit and commit != HARBOR_REV:
        raise SystemExit(f"FAIL harbor was installed from commit {commit}; the harness needs {HARBOR_REV}")
    return Path(harbor.__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", "--model", dest="tokenizer_dir", type=Path,
                    help="directory containing the served model's tokenizer.json (required unless --check)")
    ap.add_argument("--manifest-dir", type=Path, default=Path.home() / ".cache" / "pswf" / "harbor-patches",
                    help="where the patch attestations are written (default: %(default)s)")
    ap.add_argument("--check", action="store_true", help="only verify the installed tree, do not patch")
    args = ap.parse_args()

    hp = installed_harbor()
    count, digest = tree_digest(hp)
    if not args.check:
        if (count, digest) not in (PRISTINE_TREE, PATCHED_TREE):
            raise SystemExit(f"FAIL {hp} is neither pristine nor patched harbor {HARBOR_REV} ({count} files, {digest})")
        if args.tokenizer_dir is None or not (args.tokenizer_dir / "tokenizer.json").is_file():
            raise SystemExit("FAIL --tokenizer-dir must point at a directory containing tokenizer.json")
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
        subs = {"hp": str(hp), "m": str(args.manifest_dir), "tok": str(args.tokenizer_dir.resolve())}
        for script, argv in PATCHES:
            cmd = [sys.executable, str(HERE / script)] + [a.format(**subs) for a in argv]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                sys.stderr.write(done.stdout + done.stderr)
                raise SystemExit(f"FAIL {script} exited {done.returncode}")
            print(f"[pswf-harbor-patches] {script}: ok")
        count, digest = tree_digest(hp)

    if (count, digest) != PATCHED_TREE:
        raise SystemExit(f"FAIL patched harbor tree is {count} files / {digest}; expected {PATCHED_TREE[1]}")
    print(f"[pswf-harbor-patches] harbor tree OK: {count} files, sha256 {digest}")

    import terminus2_dynwf

    identity = harness_identity(Path(terminus2_dynwf.__file__).resolve().parent)
    if identity != HARNESS_IDENTITY:
        raise SystemExit(f"FAIL terminus2_dynwf identity {identity}; expected {HARNESS_IDENTITY}")
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    for name in ("_q38_custom_tokenizer", "_q38_count_messages", "_q38_fit_subagent_history"):
        if not hasattr(Terminus2, name):
            raise SystemExit(f"FAIL patched Terminus2 lacks {name}")
    print(f"[pswf-harbor-patches] PSWF harness identity OK: {identity}")
    print("PSWF_HARBOR_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

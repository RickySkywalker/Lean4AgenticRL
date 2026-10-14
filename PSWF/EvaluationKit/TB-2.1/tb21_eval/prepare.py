"""Prepare a kit work directory: the pinned TB-2.1 checkout, the 20,000 s task tree and the 89 task images.

    python -m tb21_eval.prepare --work-dir W [--pull-jobs 12] [--pull-timeout 3600] [--no-pull]

Every step is idempotent. A re-run re-verifies what is already there and does only the work that is missing.

  1. checkout  W/terminal-bench-2-1 is a blobless clone of TB21_REPO, detached at TB21_REV. HEAD must equal
               TB21_REV, and tasks/ must be clean and hash to TB21_SOURCE_TREE_SHA256 (the pinned source tree).
  2. tasks-aa  W/tasks-aa and W/task-manifest.json are produced by the vendored prepare_tasks_aa.py. It raises each
               task's [agent].timeout_sec to AGENT_TIMEOUT_SEC (20,000 s) and changes nothing else. The manifest
               and the tree on disk must match the pinned protocol: TASKS_AA_TREE_SHA256, 89 tasks, 89 raised. If an
               existing tree fails this check, it is derived again.
  3. images    Each W/tasks-aa/*/task.toml [environment].docker_image must equal assets/tb21_images.json. Missing
               images are pulled by tag, in parallel. After that, every image must be present under its tag with
               the pinned digest. Images are never retagged, pulled over or removed. A tag that
               points at a different image is reported, not "fixed".
  4. record    Writes W/prepare.json (commit, tree digests, result of every image check) and prints the line
               TB21_PREPARE_OK. On any failure the command exits 1 with a FATAL message. W/prepare.json only
               exists after a run that fully succeeded.

assets/tb21_images.json pins an image reference and registry digest for each task.
The task_toml_docker_image field pins the task's Docker image declaration.
Docker's containerd store exposes the registry manifest digest as .Id; the
classic store exposes it in .RepoDigests. Either form is accepted.
Keep the work directory and generated manifests outside the source checkout.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):  # tomllib, used here and by the vendored prepare_tasks_aa.py
    raise SystemExit(f"[prepare] FATAL: Python >= 3.11 is required (this is {sys.version.split()[0]})")

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tb21_eval as T
from tb21_eval import prepare_tasks_aa as aa

# ------------------------------------------------------------------------------------------ work-directory layout
CHECKOUT = "terminal-bench-2-1"   # W/<CHECKOUT>: the pinned TB-2.1 git checkout (never edited)
TASKS_AA = "tasks-aa"             # W/<TASKS_AA>: the derived task tree that harbor runs
MANIFEST = "task-manifest.json"   # W/<MANIFEST>: prepare_tasks_aa.py's audit manifest of that derivation
RECORD = "prepare.json"           # W/<RECORD>: written last, and only when every check passed
PULL_LOGS = "logs/pull"           # W/<PULL_LOGS>/<task>.log: output of each `docker pull`
LOCK = ".prepare.lock"            # serializes concurrent runs on the same work directory
RECORD_SCHEMA = "tb21_eval.prepare.v1"

# --------------------------------------------------------------------------------------------- image protocol
IMAGES_ASSET = T.ASSETS / "tb21_images.json"
# Digest of the canonical image-identity JSON: sorted keys, compact separators.
# Registry digests in the packaged manifest fix the expected image set.
EVALUATED_IMAGE_SET_SHA256 = "294686c9da76f1bcb8756d14010b7b8f7dff9e14d02aac44b6089d11ce560315"
EVALUATED_PLATFORM = ("linux", "amd64")    # os/architecture required for every task image

GIT_TIMEOUT_SEC = 3600
INSPECT_JOBS = 8
PULL_ATTEMPTS = 3
PULL_RETRY_DELAY_SEC = 20
# Pull errors that a retry a few seconds later cannot fix (classic and containerd image store wording).
PERMANENT_PULL_ERROR_RE = re.compile(r"toomanyrequests|manifest unknown|pull access denied|unauthorized|"
                                     r"no matching manifest|: not found\s*$", re.IGNORECASE)
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
IMAGE_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")   # <repository>:<tag>

# The vendored script must describe the same benchmark as the package constants.
if aa.EXPECTED_COMMIT != T.TB21_REV or aa.EXPECTED_TASKS != T.N_TASKS:
    raise RuntimeError("tb21_eval/prepare_tasks_aa.py and tb21_eval/__init__.py pin different benchmarks")


class PrepareError(RuntimeError):
    """A fail-closed preparation error. main() reports it as FATAL and exits 1."""


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def run(cmd: list[str], *, timeout: float | None = None, env: dict[str, str] | None = None,
        ) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its text output. A missing executable or a timeout is a PrepareError."""
    try:
        return subprocess.run(cmd, env=env, timeout=timeout, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        raise PrepareError(f"`{cmd[0]}` is not installed or not on PATH") from None
    except subprocess.TimeoutExpired:
        raise PrepareError(f"`{shlex.join(cmd)}` did not finish within {timeout:.0f} s") from None


def git_env() -> dict[str, str]:
    # Never wait for credentials on a terminal: a wrong URL or a network problem must fail instead of hanging.
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def git(repo: Path, *args: str, timeout: float = GIT_TIMEOUT_SEC) -> str:
    done = run(["git", "-C", str(repo), *args], timeout=timeout, env=git_env())
    if done.returncode != 0:
        detail = done.stderr.strip() or done.stdout.strip()
        raise PrepareError(f"`git {' '.join(args)}` failed in {repo}: {detail}")
    return done.stdout.strip()


def tree_sha256(root: Path) -> str:
    """Tree digest exactly as prepare_tasks_aa.py computes it (path, kind, size and sha256 of every entry)."""
    return aa._records_digest(aa._path_records(root))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_set_sha256(identities: dict[str, Any]) -> str:
    """The canonical image-set digest (see EVALUATED_IMAGE_SET_SHA256)."""
    canonical = json.dumps(identities, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def familiar(ref: str) -> str:
    """Docker Hub references with or without the implicit registry prefix compare equal."""
    for prefix in ("docker.io/", "index.docker.io/"):
        if ref.startswith(prefix):
            return ref[len(prefix):]
    return ref


def repository(image: str) -> str:
    return image.rsplit(":", 1)[0]


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@contextmanager
def work_dir_lock(work_dir: Path) -> Iterator[None]:
    with open(work_dir / LOCK, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PrepareError(f"another `tb21_eval.prepare` is already running on {work_dir}") from None
        yield


# ------------------------------------------------------------------------------------------------ 1. checkout
def clone(repo: Path) -> None:
    """Blobless clone of TB21_REPO detached at TB21_REV. It is built next to `repo` and moved there once complete."""
    partial = repo.with_name(f".{repo.name}.partial")
    if partial.exists():                    # left behind by an interrupted clone of ours
        shutil.rmtree(partial)
    log(f"checkout: cloning {T.TB21_REPO} -> {repo}")
    # core.autocrlf=false keeps the checkout byte-exact even if the user's global git config converts line endings.
    done = run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", "--config", "core.autocrlf=false",
                T.TB21_REPO, str(partial)], timeout=GIT_TIMEOUT_SEC, env=git_env())
    if done.returncode != 0:
        raise PrepareError(f"git clone {T.TB21_REPO} failed: {done.stderr.strip()}")
    if run(["git", "-C", str(partial), "cat-file", "-e", f"{T.TB21_REV}^{{commit}}"], env=git_env()).returncode:
        # If no branch reaches the pinned commit any more, fetch it directly by its id.
        git(partial, "fetch", "--quiet", "--filter=blob:none", "origin", T.TB21_REV)
    git(partial, "checkout", "--quiet", "--detach", T.TB21_REV)
    partial.rename(repo)


def ensure_checkout(work_dir: Path) -> dict[str, Any]:
    """Step 1: W/terminal-bench-2-1 at TB21_REV, with a pristine tasks/ tree."""
    repo = work_dir / CHECKOUT
    if repo.exists():
        log(f"checkout: verifying existing {repo}")
    else:
        clone(repo)
    try:
        toplevel = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    except PrepareError as exc:
        raise PrepareError(f"{repo} exists but is not a usable git checkout ({exc}): remove it and re-run") from None
    if toplevel != repo.resolve():
        raise PrepareError(f"{repo} is not a git checkout of its own (it lies inside {toplevel}): "
                           "remove it and re-run")
    head = git(repo, "rev-parse", "HEAD")
    if head != T.TB21_REV:
        raise PrepareError(f"{repo} is at {head}, not the pinned revision {T.TB21_REV}: run "
                           f"`git -C {repo} checkout --detach {T.TB21_REV}` (or remove {repo}) and re-run")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", "tasks").splitlines()
    if dirty:
        preview = "; ".join(line.strip() for line in dirty[:5]) + (" ..." if len(dirty) > 5 else "")
        raise PrepareError(f"{repo}/tasks has local changes ({preview}): remove {repo} and re-run")
    # The hash also catches what `git status` cannot see, e.g. ignored files or a converting checkout.
    source_sha = tree_sha256(repo / "tasks")
    if source_sha != T.TB21_SOURCE_TREE_SHA256:
        raise PrepareError(f"{repo}/tasks hashes to {source_sha}, not to the pinned source tree "
                           f"{T.TB21_SOURCE_TREE_SHA256} (extra ignored files?): remove {repo} and re-run")
    origin = run(["git", "-C", str(repo), "config", "--get", "remote.origin.url"]).stdout.strip()
    log(f"checkout: HEAD {head}, tasks/ clean, source tree sha256 {source_sha}")
    return {"repo": T.TB21_REPO, "origin_url": origin or None, "rev": T.TB21_REV, "head": head,
            "path": str(repo), "tasks_dir": str(repo / "tasks"), "source_tree_sha256": source_sha}


# ------------------------------------------------------------------------------------------------ 2. tasks-aa
def tasks_aa_problem(src: Path, dst: Path, manifest_path: Path) -> str | None:
    """Return why W/tasks-aa is not (or is no longer) the pinned task tree, or None if it is."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"{manifest_path} is missing"
    except (OSError, ValueError) as exc:
        return f"{manifest_path} is unreadable ({exc})"
    if not isinstance(manifest, dict):
        return f"{manifest_path} is not a JSON object"
    expected = {
        "schema_version": aa.SCHEMA_VERSION,
        "expected_commit": T.TB21_REV,
        "actual_commit": T.TB21_REV,
        "source": str(src.resolve()),           # prepare_tasks_aa.py records resolved paths
        "destination": str(dst.resolve()),
        "agent_timeout_floor_sec": T.AGENT_TIMEOUT_SEC,
        "task_count": T.N_TASKS,
        "raised_count": T.N_TASKS,          # every official agent timeout (at most 12,000 s) is below 20,000 s
        "kept_count": 0,
        "source_tree_sha256": T.TB21_SOURCE_TREE_SHA256,
        "derived_tree_sha256": T.TASKS_AA_TREE_SHA256,
    }
    for key, want in expected.items():
        if manifest.get(key) != want:
            return f"manifest {key} is {manifest.get(key)!r}, expected {want!r}"
    rows = manifest.get("tasks")
    if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
            or sorted(str(row.get("task")) for row in rows) != sorted(T.TASK_NAMES)):
        return f"manifest task list differs from the {T.N_TASKS} TB-2.1 tasks"
    if any(row.get("aa_agent_timeout_sec") != T.AGENT_TIMEOUT_SEC for row in rows):
        return f"manifest has an agent timeout other than {T.AGENT_TIMEOUT_SEC} s"
    # The manifest is only a claim. The tree on disk must still hash to it.
    derived_sha = tree_sha256(dst)
    if derived_sha != T.TASKS_AA_TREE_SHA256:
        return f"{dst} hashes to {derived_sha}, not {T.TASKS_AA_TREE_SHA256} (modified after it was derived)"
    return None


def derive_tasks_aa(work_dir: Path, src: Path, dst: Path, manifest_path: Path, *, replace_existing: bool) -> None:
    # Same interpreter, isolated mode (-I): the vendored script needs only the standard library, so nothing from
    # the environment or from this package's directory can shadow it.
    cmd = [sys.executable, "-I", str(T.PKG_DIR / "prepare_tasks_aa.py"), "--src", str(src), "--dst", str(dst),
           "--manifest", str(manifest_path), "--floor-sec", str(T.AGENT_TIMEOUT_SEC)]
    if replace_existing:
        cmd.append("--replace-existing")
    log(f"tasks-aa: {shlex.join(cmd)}")
    done = subprocess.run(cmd, cwd=work_dir, check=False)    # prints its own summary or error
    if done.returncode != 0:
        raise PrepareError(f"prepare_tasks_aa.py failed with exit code {done.returncode} (its error is shown above)")


def ensure_tasks_aa(work_dir: Path, src: Path) -> dict[str, Any]:
    """Step 2: W/tasks-aa, derived from the checkout and verified against the pinned protocol."""
    dst, manifest_path = work_dir / TASKS_AA, work_dir / MANIFEST
    if dst.is_symlink() or (dst.exists() and not dst.is_dir()):
        # Re-deriving would replace whatever the link points at, possibly outside W: leave that to the user.
        raise PrepareError(f"{dst} exists but is not a plain directory: remove it and re-run")
    if dst.exists():
        problem = tasks_aa_problem(src, dst, manifest_path)
        if problem is None:
            action = "verified"
        else:
            log(f"tasks-aa: existing {dst} does not verify ({problem}); deriving it again")
            derive_tasks_aa(work_dir, src, dst, manifest_path, replace_existing=True)
            action = "re-derived"
    else:
        derive_tasks_aa(work_dir, src, dst, manifest_path, replace_existing=False)
        action = "derived"
    if action != "verified":
        problem = tasks_aa_problem(src, dst, manifest_path)
        if problem is not None:
            raise PrepareError(f"the derived task tree does not match the expected one: {problem}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log(f"tasks-aa: {action} {dst}: {manifest['task_count']} tasks, {manifest['raised_count']} agent timeouts "
        f"raised to {T.AGENT_TIMEOUT_SEC} s, tree sha256 {manifest['derived_tree_sha256']}")
    return {"path": str(dst), "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "action": action, "task_count": manifest["task_count"], "raised_count": manifest["raised_count"],
            "kept_count": manifest["kept_count"], "agent_timeout_sec": T.AGENT_TIMEOUT_SEC,
            "source_tree_sha256": manifest["source_tree_sha256"],
            "derived_tree_sha256": manifest["derived_tree_sha256"]}


# -------------------------------------------------------------------------------------------------- 3. images
def load_images_asset() -> dict[str, dict[str, str]]:
    """Load assets/tb21_images.json and check it against the task list and the pinned image-set identity."""
    try:
        asset = json.loads(IMAGES_ASSET.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PrepareError(f"cannot read {IMAGES_ASSET}: {exc}") from None
    if not isinstance(asset, dict) or sorted(asset) != sorted(T.TASK_NAMES):
        raise PrepareError(f"{IMAGES_ASSET} must have exactly one entry for each of the {T.N_TASKS} TB-2.1 tasks")
    for task, entry in asset.items():
        image = entry.get("image") if isinstance(entry, dict) else None
        digest = entry.get("id") if isinstance(entry, dict) else None
        if not (isinstance(image, str) and IMAGE_RE.fullmatch(image)
                and isinstance(digest, str) and DIGEST_RE.fullmatch(digest)
                and entry.get("task_toml_docker_image") == image):
            raise PrepareError(f"{IMAGES_ASSET}: malformed entry for {task}: {entry!r}")
    # Rebuild the provenance record's identity document from the asset. Any edit to an image or id breaks it.
    expected = {e["image"]: {"architecture": EVALUATED_PLATFORM[1], "id": e["id"], "os": EVALUATED_PLATFORM[0],
                             "repo_digests": [f"{repository(e['image'])}@{e['id']}"]} for e in asset.values()}
    if len(expected) != T.N_TASKS or image_set_sha256(expected) != EVALUATED_IMAGE_SET_SHA256:
        raise PrepareError(f"{IMAGES_ASSET} does not match the pinned image-set identity "
                           f"{EVALUATED_IMAGE_SET_SHA256}: the file was modified")
    return asset


def task_images(tasks_dir: Path) -> dict[str, str]:
    """[environment].docker_image of every task.toml in the derived tree."""
    images: dict[str, str] = {}
    for toml_path in sorted(tasks_dir.glob("*/task.toml")):
        try:
            environment = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("environment") or {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PrepareError(f"cannot parse {toml_path}: {exc}") from None
        image = environment.get("docker_image")
        if not isinstance(image, str) or not image:
            raise PrepareError(f"{toml_path} has no [environment].docker_image")
        images[toml_path.parent.name] = image
    if sorted(images) != sorted(T.TASK_NAMES):
        raise PrepareError(f"{tasks_dir} does not hold exactly the {T.N_TASKS} TB-2.1 tasks")
    return images


def docker_server() -> dict[str, str]:
    """The Docker daemon's version and image store. Fails if the daemon cannot be reached."""
    done = run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=120)
    if done.returncode != 0:
        raise PrepareError(f"cannot talk to the Docker daemon ({done.stderr.strip()}): is it running, "
                           "and is this user allowed to use it?")
    store = "unknown"                       # informational only
    try:
        info = run(["docker", "info", "--format", "{{json .DriverStatus}}"], timeout=120)
        if info.returncode == 0:
            store = "containerd" if "io.containerd.snapshotter" in info.stdout else "classic"
    except PrepareError:
        pass
    return {"server_version": done.stdout.strip(), "image_store": store}


def inspect_image(image: str) -> dict[str, Any] | None:
    """`docker image inspect` of an exact tag, or None if no image carries that tag."""
    done = run(["docker", "image", "inspect", image], timeout=300)
    if done.returncode != 0:
        if "no such image" in done.stderr.lower():
            return None
        raise PrepareError(f"`docker image inspect {image}` failed: {done.stderr.strip()}")
    try:
        return json.loads(done.stdout)[0]
    except (ValueError, IndexError, TypeError) as exc:
        raise PrepareError(f"unexpected `docker image inspect {image}` output: {exc}") from None


def inspect_all(images: list[str]) -> dict[str, dict[str, Any] | None]:
    with ThreadPoolExecutor(max_workers=INSPECT_JOBS) as pool:
        return dict(zip(images, pool.map(inspect_image, images)))


def check_image(image: str, expected_id: str, item: dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    """Compare a local image with the expected one. Returns (facts for the record, problem or None)."""
    if item is None:
        return {"present": False}, "not present"
    repo_digests = sorted(item.get("RepoDigests") or [])
    facts: dict[str, Any] = {"present": True, "local_id": item.get("Id"), "repo_digests": repo_digests,
                             "os": item.get("Os"), "architecture": item.get("Architecture")}
    if image not in {familiar(tag) for tag in item.get("RepoTags") or []}:
        return facts, "the inspected image does not carry this tag"
    if item.get("Id") == expected_id:                                     # containerd image store
        facts["matched_by"] = "id"
    elif f"{repository(image)}@{expected_id}" in {familiar(d) for d in repo_digests}:   # classic image store
        facts["matched_by"] = "repo_digest"
    else:
        listed = ", ".join(repo_digests) or "none"
        return facts, f"the local tag points at {item.get('Id')} (repo digests: {listed}), not at {expected_id}"
    if (facts["os"], facts["architecture"]) != EVALUATED_PLATFORM:
        return facts, (f"platform is {facts['os']}/{facts['architecture']}; "
                       f"the kit requires {'/'.join(EVALUATED_PLATFORM)}")
    return facts, None


def pull_image(image: str, log_path: Path, budget_sec: float) -> tuple[bool, str, float]:
    """`docker pull <image>` by tag, retrying transient failures within budget_sec. Returns (ok, error, seconds)."""
    start = time.monotonic()
    deadline = start + budget_sec
    error = ""
    with log_path.open("w", encoding="utf-8") as handle:
        for attempt in range(1, PULL_ATTEMPTS + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            handle.write(f"$ docker pull {image}    # attempt {attempt}/{PULL_ATTEMPTS}\n")
            handle.flush()
            try:
                done = subprocess.run(["docker", "pull", image], stdout=handle, stderr=subprocess.STDOUT,
                                      timeout=remaining, check=False)
            except subprocess.TimeoutExpired:
                handle.write(f"# timed out after {budget_sec:.0f} s\n")
                return False, f"timed out after {budget_sec:.0f} s", time.monotonic() - start
            if done.returncode == 0:
                return True, "", time.monotonic() - start
            handle.flush()
            lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                     if line.strip()]
            error = lines[-1].strip() if lines else f"docker pull exited with {done.returncode}"
            if PERMANENT_PULL_ERROR_RE.search(error):
                break
            if attempt < PULL_ATTEMPTS:
                time.sleep(max(0.0, min(PULL_RETRY_DELAY_SEC * attempt, deadline - time.monotonic())))
    if "toomanyrequests" in error.lower():
        error += " -- Docker Hub's pull rate limit: `docker login` (or wait), then re-run"
    return False, error or f"no time left within the {budget_sec:.0f} s budget", time.monotonic() - start


def pull_missing(work_dir: Path, missing: dict[str, str], jobs: int, budget_sec: float) -> dict[str, str]:
    """Pull the missing images (task -> image) in parallel. Returns task -> error message for every failed pull."""
    log_dir = work_dir / PULL_LOGS
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = min(jobs, len(missing))
    log(f"images: pulling {len(missing)} missing image(s), {workers} at a time (logs: {log_dir}/<task>.log)")
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(pull_image, image, log_dir / f"{task}.log", budget_sec): task
                   for task, image in missing.items()}
        for n, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            ok, error, seconds = future.result()
            if ok:
                log(f"images: [{n}/{len(missing)}] pulled {missing[task]} ({seconds:.0f} s)")
            else:
                failures[task] = error
                log(f"images: [{n}/{len(missing)}] FAILED to pull {missing[task]}: {error}")
    return failures


def ensure_images(work_dir: Path, tasks_dir: Path, *, pull: bool, pull_jobs: int,
                  pull_timeout: float) -> dict[str, Any]:
    """Step 3: every task image present under its tag, with the pinned digest."""
    asset = load_images_asset()
    declared = task_images(tasks_dir)
    disagree = [f"{task}: task.toml has {declared[task]}, the kit requires {asset[task]['image']}"
                for task in T.TASK_NAMES if declared[task] != asset[task]["image"]]
    if disagree:
        raise PrepareError(f"task.toml docker_image differs from {IMAGES_ASSET.name}:\n  " + "\n  ".join(disagree))
    log(f"images: the {T.N_TASKS} task.toml docker_image references equal {IMAGES_ASSET.name}")

    server = docker_server()
    images = [asset[task]["image"] for task in T.TASK_NAMES]
    found = inspect_all(images)
    missing = {task: asset[task]["image"] for task in T.TASK_NAMES if found[asset[task]["image"]] is None}
    log(f"images: {T.N_TASKS - len(missing)}/{T.N_TASKS} present locally (docker {server['server_version']}, "
        f"{server['image_store']} image store)")
    pull_failures: dict[str, str] = {}
    if missing and pull:
        pull_failures = pull_missing(work_dir, missing, pull_jobs, pull_timeout)
        found = inspect_all(images)          # verify everything again, freshly pulled or not

    records: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    identities: dict[str, Any] = {}
    for task in T.TASK_NAMES:
        image, expected_id = asset[task]["image"], asset[task]["id"]
        item = found[image]
        facts, problem = check_image(image, expected_id, item)
        records[task] = {"image": image, "id": expected_id, "task_toml_docker_image": declared[task],
                         "source": "pulled" if task in missing else "already present", **facts}
        if problem is not None:
            if task in pull_failures:
                problem = f"pull failed: {pull_failures[task]}"
            elif task in missing and not pull:
                problem += " (--no-pull was given: re-run without it to pull the image)"
            elif task in missing and facts.get("present") and "matched_by" not in facts:
                problem += " (just pulled by tag: the registry's tag no longer points at the pinned image)"
            problems.append(f"{image}: {problem}")
        if item is not None:
            identities[image] = {"architecture": item.get("Architecture"), "id": item.get("Id"),
                                 "os": item.get("Os"), "repo_digests": sorted(item.get("RepoDigests") or [])}
    if problems:
        wrong_tag = any(r.get("present") and "matched_by" not in r for r in records.values())
        raise PrepareError(f"{len(problems)} of {T.N_TASKS} task images are not the expected ones:\n  "
                           + "\n  ".join(problems)
                           + ("\n(prepare never retags, overwrites or removes images: fix the listed tags yourself)"
                              if wrong_tag else ""))

    # Informational: on a containerd image store that holds exactly these images, this equals the provenance value.
    identity = image_set_sha256(identities)
    pulled = sorted(missing.values())
    same = identity == EVALUATED_IMAGE_SET_SHA256
    verdict = "equals the pinned image-set digest" if same else "differs from the pinned image-set digest (normal off containerd)"
    log(f"images: {T.N_TASKS}/{T.N_TASKS} verified ({T.N_TASKS - len(pulled)} already present, {len(pulled)} "
        f"pulled); image-set identity {identity} {verdict}")
    return {"asset": str(IMAGES_ASSET), "asset_sha256": file_sha256(IMAGES_ASSET),
            "docker_server_version": server["server_version"], "image_store": server["image_store"],
            "count": T.N_TASKS, "verified": T.N_TASKS, "already_present": T.N_TASKS - len(pulled), "pulled": pulled,
            "image_set_sha256": identity, "image_set_matches_evaluated_run": same,
            "tasks": records}


# ------------------------------------------------------------------------------------------------------- main
def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tb21_eval.prepare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="kit work directory (created if missing); everything this command writes goes below it")
    ap.add_argument("--pull-jobs", type=positive_int, default=12,
                    help="docker pulls to run in parallel (default: %(default)s)")
    ap.add_argument("--pull-timeout", type=positive_float, default=3600.0,
                    help="time budget per image pull in seconds, retries included (default: %(default)s)")
    ap.add_argument("--no-pull", action="store_true",
                    help="never pull: only verify, and fail if a task image is missing")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = args.work_dir.expanduser().resolve()
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        with work_dir_lock(work_dir):
            record_path = work_dir / RECORD
            record_path.unlink(missing_ok=True)    # a stale success record must not survive a failing re-run
            checkout = ensure_checkout(work_dir)
            tasks_aa = ensure_tasks_aa(work_dir, Path(checkout["tasks_dir"]))
            images = ensure_images(work_dir, Path(tasks_aa["path"]), pull=not args.no_pull,
                                   pull_jobs=args.pull_jobs, pull_timeout=args.pull_timeout)
            record = {
                "schema_version": RECORD_SCHEMA,
                "ok": True,
                "kit_version": T.__version__,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "work_dir": str(work_dir),
                "python": sys.version.split()[0],
                "checkout": checkout,
                "tasks_aa": tasks_aa,
                "images": images,
            }
            write_json_atomic(record_path, record)
    except PrepareError as exc:
        print(f"[prepare] FATAL: {exc}", file=sys.stderr, flush=True)
        return 1
    except OSError as exc:
        print(f"[prepare] FATAL: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("[prepare] interrupted; re-run to resume (finished steps are verified, not redone)", file=sys.stderr)
        return 130
    print(f"TB21_PREPARE_OK rev={checkout['head']} tasks_aa_sha256={tasks_aa['derived_tree_sha256']} "
          f"images={images['verified']}/{T.N_TASKS} (pulled {len(images['pulled'])}) record={record_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

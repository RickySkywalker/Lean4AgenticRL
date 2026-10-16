"""Score TB-2.1 harbor job directories: Mean@k, pass@k and per-task results.

    python -m tb21_eval.score DIR [DIR ...] [--k 5] [--policy kit|stock] [--expected-tasks FILE] [--json OUT.json]

Keep input runs and optional JSON output outside the source checkout.

Inputs
------
Each DIR is a harbor job directory or a run directory containing jobs/<job>/.
Jobs in a run directory are ordered by their job-level started_at timestamp; missing
timestamps sort last, with directory names breaking ties. Explicit job directories
are used in argument order. These jobs form the scoring rounds.

A trial directory contains config.json and, when finished, result.json. Trials are
deduplicated by trial name, keeping the first occurrence in round order.

Classification
--------------
Policy kit classifies each trial as:
* SCORED: a finite reward, a valid nonempty pytest CTRF report, and no exception other
  than an agent timeout or an output-parser/tool-call error.
* ZERO: a verifier timeout without a reward after pytest started.
* INFRA: missing or invalid verification evidence, other harbor errors, unfinished
  trials, or an episode ended by a model-guard refusal. A Hugging Face transport
  failure in the reshard-c4-data session fixture is also infrastructure failure.

Unfinished trials in a running job stay in progress. Policy stock uses harbor's
recorded rewards, counting a missing reward as zero; unfinished trials still block
selection. Both policies are shown so the treatment of infrastructure is explicit.

Selection
---------
For each task, use the first round containing at least k trials and no INFRA trial
for that task. Select the first k trials by trial name, discarding earlier rounds
for that task. Tasks without a complete round remain pending. Recovery commands
rerun all k attempts of each pending task.

Metrics
-------
Mean@k is passing attempts divided by expected tasks times k, with pending slots
counted as zero. pass@k is the share of tasks with a pass. Mean@3 uses the first
three selected attempts by started_at, with trial names breaking ties.

Reports include a Wilson interval, solve histogram, per-task table, outcome counts
and duration quantiles. An explicit --reference FILE adds a paired task-cluster
bootstrap comparison with user-supplied per-task vectors. No reference data is bundled.

Exit status: 0 = complete; 1 = pending tasks; 2 = invalid input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from tb21_eval import N_ATTEMPTS, N_CONCURRENT_TRIALS, TASK_NAMES, __version__

# --------------------------------------------------------------------------------------------------- protocol rules
POLICIES = ("kit", "stock")
SCORED, ZERO, INFRA = "SCORED", "ZERO", "INFRA"

# Harbor names trial dirs "<task[:32]>__<7-character ShortUUID>"; nothing at job level matches this.
TRIAL_DIR_RE = re.compile(r"^.+__[A-Za-z0-9]{4,}$")
# Files harbor writes into every job dir when the job starts.
JOB_FILES = ("config.json", "lock.json", "result.json", "job.log")

AGENT_TIMEOUT = "AgentTimeoutError"
VERIFIER_TIMEOUT = "VerifierTimeoutError"
# An exception leaves a trial gradeable when "<exception_type> <exception_message>" (lower-cased) contains one of
# these. Each means the agent ran out of budget or emitted unparseable output, and harbor still ran the verifier.
SCORED_EXCEPTION_PATTERNS = (
    "agenttimeouterror", "agent execution timed out", "outputparser", "parsererror", "toolcall",
)
# pytest's own banners (80-column section rules). Any of them in verifier/test-stdout.txt (matched
# case-insensitively) means test.sh got as far as running the tests.
PYTEST_START_MARKERS = ("test session starts", "=" * 35 + " FAILURES " + "=" * 35, "=" * 36 + " ERRORS " + "=" * 36)

# The reshard-c4-data session fixture downloads a C4 shard from Hugging Face.
# A transport failure in that fixture is infrastructure failure rather than a model failure.
C4_TASK_NAME = "terminal-bench/reshard-c4-data"
C4_TEST_NAME = "test_outputs.py::test_compress_decompress_workflow"
C4_HIDDEN_SHARD = "en/c4-train.00009-of-01024.json.gz"
C4_SETUP_TRACE_MARKERS = (
    '@pytest.fixture(scope="session")',
    "def generate_test_data():",
    "dataset = load_dataset(",
    '"allenai/c4"',
    C4_HIDDEN_SHARD,
    "Captured stdout setup",
    "Generating test data.",
    "Captured stderr setup",
)
C4_SETUP_TRANSPORT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hf_cas_middleware",
     re.compile(r"file reconstruction error:\s*cas client error:\s*request middleware error", re.IGNORECASE)),
    ("hf_xet_transport",
     re.compile(r"(?:huggingface|hf_hub|cdn-lfs|xet|cas client|cas-bridge)"
                r"[^\n]{0,240}(?:error|failed|forbidden|timed? out|offline|"
                r"not found|connection|middleware)", re.IGNORECASE)),
)

PASS_THRESHOLD = 1.0       # TB-2.1 rewards are binary: a pass is reward >= 1
FIRST_N = 3                # reward-blind Mean@3: the first 3 selected trials of a task by started_at
WILSON_Z = 1.96
BOOTSTRAP_DRAWS = 4000     # paired task-cluster bootstrap resamples
BOOTSTRAP_SEED = 20260914     # fixed seed for deterministic comparisons
MAX_LISTED_TRIALS = 40     # flagged trials printed on stdout (the JSON report lists all of them)

OUTCOME_LABELS = {
    "pass": "pass",
    "fail": "fail",
    "agent_timeout": "agent timeout",
    "verifier_timeout_zero": "verifier timeout (0)",
    "other_exception_zero": "other exception (0)",  # stock policy only: the kit calls these INFRA
    "infra": "infra",
}
_EPOCH = dt.datetime.min.replace(tzinfo=dt.UTC)


class ScoreError(Exception):
    """Invalid input. main() reports it on stderr and exits with status 2."""


# ------------------------------------------------------------------------------------------------------ data model
@dataclass
class Trial:
    """What the scorer reads from one trial dir. Verdicts are derived per policy by classify()."""

    name: str
    task: str
    path: Path
    round: int                         # 1-based index of the round (job dir) that holds the trial
    finished: bool                     # a result.json with a trial_name exists
    problem: str = ""                  # unfinished/unreadable trials: why
    reward: float | None = None        # verifier_result.rewards.reward, if it is a finite number
    exception: str | None = None       # exception_info.exception_type
    exception_message: str = ""
    started_at: dt.datetime | None = None
    duration_s: float | None = None    # started_at -> finished_at
    agent_s: float | None = None       # the agent_execution phase
    ctrf_problem: str | None = None    # None: verifier/ctrf.json is a non-empty pytest-json-ctrf report
    pytest_started: bool = False       # only evaluated for verifier timeouts
    safety_stop: str | None = None     # the refused rule when the model guard ended the episode (dynwf_meta.json)
    c4_transport: str | None = None    # reshard-c4-data fixture transport failure (matched signal codes)
    pswf: dict | None = None           # PSWF phase summary from agent/dynwf_meta.json (None: no such file)
    running: bool = False              # unfinished trial of a job that is still running


@dataclass(frozen=True)
class Verdict:
    status: str           # SCORED | ZERO | INFRA
    score: float | None   # the reward that enters the metrics (None for INFRA)
    reason: str


@dataclass
class Round:
    index: int
    job_dir: Path
    job_name: str
    started_at: str | None
    finished_at: str | None
    trials: list[Trial] = field(default_factory=list)


@dataclass
class TaskSelection:
    task: str
    round: int | None = None                                       # None: PENDING
    chosen: list[tuple[Trial, Verdict]] = field(default_factory=list)  # k trials, trial-name order
    notes: list[str] = field(default_factory=list)                 # why earlier rounds were not used


# ------------------------------------------------------------------------------------------------- reading trials
def read_text(path: Path, errors: str = "strict") -> str:
    """A file's text. A permission problem aborts the run: unreadable evidence must not be scored as INFRA."""
    try:
        return path.read_text(encoding="utf-8", errors=errors)
    except PermissionError as exc:
        raise ScoreError(f"cannot read {path}: permission denied (run the scorer as the owner of the trial "
                         "files)") from exc


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def parse_time(value: Any) -> dt.datetime | None:
    """A harbor ISO-8601 timestamp as an aware UTC datetime (naive values are taken as UTC); None if absent."""
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.UTC)


def phase_seconds(node: Any) -> float | None:
    """Length in seconds of a harbor ``{started_at, finished_at}`` record."""
    if not isinstance(node, dict):
        return None
    start, end = parse_time(node.get("started_at")), parse_time(node.get("finished_at"))
    return (end - start).total_seconds() if start and end else None


def ctrf_problem(trial_dir: Path) -> str | None:
    """None if verifier/ctrf.json is a pytest-json-ctrf report with >= 1 test, else what is wrong with it."""
    path = trial_dir / "verifier" / "ctrf.json"
    if not path.is_file():
        return "verifier/ctrf.json missing"
    try:
        doc = read_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        return f"verifier/ctrf.json unreadable: {type(exc).__name__}"
    results = doc.get("results") if isinstance(doc, dict) else None
    summary = results.get("summary") if isinstance(results, dict) else None
    tests = results.get("tests") if isinstance(results, dict) else None
    n_tests = summary.get("tests") if isinstance(summary, dict) else None
    if (type(n_tests) is not int or not isinstance(tests, list)
            or not all(isinstance(test, dict) for test in tests)):
        return "verifier/ctrf.json is not a pytest-json-ctrf report"
    if n_tests < 1 or not tests:
        return "verifier/ctrf.json reports no tests"
    return None


def pytest_started(trial_dir: Path) -> bool:
    """Whether verifier/test-stdout.txt shows that pytest started (see PYTEST_START_MARKERS)."""
    path = trial_dir / "verifier" / "test-stdout.txt"
    if not path.is_file():
        return False
    text = read_text(path, errors="replace").lower()
    return any(marker.lower() in text for marker in PYTEST_START_MARKERS)


def c4_transport_error(trial_dir: Path, task_name: str | None) -> str | None:
    """Matched signal codes if this is the reshard-c4-data session-fixture transport failure, else None."""
    if task_name != C4_TASK_NAME:
        return None
    try:
        doc = read_json(trial_dir / "verifier" / "ctrf.json")
        tests = doc["results"]["tests"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None
    for test in tests if isinstance(tests, list) else []:
        if (not isinstance(test, dict) or test.get("name") != C4_TEST_NAME
                or str(test.get("status") or "").lower() not in {"failed", "error"}):
            continue
        trace = test.get("trace")
        if not isinstance(trace, str) or "AssertionError" in trace:
            continue
        if not trace.lstrip().startswith(C4_SETUP_TRACE_MARKERS[0]) or any(
                marker not in trace for marker in C4_SETUP_TRACE_MARKERS[1:]):
            continue
        matched = [code for code, pattern in C4_SETUP_TRANSPORT_RULES if pattern.search(trace)]
        if matched:
            return ",".join(matched)
    return None


def dynwf_meta(trial_dir: Path) -> dict | None:
    """The PSWF agent's per-trial metadata (agent/dynwf_meta.json), or None."""
    path = trial_dir / "agent" / "dynwf_meta.json"
    if not path.is_file():
        return None
    try:
        meta = read_json(path)
    except (OSError, UnicodeError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def safety_stop_of(trial_dir: Path) -> str | None:
    """The rule the model guard refused, if the PSWF agent ended this episode on a guard refusal; else None."""
    meta = dynwf_meta(trial_dir)
    stop = meta.get("safety_stop") if meta else None
    if not stop:
        return None
    return str(stop.get("kind") or "policy") if isinstance(stop, dict) else "policy"


def exception_is_scored(trial: Trial) -> bool:
    """Agent timeouts and output-parser/tool-call errors leave a trial gradeable (see SCORED_EXCEPTION_PATTERNS)."""
    text = f"{trial.exception} {trial.exception_message}".lower()
    return any(pattern in text for pattern in SCORED_EXCEPTION_PATTERNS)


def _finished_trial(trial_dir: Path, round_index: int, doc: dict) -> Trial:
    task_name = doc.get("task_name")
    task = task_name.split("/")[-1] if isinstance(task_name, str) and task_name else trial_dir.name.rsplit("__", 1)[0]
    verifier = doc.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    raw = rewards.get("reward") if isinstance(rewards, dict) else None
    # type() rather than isinstance(): a JSON true/false is not a reward.
    reward = float(raw) if type(raw) in (int, float) and math.isfinite(raw) else None
    info = doc.get("exception_info")
    exception, message = None, ""
    if isinstance(info, dict) and info:
        exception = str(info.get("exception_type") or "UnknownException")
        message = str(info.get("exception_message") or "")
    elif info:
        exception, message = "UnknownException", str(info)
    trial = Trial(
        name=doc["trial_name"], task=task, path=trial_dir, round=round_index, finished=True, reward=reward,
        exception=exception, exception_message=message, started_at=parse_time(doc.get("started_at")),
        duration_s=phase_seconds(doc), agent_s=phase_seconds(doc.get("agent_execution")),
        ctrf_problem=ctrf_problem(trial_dir),
    )
    if exception == VERIFIER_TIMEOUT:
        trial.pytest_started = pytest_started(trial_dir)
    trial.safety_stop = safety_stop_of(trial_dir)
    meta = dynwf_meta(trial_dir)
    if meta is not None:
        trial.pswf = {"phase1": meta.get("phase1_stopped_by"), "have_spec": bool(meta.get("have_spec")),
                      "injected": bool(meta.get("phase2_spec_injected"))}
    trial.c4_transport = c4_transport_error(trial_dir, task_name if isinstance(task_name, str) else None)
    return trial


def _unfinished_trial(trial_dir: Path, round_index: int, problem: str) -> Trial:
    """A trial without a usable result.json. Its identity comes from config.json, because harbor truncates task
    names to 32 characters in trial dir names."""
    name, task = trial_dir.name, trial_dir.name.rsplit("__", 1)[0]
    try:
        config = read_json(trial_dir / "config.json")
    except (OSError, UnicodeError, ValueError):
        config = None
    if isinstance(config, dict):
        name = str(config.get("trial_name") or name)
        spec = config.get("task") if isinstance(config.get("task"), dict) else {}
        if spec.get("path"):
            task = PurePosixPath(str(spec["path"])).name
        elif spec.get("name"):
            task = str(spec["name"]).split("/")[-1]
    return Trial(name=name, task=task, path=trial_dir, round=round_index, finished=False, problem=problem)


def load_trial(trial_dir: Path, round_index: int) -> Trial | None:
    """Read one ``<name>__<suffix>/`` dir; None if it holds neither result.json nor config.json (not a trial)."""
    result_path = trial_dir / "result.json"
    if result_path.is_file():
        try:
            doc = read_json(result_path)
        except (OSError, UnicodeError, ValueError) as exc:
            return _unfinished_trial(trial_dir, round_index, f"result.json unreadable: {type(exc).__name__}")
        if not isinstance(doc, dict) or not isinstance(doc.get("trial_name"), str) or not doc["trial_name"]:
            return _unfinished_trial(trial_dir, round_index, "result.json without trial_name")
        return _finished_trial(trial_dir, round_index, doc)
    if (trial_dir / "config.json").is_file():
        return _unfinished_trial(trial_dir, round_index, "unfinished (no result.json yet)")
    return None


# ------------------------------------------------------------------------------------------------ finding rounds
def _is_trial_dir(path: Path) -> bool:
    return (TRIAL_DIR_RE.match(path.name) is not None and path.is_dir()
            and ((path / "result.json").is_file() or (path / "config.json").is_file()))


def _is_job_dir(path: Path) -> bool:
    return any((path / name).is_file() for name in JOB_FILES) or any(_is_trial_dir(c) for c in path.iterdir())


def _is_trial_result(path: Path) -> bool:
    """Whether `path` is a trial-level result.json (job-level ones carry no trial_name)."""
    try:
        doc = read_json(path)
    except (OSError, UnicodeError, ValueError):
        return False
    return isinstance(doc, dict) and isinstance(doc.get("trial_name"), str)


def _job_meta(job_dir: Path) -> tuple[str, str | None, str | None]:
    """(job_name, started_at, finished_at) from harbor's job-level config.json and result.json."""
    name, started, finished = job_dir.name, None, None
    try:
        config = read_json(job_dir / "config.json")
    except (OSError, UnicodeError, ValueError):
        config = None
    if isinstance(config, dict) and isinstance(config.get("job_name"), str) and config["job_name"]:
        name = config["job_name"]
    try:
        result = read_json(job_dir / "result.json")
    except (OSError, UnicodeError, ValueError):
        result = None
    if isinstance(result, dict):
        started = result.get("started_at") if isinstance(result.get("started_at"), str) else None
        finished = result.get("finished_at") if isinstance(result.get("finished_at"), str) else None
    return name, started, finished


def _job_order_key(job_dir: Path) -> tuple[bool, dt.datetime, str]:
    started = parse_time(_job_meta(job_dir)[1])
    return started is None, started or _EPOCH, job_dir.name


def job_dirs_of(arg: Path, warnings: list[str]) -> list[Path]:
    """Expand one positional DIR into its job dir(s), in round order (see the module docstring)."""
    if not arg.is_dir():
        raise ScoreError(f"{arg}: no such directory")
    jobs = arg / "jobs"
    if jobs.is_dir():
        found = []
        for child in sorted(p for p in jobs.iterdir() if p.is_dir()):
            if _is_job_dir(child):
                found.append(child)
            else:
                warnings.append(f"{child}: empty directory under jobs/, skipped")
        if not found:
            raise ScoreError(f"{arg}: run dir without any harbor job dir under jobs/")
        return sorted(found, key=_job_order_key)
    if TRIAL_DIR_RE.match(arg.name) and _is_trial_result(arg / "result.json"):
        raise ScoreError(f"{arg}: this is a single trial dir; pass the job dir that contains it")
    if _is_job_dir(arg):
        return [arg]
    if any(p.is_dir() and _is_job_dir(p) for p in arg.iterdir()):
        raise ScoreError(f"{arg}: looks like a harbor jobs_dir; pass the run dir above it, or the job dirs inside it")
    raise ScoreError(f"{arg}: neither a harbor job dir (<task>__<suffix>/ trial dirs) nor a run dir (jobs/<job>/)")


def load_rounds(dirs: list[Path], warnings: list[str]) -> list[Round]:
    """Every job dir behind `dirs` as one round, with its trials (de-duplicated by trial name across rounds)."""
    job_dirs = [job for arg in dirs for job in job_dirs_of(arg, warnings)]
    seen: dict[Path, Path] = {}
    for job in job_dirs:
        key = job.resolve()
        if key in seen:
            raise ScoreError(f"job dir {job} is given twice (also as {seen[key]})")
        seen[key] = job
    rounds: list[Round] = []
    owners: dict[str, Path] = {}  # trial name -> the dir it was first read from
    for index, job_dir in enumerate(job_dirs, start=1):
        rnd = Round(index, job_dir, *_job_meta(job_dir))
        for child in sorted(job_dir.iterdir()):
            if not (child.is_dir() and TRIAL_DIR_RE.match(child.name)):
                continue
            trial = load_trial(child, index)
            if trial is None:
                warnings.append(f"{child}: neither result.json nor config.json, ignored")
                continue
            if trial.name in owners:
                warnings.append(f"{child}: duplicate of trial {trial.name} (first read from {owners[trial.name]}), "
                                "ignored")
                continue
            if trial.finished and trial.name != child.name:
                warnings.append(f"{child}: result.json belongs to trial {trial.name}")
            owners[trial.name] = child
            if not trial.finished and rnd.finished_at is None:
                trial.running = True
                trial.problem = "running (the job has not finished)"
            rnd.trials.append(trial)
        rounds.append(rnd)
    return rounds


# ------------------------------------------------------------------------------------- classification / selection
def _exception_text(trial: Trial) -> str:
    first_line = trial.exception_message.strip().splitlines()[0][:100] if trial.exception_message.strip() else ""
    return f"{trial.exception}: {first_line}" if first_line else str(trial.exception)


def classify(trial: Trial, policy: str) -> Verdict:
    """The verdict on one trial under `policy` (see the module docstring)."""
    if not trial.finished:
        return Verdict(INFRA, None, trial.problem)
    if policy == "stock":
        if trial.reward is not None:
            return Verdict(SCORED, trial.reward, "recorded reward")
        cause = f" ({_exception_text(trial)})" if trial.exception else ""
        return Verdict(SCORED, 0.0, f"no reward recorded{cause}: counted 0")
    if trial.safety_stop:
        return Verdict(INFRA, None, f"episode ended by a model-guard refusal ({trial.safety_stop})")
    if trial.exception == VERIFIER_TIMEOUT:
        if trial.reward is not None:
            return Verdict(INFRA, None, "VerifierTimeoutError with a recorded reward")
        if trial.pytest_started:
            return Verdict(ZERO, 0.0, "VerifierTimeoutError after pytest started: scored 0")
        return Verdict(INFRA, None, "VerifierTimeoutError before pytest started")
    if trial.exception is not None and not exception_is_scored(trial):
        return Verdict(INFRA, None, f"harbor exception {_exception_text(trial)}")
    if trial.reward is None:
        return Verdict(INFRA, None, "no numeric verifier_result.rewards.reward")
    if trial.ctrf_problem:
        return Verdict(INFRA, None, f"{trial.ctrf_problem} (recorded reward {trial.reward:g})")
    if trial.c4_transport:
        return Verdict(INFRA, None, f"reshard-c4-data fixture could not fetch its data ({trial.c4_transport})")
    return Verdict(SCORED, trial.reward, trial.exception or "graded")


def outcome(trial: Trial, verdict: Verdict) -> str:
    """Outcome category of a trial (a key of OUTCOME_LABELS)."""
    if verdict.status == INFRA:
        return "infra"
    if verdict.score is not None and verdict.score >= PASS_THRESHOLD:
        return "pass"
    if trial.exception == AGENT_TIMEOUT:
        return "agent_timeout"
    if trial.exception == VERIFIER_TIMEOUT:
        return "verifier_timeout_zero"
    if trial.exception is None or exception_is_scored(trial):
        return "fail"
    return "other_exception_zero"


def select(rounds: list[Round], expected: list[str], k: int, policy: str) -> dict[str, TaskSelection]:
    """Reward-blind selection: per task, the first round with >= k trials of it and none of them INFRA."""
    by_task: dict[str, dict[int, list[tuple[Trial, Verdict]]]] = defaultdict(lambda: defaultdict(list))
    for rnd in rounds:
        for trial in rnd.trials:
            by_task[trial.task][rnd.index].append((trial, classify(trial, policy)))
    selection: dict[str, TaskSelection] = {}
    for task in expected:
        sel = TaskSelection(task)
        for rnd in rounds:
            entries = by_task[task].get(rnd.index, [])
            if not entries:
                continue
            infra = [verdict.reason for _, verdict in entries if verdict.status == INFRA]
            if infra:
                why = "; ".join(f"{n}x {reason}" if n > 1 else reason for reason, n in sorted(Counter(infra).items()))
                sel.notes.append(f"round {rnd.index}: {len(infra)} of {len(entries)} trials INFRA: {why}")
                continue
            if len(entries) < k:
                sel.notes.append(f"round {rnd.index}: only {len(entries)} of {k} trials")
                continue
            entries = sorted(entries, key=lambda entry: entry[0].name)
            if len(entries) > k:
                sel.notes.append(f"round {rnd.index}: {len(entries)} trials, the first {k} by name are used")
            sel.round, sel.chosen = rnd.index, entries[:k]
            break
        selection[task] = sel
    return selection


# ---------------------------------------------------------------------------------------------------- statistics
def wilson(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion, as fractions."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def nearest_rank(sorted_values: list[float], percent: int) -> float:
    """The nearest-rank percentile: the smallest value with at least `percent` % of the sample at or below it."""
    rank = -(-percent * len(sorted_values) // 100)  # ceil(percent/100 * n) in exact integer arithmetic
    return sorted_values[max(0, rank - 1)]


def distribution(values: list[float | None]) -> dict | None:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    return {"n": len(data), "mean": sum(data) / len(data), "p50": nearest_rank(data, 50),
            "p90": nearest_rank(data, 90), "max": data[-1], "sum": sum(data)}


def _n_pass(entries: list[tuple[Trial, Verdict]]) -> int:
    return sum(1 for _, verdict in entries if verdict.score is not None and verdict.score >= PASS_THRESHOLD)


def _started_key(entry: tuple[Trial, Verdict]) -> tuple[bool, dt.datetime, str]:
    trial = entry[0]
    return trial.started_at is None, trial.started_at or _EPOCH, trial.name


def reading(selection: dict[str, TaskSelection], expected: list[str], k: int, policy: str) -> dict:
    """The headline numbers of one policy's selection (PENDING slots zero-filled)."""
    chosen = [sel for sel in selection.values() if sel.round is not None]
    pending = [task for task in expected if selection[task].round is None]
    n_tasks, n_trials = len(expected), len(expected) * k
    passes = sum(_n_pass(sel.chosen) for sel in chosen)
    solved = sum(1 for sel in chosen if _n_pass(sel.chosen) > 0)
    out = {
        "policy": policy, "complete": not pending, "pending": pending,
        "passes": passes, "trials": n_trials, "mean_at_k": passes / n_trials, "wilson95": wilson(passes, n_trials),
        "reward_sum": sum(verdict.score for sel in chosen for _, verdict in sel.chosen),
        "solved_tasks": solved, "tasks": n_tasks, "pass_at_k": solved / n_tasks,
        "pass_at_k_wilson95": wilson(solved, n_tasks),
        "selected_from_round": dict(sorted(Counter(sel.round for sel in chosen).items())),
    }
    if k >= FIRST_N:
        first = sum(_n_pass(sorted(sel.chosen, key=_started_key)[:FIRST_N]) for sel in chosen)
        out.update(mean_at_3=first / (n_tasks * FIRST_N), mean_at_3_passes=first,
                   mean_at_3_trials=n_tasks * FIRST_N, mean_at_3_wilson95=wilson(first, n_tasks * FIRST_N))
    return out


# ------------------------------------------------------------------------------------------ reference comparison
def load_reference(path: Path) -> dict[str, list[int]]:
    """Validate a user-supplied per-task vector file ``{task: [0/1, ...]}``."""
    try:
        doc = read_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ScoreError(f"reference {path}: cannot read ({exc})") from exc
    if not isinstance(doc, dict) or not doc:
        raise ScoreError(f"reference {path}: expected a non-empty JSON object {{task: [0/1, ...]}}")
    vectors: dict[str, list[int]] = {}
    for task, vector in doc.items():
        if not (isinstance(vector, list) and vector and all(type(x) in (int, float) and x in (0, 1) for x in vector)):
            raise ScoreError(f"reference {path}: {task}: expected a non-empty list of 0/1 rewards")
        vectors[task] = [int(x) for x in vector]
    return vectors


def compare(selection: dict[str, TaskSelection], expected: list[str], k: int,
            reference: dict[str, list[int]], label: str) -> dict | None:
    """Paired comparison with a reference: per-task pass-rate differences, task-cluster bootstrap 95 % CI.

    Sort shared tasks before resampling so the fixed seed gives deterministic intervals.
    Return None when the reference shares no tasks with the requested task set.
    """
    tasks = sorted(set(expected) & set(reference))
    if not tasks:
        return None
    ours = {task: _n_pass(selection[task].chosen) for task in tasks}   # PENDING -> 0, as in Mean@k
    theirs = {task: sum(reference[task]) for task in tasks}
    diffs = [ours[t] / k - theirs[t] / len(reference[t]) for t in tasks]
    n = len(tasks)
    rng = random.Random(BOOTSTRAP_SEED)
    boots = sorted(100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_DRAWS))
    same_k = all(len(reference[task]) == k for task in tasks)
    return {
        "reference": label,
        "tasks": n,
        "tasks_not_in_reference": sorted(set(expected) - set(reference)),
        "passes": sum(ours.values()), "trials": n * k,
        "reference_passes": sum(theirs.values()), "reference_trials": sum(len(reference[t]) for t in tasks),
        "delta_passes": sum(ours.values()) - sum(theirs.values()) if same_k else None,
        "delta_pp": 100.0 * sum(diffs) / n,
        "bootstrap95_pp": [boots[int(0.025 * BOOTSTRAP_DRAWS)], boots[int(0.975 * BOOTSTRAP_DRAWS) - 1]],
        "bootstrap_draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
        "tasks_up": sum(1 for d in diffs if d > 0), "tasks_down": sum(1 for d in diffs if d < 0),
        "tasks_equal": sum(1 for d in diffs if d == 0),
        "identical_vectors": all(selection[t].round is not None and _vector(selection[t].chosen) == reference[t]
                                 for t in tasks),
        "provisional": any(selection[t].round is None for t in tasks),
    }


# ------------------------------------------------------------------------------------------------ re-run commands
_ROUND_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_r(?P<nn>\d+)$")


def _run_dir_of(job_dir: Path) -> Path | None:
    """The make_job run dir R of a job dir R/jobs/<job>, if the job dir sits in that layout."""
    return job_dir.parent.parent if job_dir.parent.name == "jobs" else None


def _make_job_inputs(run_dir: Path | None) -> tuple[str, str]:
    """(--model, --work-dir) of the main job, read back from R/harbor.env and R/job.json; placeholders if absent."""
    model, work = "<model-dir>", "<work-dir>"
    if run_dir is None:
        return model, work
    try:
        for line in (run_dir / "harbor.env").read_text().splitlines():
            if line.startswith("Q38_TOKENIZER_JSON="):
                model = str(Path(line.split("=", 1)[1]).parent)
        job = json.loads((run_dir / "job.json").read_text())
        work = str(Path(job["datasets"][0]["path"]).parent)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    return model, work


def _venv_bin(name: str) -> str:
    """`name` next to the running interpreter (the Quickstart never activates the virtual environment)."""
    return str(Path(sys.executable).parent / name)


def rerun_plan(pending: list[str], rounds: list[Round], inputs: list[str]) -> dict:
    """Commands for the next recovery round in the layout of tb21_eval.make_job: a new job holding only the PENDING
    tasks (all k attempts), written as R/job.<name>_rNN.json and run into R/jobs/<name>_rNN/. R is the run dir of
    the last round, so the new job also sorts last when R is re-scored (job dirs in a run dir go by start time)."""
    first = rounds[0].job_name if rounds else "pswf-tb21"
    match = _ROUND_SUFFIX_RE.match(first)
    base = match.group("base") if match else first
    taken = {rnd.job_name for rnd in rounds}
    for rnd in rounds:  # never suggest a job name that already exists next to one of the given jobs
        taken.update(p.name for p in rnd.job_dir.parent.iterdir() if p.is_dir())
    numbers = [int(m.group("nn")) for name in taken
               if (m := _ROUND_SUFFIX_RE.match(name)) and m.group("base") == base]
    job_name = f"{base}_r{max(numbers, default=0) + 1:02d}"

    run_dir = _run_dir_of(rounds[-1].job_dir) if rounds else None
    run = str(run_dir) if run_dir else "<run-dir>"
    # Re-scoring a run dir picks the new job up by itself; explicitly listed job dirs need it appended.
    covered = run_dir is not None and any(Path(arg).resolve() == run_dir.resolve() for arg in inputs)
    rescore = inputs if covered else [*inputs, f"{run}/jobs/{job_name}"]

    python, harbor = sys.executable, _venv_bin("harbor")
    model, work = _make_job_inputs(run_dir)
    resume = []  # finished jobs holding unfinished trials (harbor died): `harbor job resume` finishes them in place
    for rnd in rounds:
        if rnd.finished_at is None and any(t.running for t in rnd.trials):
            continue
        if any(not t.finished and t.task in pending for t in rnd.trials):
            env = f"{_run_dir_of(rnd.job_dir) or '<run-dir>'}/harbor.env"
            # `job resume` has no --env-file, so the env file is exported into its environment explicitly.
            resume.append(f"(set -a; . {env}; set +a; {harbor} job resume --job-path {rnd.job_dir})")
    return {
        "tasks": pending,
        "job_name": job_name,
        "commands": [
            f"{python} -m tb21_eval.make_job --model {model} --work-dir {work} --run-dir {run} "
            f"--tasks {','.join(pending)} --job-name {job_name}",
            f"{harbor} run -c {run}/job.{job_name}.json --env-file {run}/harbor.env",
            f"{python} -m tb21_eval.score " + " ".join(rescore),
        ],
        "unfinished_trials": sorted({t.name for rnd in rounds for t in rnd.trials
                                     if not t.finished and t.task in pending}),
        "resume_commands": resume,
    }


def split_pending(pending: list[str], rounds: list[Round], k: int, policy: str) -> tuple[list[str], list[str]]:
    """(in progress, needs a recovery job). A PENDING task is in progress while the last round holding it is still
    running and has not produced a finished INFRA trial of it (or holds none of it yet); otherwise it needs a
    recovery job."""
    running = {rnd.index for rnd in rounds if rnd.finished_at is None}
    in_progress, rerun = [], []
    for task in pending:
        holding = [rnd for rnd in rounds if any(t.task == task for t in rnd.trials)]
        if not holding:
            (in_progress if running else rerun).append(task)
            continue
        last = holding[-1]
        trials = [t for t in last.trials if t.task == task]
        broken = any(t.finished and classify(t, policy).status == INFRA for t in trials)
        if last.index in running and not broken:
            in_progress.append(task)
        else:
            rerun.append(task)
    return in_progress, rerun


# ---------------------------------------------------------------------------------------------------- the report
def _vector(entries: list[tuple[Trial, Verdict]]) -> list[int]:
    return [1 if verdict.score is not None and verdict.score >= PASS_THRESHOLD else 0 for _, verdict in entries]


def _iso(stamp: dt.datetime | None) -> str | None:
    return stamp.isoformat().replace("+00:00", "Z") if stamp else None


def parse_task_file(path: Path) -> list[str]:
    """Task names separated by whitespace or newlines; '#' starts a comment."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScoreError(f"--expected-tasks {path}: {exc}") from exc
    tasks = [name for line in text.splitlines() for name in line.split("#", 1)[0].split()]
    if not tasks:
        raise ScoreError(f"--expected-tasks {path}: no task names")
    return tasks


def score(dirs: list[str | Path], *, k: int = N_ATTEMPTS, policy: str = "kit", expected: list[str] | None = None,
          reference: Path | None = None) -> dict:
    """Score the rounds behind `dirs` (launch order). Returns the full JSON-serialisable report."""
    if k < 1:
        raise ScoreError(f"k must be >= 1 (got {k})")
    if policy not in POLICIES:
        raise ScoreError(f"unknown policy {policy!r} (choose from {', '.join(POLICIES)})")
    expected = list(TASK_NAMES if expected is None else expected)
    if not expected:
        raise ScoreError("no expected tasks")
    duplicated = sorted(task for task, n in Counter(expected).items() if n > 1)
    if duplicated:
        raise ScoreError(f"expected tasks listed twice: {', '.join(duplicated)}")
    if not dirs:
        raise ScoreError("no job or run dir given")

    warnings: list[str] = []
    rounds = load_rounds([Path(d) for d in dirs], warnings)
    all_trials = [trial for rnd in rounds for trial in rnd.trials]
    if not all_trials:
        warnings.append("no trial dirs found in any round")
    expected_set = set(expected)
    unexpected = Counter(trial.task for trial in all_trials if trial.task not in expected_set)
    if unexpected:
        listed = ", ".join(f"{task} ({n})" for task, n in sorted(unexpected.items()))
        warnings.append(f"{sum(unexpected.values())} trial(s) of tasks outside the expected set ignored: {listed}")

    selections = {p: select(rounds, expected, k, p) for p in POLICIES}
    readings = {p: reading(selections[p], expected, k, p) for p in POLICIES}
    primary, headline = selections[policy], readings[policy]
    selected_names = {trial.name for sel in primary.values() for trial, _ in sel.chosen}

    per_task = {}
    for task in expected:
        sel = primary[task]
        done = sel.round is not None
        per_task[task] = {
            "round": sel.round,
            "passes": _n_pass(sel.chosen) if done else None,
            "vector": _vector(sel.chosen) if done else None,
            "rewards": [verdict.score for _, verdict in sel.chosen] if done else None,
            "trials": [trial.name for trial, _ in sel.chosen],
            "agent_timeouts": sum(1 for t, v in sel.chosen if outcome(t, v) == "agent_timeout"),
            "verifier_timeout_zeros": sum(1 for t, v in sel.chosen if outcome(t, v) == "verifier_timeout_zero"),
            "notes": sel.notes,
        }

    trials_out, flagged = [], []
    outcomes = {"selected": Counter(), "all": Counter()}
    for trial in sorted(all_trials, key=lambda t: (t.round, t.name)):
        verdicts = {p: classify(trial, p) for p in POLICIES}
        mine = verdicts[policy]
        if trial.task in expected_set:
            outcomes["all"][outcome(trial, mine)] += 1
            if trial.name in selected_names:
                outcomes["selected"][outcome(trial, mine)] += 1
            if mine.status != SCORED or trial.reward is None:
                flagged.append({"round": trial.round, "trial": trial.name, "status": mine.status,
                                "reason": mine.reason, "dir": str(trial.path)})
        trials_out.append({
            "round": trial.round, "trial": trial.name, "task": trial.task, "dir": str(trial.path),
            "finished": trial.finished, "problem": trial.problem or None, "reward": trial.reward,
            "exception": trial.exception, "started_at": _iso(trial.started_at), "duration_s": trial.duration_s,
            "agent_execution_s": trial.agent_s, "ctrf_problem": trial.ctrf_problem,
            "pytest_started": trial.pytest_started if trial.exception == VERIFIER_TIMEOUT else None,
            "selected": trial.name in selected_names,
            **{p: {"status": v.status, "score": v.score, "reason": v.reason, "outcome": outcome(trial, v)}
               for p, v in verdicts.items()},
        })

    chosen_trials = [trial for sel in primary.values() for trial, _ in sel.chosen]
    finished_trials = [trial for trial in all_trials if trial.finished and trial.task in expected_set]
    comparison = None
    if reference is not None:
        comparison = compare(primary, expected, k, load_reference(Path(reference)), str(reference))
        if comparison is None:
            warnings.append(f"reference {reference} shares no task with the expected tasks: comparison skipped")
    pending = headline["pending"]
    in_progress, needs_rerun = split_pending(pending, rounds, k, policy)
    with_meta = [t.pswf for t in finished_trials if t.pswf is not None]
    pswf = {"finished_trials": len(finished_trials), "with_meta": len(with_meta),
            "plan_injected": sum(m["injected"] for m in with_meta),
            "phase1_stopped_by": dict(Counter(str(m["phase1"]) for m in with_meta))} if finished_trials else None
    return {
        "tool": "tb21_eval.score", "version": __version__,
        "policy": policy, "k": k, "complete": headline["complete"],
        "inputs": [str(d) for d in dirs], "expected_tasks": len(expected),
        "rounds": [{
            "round": rnd.index, "job_dir": str(rnd.job_dir), "job_name": rnd.job_name,
            "started_at": rnd.started_at, "finished_at": rnd.finished_at, "trials": len(rnd.trials),
            "status": {s: sum(1 for t in rnd.trials if classify(t, policy).status == s) for s in (SCORED, ZERO, INFRA)},
            "tasks_selected": headline["selected_from_round"].get(rnd.index, 0),
        } for rnd in rounds],
        "headline": headline,
        "readings": readings,
        "per_task": per_task,
        # Same shape as the reference file (selected tasks only), so the two can be diffed directly.
        "per_task_vector": {task: row["vector"] for task, row in per_task.items() if row["vector"] is not None},
        "solve_histogram": {str(n): sum(1 for row in per_task.values() if row["passes"] == n) for n in range(k + 1)},
        "outcomes": {scope: {key: counts.get(key, 0) for key in OUTCOME_LABELS} for scope, counts in outcomes.items()},
        "durations_s": {
            "selected": {"trial": distribution([t.duration_s for t in chosen_trials]),
                         "agent_execution": distribution([t.agent_s for t in chosen_trials])},
            "all_finished": {"trial": distribution([t.duration_s for t in finished_trials]),
                             "agent_execution": distribution([t.agent_s for t in finished_trials])},
        },
        "comparison": comparison,
        "pswf": pswf,
        "pending": pending,
        "in_progress": in_progress,
        "running_jobs": [rnd.job_name for rnd in rounds if rnd.finished_at is None],
        "rerun": rerun_plan(needs_rerun, rounds, [str(d) for d in dirs]) if needs_rerun else None,
        "flagged_trials": flagged,
        "warnings": warnings,
        "trials": trials_out,
    }


# ----------------------------------------------------------------------------------------------- text rendering
def _table(headers: list[str], rows: list[list[str]], right: frozenset[int] | set[int] = frozenset(),
           indent: str = "  ") -> list[str]:
    """Rows of left-aligned text columns (the columns in `right` right-aligned), two spaces apart."""
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]

    def line(cells: list[str]) -> str:
        return indent + "  ".join(cell.rjust(w) if i in right else cell.ljust(w)
                                  for i, (cell, w) in enumerate(zip(cells, widths))).rstrip()

    return [line(headers), *map(line, rows)]


def _pct(fraction: float) -> str:
    return f"{100 * fraction:.2f}%"


def _ci(bounds: list[float] | tuple[float, float]) -> str:
    return f"[{100 * bounds[0]:.2f}, {100 * bounds[1]:.2f}]"


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} = {_pct(numerator / denominator)}"


def _vector_text(rewards: list[float]) -> str:
    if all(r in (0.0, 1.0) for r in rewards):
        return "".join("1" if r >= PASS_THRESHOLD else "0" for r in rewards)
    return " ".join(f"{r:g}" for r in rewards)


def render(rep: dict) -> str:
    """The human-readable form of a report from score()."""
    k, policy, head = rep["k"], rep["policy"], rep["headline"]
    other = next(p for p in POLICIES if p != policy)
    out = [f"TB-2.1 score (tb21_eval {rep['version']}): policy {policy}, k = {k}, "
           f"{rep['expected_tasks']} expected tasks, {len(rep['rounds'])} round(s)", ""]

    out.append("Rounds, in launch order (a task's k trials all come from its first infra-clean round)")
    out += _table(
        ["round", "job", "trials", "SCORED", "ZERO", "INFRA", "tasks used", "job finished", "dir"],
        [[str(r["round"]), r["job_name"], str(r["trials"]), str(r["status"][SCORED]), str(r["status"][ZERO]),
          str(r["status"][INFRA]), str(r["tasks_selected"]), "yes" if r["finished_at"] else "no", r["job_dir"]]
         for r in rep["rounds"]],
        right={0, 2, 3, 4, 5, 6})

    if rep["warnings"]:
        out += ["", "Warnings"] + [f"  ! {w}" for w in rep["warnings"]]

    flagged = rep["flagged_trials"]
    if flagged:
        out += ["", f"Trials not counted at their recorded reward under policy {policy} ({len(flagged)})"]
        out += _table(["round", "trial", "status", "reason"],
                      [[str(f["round"]), f["trial"], f["status"], f["reason"]] for f in flagged[:MAX_LISTED_TRIALS]],
                      right={0})
        if len(flagged) > MAX_LISTED_TRIALS:
            out.append(f"  ... and {len(flagged) - MAX_LISTED_TRIALS} more (see --json)")

    out += ["", f"Per-task results (vector = rewards of the {k} selected trials in trial-name order)"]
    rows = []
    for i, (task, row) in enumerate(rep["per_task"].items(), start=1):
        notes = []
        if row["agent_timeouts"]:
            notes.append(f"{row['agent_timeouts']} agent timeout(s)")
        if row["verifier_timeout_zeros"]:
            notes.append(f"{row['verifier_timeout_zeros']} verifier timeout(s) scored 0")
        notes += row["notes"]
        if row["round"] is None:
            rows.append([str(i), task, "-", "PENDING", "-", "; ".join(notes) or "no trials"])
        else:
            rows.append([str(i), task, f"{row['passes']}/{k}", _vector_text(row["rewards"]), str(row["round"]),
                         "; ".join(notes)])
    out += _table(["#", "task", f"k/{k}", "vector", "round", "notes"], rows, right={0, 4})

    histogram = "   ".join(f"{n}: {count}" for n, count in rep["solve_histogram"].items())
    out += ["", f"Solve histogram (tasks by passes out of {k}; PENDING tasks excluded)", f"  {histogram}"]

    out += ["", "Outcomes"]
    counts = rep["outcomes"]
    keys = [key for key in OUTCOME_LABELS if key != "other_exception_zero" or counts["all"][key]]
    out += _table(["", "selected", "all rounds"],
                  [[OUTCOME_LABELS[key], str(counts["selected"][key]), str(counts["all"][key])] for key in keys]
                  + [["total", str(sum(counts["selected"].values())), str(sum(counts["all"].values()))]],
                  right={1, 2})

    selected = rep["durations_s"]["selected"]
    if selected["trial"]:
        n = selected["trial"]["n"]
        out += ["", f"Trial durations of the {n} selected trials (seconds; nearest-rank quantiles)"]
        out += _table(["", "mean", "p50", "p90", "max"],
                      [[label, *(f"{stats[key]:,.0f}" for key in ("mean", "p50", "p90", "max"))]
                       for label, stats in (("whole trial (started_at -> finished_at)", selected["trial"]),
                                            ("agent execution", selected["agent_execution"])) if stats],
                      right={1, 2, 3, 4})
        every = rep["durations_s"]["all_finished"]["trial"]
        out.append(f"  compute: {selected['trial']['sum'] / 3600:,.1f} trial-hours selected, "
                   f"{every['sum'] / 3600:,.1f} over all {every['n']} finished trials")

    pswf = rep.get("pswf")
    if pswf and pswf["with_meta"]:
        ended = ", ".join(f"{k}: {v}" for k, v in sorted(pswf["phase1_stopped_by"].items()))
        out += ["", f"PSWF workflow: the phase-1 plan was injected into phase 2 in {pswf['plan_injected']}/"
                f"{pswf['with_meta']} finished trials (phase 1 ended by {ended})"]

    out += ["", f"Readings ({policy} is the selected policy; stock = harbor semantics, no infra re-runs)"]
    columns = [policy, other]
    reading_rows = [
        [f"Mean@{k}"] + [_ratio(rep["readings"][p]["passes"], rep["readings"][p]["trials"]) for p in columns],
        ["  Wilson 95% CI"] + [_ci(rep["readings"][p]["wilson95"]) for p in columns],
        [f"pass@{k}"] + [_ratio(rep["readings"][p]["solved_tasks"], rep["readings"][p]["tasks"]) for p in columns],
    ]
    if "mean_at_3" in head:
        reading_rows.append(["Mean@3 (first 3 by started_at)"] + [
            _ratio(rep["readings"][p]["mean_at_3_passes"], rep["readings"][p]["mean_at_3_trials"]) for p in columns])
    reading_rows.append(["PENDING tasks"] + [str(len(rep["readings"][p]["pending"])) for p in columns])
    out += _table(["", *columns], reading_rows)

    cmp = rep["comparison"]
    if cmp:
        ref = _ratio(cmp["reference_passes"], cmp["reference_trials"])
        delta = f"{cmp['delta_passes']:+d} passes = " if cmp["delta_passes"] is not None else ""
        lo, hi = cmp["bootstrap95_pp"]
        out += ["", f"Paired comparison with the reference ({ref}; {cmp['tasks']} common tasks)",
                f"  {cmp['reference']}",
                f"  difference: {delta}{cmp['delta_pp']:+.2f} pp, paired task-cluster bootstrap 95% CI "
                f"[{lo:+.2f}, {hi:+.2f}] pp ({cmp['bootstrap_draws']} draws, seed {cmp['seed']})",
                f"  tasks better / worse / equal: {cmp['tasks_up']} / {cmp['tasks_down']} / {cmp['tasks_equal']}; "
                f"per-task vectors identical to the reference: {'yes' if cmp['identical_vectors'] else 'no'}"]
        if cmp["provisional"]:
            out.append("  (provisional: PENDING tasks enter the comparison zero-filled)")
        if cmp["tasks_not_in_reference"]:
            out.append(f"  not in the reference: {', '.join(cmp['tasks_not_in_reference'])}")

    out.append("")
    mean = f"Mean@{k} = {_pct(head['mean_at_k'])} ({head['passes']}/{head['trials']})"
    if rep["complete"]:
        tail = f"; Mean@3 = {_pct(head['mean_at_3'])}" if "mean_at_3" in head else ""
        out.append(f"FINAL ({policy}): {mean}, Wilson 95% CI {_ci(head['wilson95'])}; "
                   f"pass@{k} = {_ratio(head['solved_tasks'], head['tasks'])}{tail}")
    else:
        plan = rep["rerun"]
        out += [f"NOT FINAL ({policy}): {len(rep['pending'])} of {rep['expected_tasks']} tasks PENDING; "
                f"{mean} so far, with the PENDING slots zero-filled"]
        if rep["in_progress"]:
            out += [f"  IN PROGRESS ({len(rep['in_progress'])} tasks): job(s) {', '.join(rep['running_jobs'])} still "
                    "running; score again when harbor has finished"]
        if plan:
            out += [f"  NEEDS A RECOVERY JOB ({len(plan['tasks'])} tasks): {', '.join(plan['tasks'])}",
                    "",
                    f"Re-run these tasks as a new recovery job (all {k} attempts each), then re-score with every round "
                    "in launch order:",
                    *(f"  {command}" for command in plan["commands"])]
            if rep["running_jobs"]:
                out.append("  (the main job is still running: start the recovery job once fewer than "
                           f"{N_CONCURRENT_TRIALS - k} of its trials are running, so at most {N_CONCURRENT_TRIALS} "
                           "containers run at once)")
            if plan["unfinished_trials"]:
                out += ["",
                        f"Note: {len(plan['unfinished_trials'])} trial(s) of these tasks have no result.json although "
                        "their job has ended (harbor died?). Before resuming, remove the dead",
                        "job's containers (docker ps --filter network=pswf_tb21_guarded), then finish the trials in "
                        "place (finished trials are kept):",
                        *(f"  {command}" for command in plan["resume_commands"])]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tb21_eval.score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Score a TB-2.1 evaluation from harbor job dirs: Mean@k with Wilson 95% CI, pass@k, reward-blind\n"
            "Mean@3, solve histogram, per-task table, and an optional comparison with a user-supplied reference.\n\n"
            "Trials are SCORED, ZERO (verifier timeout after pytest started: 0) or INFRA (missing/invalid CTRF,\n"
            "other harbor exceptions, unfinished). Per task, the first round in the given order that holds k\n"
            "trials of it and no INFRA one is used; tasks without one are PENDING and must be re-run as a new\n"
            "recovery round. See the module docstring for the full rules."),
        epilog=(
            "examples:\n"
            "  python -m tb21_eval.score runs/lar27b                       # main run only\n"
            "  python -m tb21_eval.score runs/lar27b runs/lar27b_r01       # main run + one recovery round\n"
            "  python -m tb21_eval.score runs/lar27b --policy stock --json score-stock.json\n\n"
            "exit status: 0 = complete, 1 = NOT FINAL (PENDING tasks; re-run commands printed), 2 = invalid input"),
    )
    parser.add_argument("dirs", nargs="+", type=Path, metavar="DIR",
                        help="a harbor job dir, or a run dir holding jobs/<job>/; give the main run first, then each "
                             "recovery run, in launch order")
    parser.add_argument("--k", type=int, default=N_ATTEMPTS,
                        help="trials per task (default: %(default)s)")
    parser.add_argument("--policy", choices=POLICIES, default="kit",
                        help="kit = the kit's infrastructure rule (default); stock = harbor semantics "
                             "(recorded reward, missing = 0). The other reading is always printed alongside.")
    parser.add_argument("--expected-tasks", type=Path, metavar="FILE",
                        help=f"task names, whitespace-separated (default: the {len(TASK_NAMES)} TB-2.1 tasks)")
    parser.add_argument("--json", type=Path, metavar="OUT.json",
                        help="also write the full report (every trial with both verdicts) as JSON")
    parser.add_argument("--reference", type=Path, default=None, metavar="FILE",
                        help="optional user-supplied per-task vectors {task: [0/1, ...]} for comparison")
    parser.add_argument("--no-reference", action="store_true", help="skip the paired comparison")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        expected = parse_task_file(args.expected_tasks) if args.expected_tasks else None
        report = score(args.dirs, k=args.k, policy=args.policy, expected=expected,
                       reference=None if args.no_reference else args.reference)
    except (ScoreError, OSError) as exc:  # OSError: e.g. a directory that cannot be listed
        print(f"tb21_eval.score: error: {exc}", file=sys.stderr)
        return 2
    print(render(report))
    if args.json:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"tb21_eval.score: error: cannot write {args.json}: {exc}", file=sys.stderr)
            return 2
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Self-test of tb21_eval.score on a synthetic harbor job tree (no network, no docker, standard library only).

    python tests/test_score.py            # from the kit root (or: python -m unittest discover -s tests)

The tree holds two rounds: a main run dir ``run/jobs/main`` and a recovery run dir ``run_r01/jobs/main_r01``.
Every classification case appears once (k = 5):

    task     round 1 (main)                                         round 2 (main_r01)  kit verdict
    alpha    rewards 1,1,1,0,0 by name; the two 0s started first    -                   selected r1, 3/5
    bravo    4 x 1 + VerifierTimeoutError after pytest started      -                   selected r1, 4/5 (ZERO)
    charlie  4 x 1 + reward 0 with no ctrf.json (INFRA)             5 x 1               selected r2, 5/5
    delta    AgentTimeoutError (0), OutputParserError (1), 3 x 1    -                   selected r1, 4/5
    echo     4 x 1 + one unfinished trial (config.json only)        -                   PENDING
    foxtrot  EnvironmentStartTimeoutError + 4 x 1                   -                   PENDING
    golf     VerifierTimeoutError before pytest started + 4 x 1     -                   PENDING
    hotel    empty CTRF (0) + malformed CTRF (1) + 3 x 1            -                   PENDING
    juliet   only 4 trials                                          -                   PENDING
    kilo     no trials                                              -                   PENDING
    zulu     1 trial of a task outside the expected set             copy of an alpha    ignored, with warnings
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the kit importable without installing it

from tb21_eval import score as S  # noqa: E402

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
PYTEST_STARTED = ("============================= test session starts ==============================\n"
                  "collected 2 items\n\n../tests/test_outputs.py ")
SETUP_ONLY = ("Reading package lists...\nE: Unable to locate package curl\n"
              "/tests/test.sh: line 8: curl: command not found\n")
EXPECTED = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "juliet", "kilo"]
COMPLETE = ["alpha", "bravo", "charlie", "delta"]


# ----------------------------------------------------------------------------------------------- tree builders
def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def stamp(seconds: float, naive: bool = False) -> str:
    moment = T0 + dt.timedelta(seconds=seconds)
    return moment.replace(tzinfo=None).isoformat() if naive else moment.isoformat().replace("+00:00", "Z")


def make_job(job_dir: Path, job_name: str, started_s: float | None = 0.0) -> None:
    """Harbor's job-level files; harbor writes naive local timestamps into the job result.json."""
    write_json(job_dir / "config.json", {"job_name": job_name, "n_attempts": 5})
    if started_s is not None:
        write_json(job_dir / "result.json", {"id": job_name, "started_at": stamp(started_s, naive=True),
                                             "finished_at": stamp(started_s + 7200, naive=True),
                                             "n_total_trials": 0, "stats": {}})
    (job_dir / "job.log").write_text("", encoding="utf-8")


def make_trial(job_dir: Path, name: str, *, reward: float | None = 1.0, exception: str | None = None,
               message: str = "", ctrf: str = "valid", stdout: str = PYTEST_STARTED, start: float = 0.0,
               finished: bool = True) -> Path:
    """One harbor trial dir; ctrf is 'valid', 'missing', 'empty' or 'malformed'."""
    task = name.rsplit("__", 1)[0]
    trial = job_dir / name
    write_json(trial / "config.json", {"trial_name": name, "task": {"path": f"/work/tasks-aa/{task}"}})
    (trial / "verifier").mkdir(parents=True, exist_ok=True)
    (trial / "verifier" / "test-stdout.txt").write_text(stdout, encoding="utf-8")
    if ctrf == "valid":
        write_json(trial / "verifier" / "ctrf.json", {"results": {
            "tool": {"name": "pytest"}, "summary": {"tests": 2, "passed": 2, "failed": 0},
            "tests": [{"name": "test_a", "status": "passed"}, {"name": "test_b", "status": "passed"}]}})
    elif ctrf == "empty":
        write_json(trial / "verifier" / "ctrf.json", {"results": {"summary": {"tests": 0}, "tests": []}})
    elif ctrf == "malformed":
        (trial / "verifier" / "ctrf.json").write_text('{"results": {"summary": ', encoding="utf-8")
    if finished:
        write_json(trial / "result.json", {
            "trial_name": name, "task_name": f"terminal-bench/{task}",
            "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
            "exception_info": None if exception is None else {"exception_type": exception,
                                                               "exception_message": message},
            "started_at": stamp(start), "finished_at": stamp(start + 600),
            "agent_execution": {"started_at": stamp(start + 10), "finished_at": stamp(start + 500)},
        })
    return trial


def build_tree(root: Path) -> tuple[Path, Path]:
    """The two-round tree of the module docstring; returns (main run dir, recovery run dir)."""
    run, recovery = root / "run", root / "run_r01"
    main = run / "jobs" / "main"
    make_job(main, "main", started_s=0)

    def plain(task: str, letter: str, rewards: list[float], start: float = 0.0) -> None:
        for i, reward in enumerate(rewards, start=1):
            make_trial(main, f"{task}__{letter}{i}zzzzz", reward=reward, start=start + 100 * i)

    # alpha: by trial name the rewards read 1,1,1,0,0, but the two failures started first.
    for i, (reward, start) in enumerate([(1, 300), (1, 400), (1, 500), (0, 100), (0, 200)], start=1):
        make_trial(main, f"alpha__A{i}zzzzz", reward=reward, start=start)
    plain("bravo", "B", [1, 1, 1, 1])
    make_trial(main, "bravo__B5zzzzz", reward=None, exception="VerifierTimeoutError", ctrf="missing",
               message="Verifier execution timed out after 900.0 seconds", start=500)
    plain("charlie", "C", [1, 1, 1, 1])
    make_trial(main, "charlie__C5zzzzz", reward=0.0, ctrf="missing", stdout=SETUP_ONLY, start=500)
    make_trial(main, "delta__D1zzzzz", reward=0.0, exception="AgentTimeoutError",
               message="Agent execution timed out after 20000.0 seconds", start=100)
    make_trial(main, "delta__D2zzzzz", reward=1.0, exception="OutputParserError", start=200)
    for i in (3, 4, 5):
        make_trial(main, f"delta__D{i}zzzzz", start=100 * i)
    plain("echo", "E", [1, 1, 1, 1])
    make_trial(main, "echo__E5zzzzz", finished=False)
    make_trial(main, "foxtrot__F1zzzzz", reward=None, exception="EnvironmentStartTimeoutError", ctrf="missing",
               message="Environment start timed out after 600 seconds")
    for i in (2, 3, 4, 5):
        make_trial(main, f"foxtrot__F{i}zzzzz", start=100 * i)
    make_trial(main, "golf__G1zzzzz", reward=None, exception="VerifierTimeoutError", ctrf="missing",
               stdout=SETUP_ONLY, message="Verifier execution timed out after 900.0 seconds")
    for i in (2, 3, 4, 5):
        make_trial(main, f"golf__G{i}zzzzz", start=100 * i)
    make_trial(main, "hotel__H1zzzzz", reward=0.0, ctrf="empty")
    make_trial(main, "hotel__H2zzzzz", reward=1.0, ctrf="malformed")
    for i in (3, 4, 5):
        make_trial(main, f"hotel__H{i}zzzzz", start=100 * i)
    plain("juliet", "J", [1, 1, 1, 1])
    make_trial(main, "zulu__Z1zzzzz")
    (main / "stray__Q1w2e3r").mkdir()          # matches the trial pattern but holds nothing: warned, ignored
    (main / "notes").mkdir()                   # not a trial dir name: silently ignored

    r01 = recovery / "jobs" / "main_r01"
    make_job(r01, "main_r01", started_s=10_000)
    for i in range(1, 6):
        make_trial(r01, f"charlie__R{i}zzzzz", start=10_000 + 100 * i)
    shutil.copytree(main / "alpha__A1zzzzz", r01 / "alpha__A1zzzzz")  # same trial twice: second copy ignored
    return run, recovery


def quiet_main(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = S.main(argv)
    return code, buffer.getvalue()


# ------------------------------------------------------------------------------------------------------- tests
class ScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="tb21_score_test_"))
        cls.run_dir, cls.recovery_dir = build_tree(cls.tmp)
        cls.report = S.score([cls.run_dir, cls.recovery_dir], expected=EXPECTED, reference=None)
        cls.trials = {row["trial"]: row for row in cls.report["trials"]}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # --- classification -----------------------------------------------------------------------------------
    def verdict(self, trial: str, policy: str = "kit") -> dict:
        return self.trials[trial][policy]

    def test_kit_classification(self) -> None:
        cases = {
            "alpha__A1zzzzz": ("SCORED", 1.0, "pass"),
            "alpha__A4zzzzz": ("SCORED", 0.0, "fail"),
            "bravo__B5zzzzz": ("ZERO", 0.0, "verifier_timeout_zero"),
            "charlie__C5zzzzz": ("INFRA", None, "infra"),
            "delta__D1zzzzz": ("SCORED", 0.0, "agent_timeout"),
            "delta__D2zzzzz": ("SCORED", 1.0, "pass"),
            "echo__E5zzzzz": ("INFRA", None, "infra"),
            "foxtrot__F1zzzzz": ("INFRA", None, "infra"),
            "golf__G1zzzzz": ("INFRA", None, "infra"),
            "hotel__H1zzzzz": ("INFRA", None, "infra"),
            "hotel__H2zzzzz": ("INFRA", None, "infra"),
        }
        for trial, (status, score, outcome) in cases.items():
            with self.subTest(trial=trial):
                got = self.verdict(trial)
                self.assertEqual((got["status"], got["score"], got["outcome"]), (status, score, outcome))
        self.assertIn("ctrf.json missing", self.verdict("charlie__C5zzzzz")["reason"])
        self.assertIn("unfinished", self.verdict("echo__E5zzzzz")["reason"])
        self.assertIn("EnvironmentStartTimeoutError", self.verdict("foxtrot__F1zzzzz")["reason"])
        self.assertIn("before pytest started", self.verdict("golf__G1zzzzz")["reason"])
        self.assertIn("reports no tests", self.verdict("hotel__H1zzzzz")["reason"])
        self.assertIn("unreadable", self.verdict("hotel__H2zzzzz")["reason"])

    def test_stock_classification(self) -> None:
        # Harbor semantics: recorded reward, missing reward = 0; only the unfinished trial is INFRA.
        for trial, score in {"bravo__B5zzzzz": 0.0, "charlie__C5zzzzz": 0.0, "foxtrot__F1zzzzz": 0.0,
                             "golf__G1zzzzz": 0.0, "hotel__H1zzzzz": 0.0, "hotel__H2zzzzz": 1.0}.items():
            with self.subTest(trial=trial):
                self.assertEqual((self.verdict(trial, "stock")["status"], self.verdict(trial, "stock")["score"]),
                                 ("SCORED", score))
        self.assertEqual(self.verdict("echo__E5zzzzz", "stock")["status"], "INFRA")

    def test_discovery(self) -> None:
        rounds = self.report["rounds"]
        self.assertEqual([r["job_name"] for r in rounds], ["main", "main_r01"])
        self.assertEqual([r["trials"] for r in rounds], [45, 5])  # 44 + zulu; duplicate and stray dirs ignored
        self.assertEqual(self.trials["alpha__A1zzzzz"]["round"], 1)
        self.assertEqual(self.trials["alpha__A1zzzzz"]["dir"], str(self.run_dir / "jobs" / "main" / "alpha__A1zzzzz"))
        warnings = "\n".join(self.report["warnings"])
        self.assertIn("duplicate of trial alpha__A1zzzzz", warnings)
        self.assertIn("stray__Q1w2e3r", warnings)
        self.assertIn("zulu (1)", warnings)
        self.assertNotIn(str(self.run_dir / "jobs" / "main" / "notes"), warnings)

    # --- selection ----------------------------------------------------------------------------------------
    def test_kit_selection(self) -> None:
        per_task = self.report["per_task"]
        self.assertEqual({t: per_task[t]["round"] for t in COMPLETE}, {"alpha": 1, "bravo": 1, "charlie": 2,
                                                                        "delta": 1})
        self.assertEqual(per_task["alpha"]["vector"], [1, 1, 1, 0, 0])   # trial-name order
        self.assertEqual(per_task["bravo"]["vector"], [1, 1, 1, 1, 0])
        self.assertEqual(per_task["charlie"]["trials"], [f"charlie__R{i}zzzzz" for i in range(1, 6)])
        self.assertIn("round 1: 1 of 5 trials INFRA", per_task["charlie"]["notes"][0])
        self.assertEqual(per_task["delta"]["vector"], [0, 1, 1, 1, 1])
        self.assertEqual(self.report["pending"], ["echo", "foxtrot", "golf", "hotel", "juliet", "kilo"])
        self.assertIn("only 4 of 5 trials", per_task["juliet"]["notes"][0])
        self.assertEqual(per_task["kilo"]["notes"], [])
        self.assertFalse(self.report["complete"])

    def test_kit_metrics(self) -> None:
        head = self.report["headline"]
        self.assertEqual((head["passes"], head["trials"]), (16, 50))     # 3 + 4 + 5 + 4, pending zero-filled
        self.assertAlmostEqual(head["mean_at_k"], 16 / 50)
        self.assertEqual((head["solved_tasks"], head["tasks"]), (4, 10))
        # Reward-blind first 3 by started_at: alpha 1 (its failures started first), bravo 3, charlie 3, delta 2.
        self.assertEqual((head["mean_at_3_passes"], head["mean_at_3_trials"]), (9, 30))
        self.assertEqual(self.report["solve_histogram"], {"0": 0, "1": 0, "2": 0, "3": 1, "4": 2, "5": 1})
        self.assertEqual(self.report["outcomes"]["selected"], {"pass": 16, "fail": 2, "agent_timeout": 1,
                                                                "verifier_timeout_zero": 1,
                                                                "other_exception_zero": 0, "infra": 0})
        self.assertEqual(self.report["outcomes"]["all"]["infra"], 6)
        self.assertEqual(sum(self.report["outcomes"]["all"].values()), 49)
        durations = self.report["durations_s"]["selected"]
        self.assertEqual((durations["trial"]["n"], durations["trial"]["p50"]), (20, 600.0))
        self.assertEqual(durations["agent_execution"]["max"], 490.0)

    def test_stock_reading(self) -> None:
        stock = self.report["readings"]["stock"]
        # Round 1 for every task: alpha 3, bravo 4, charlie 4, delta 4, foxtrot 4, golf 4, hotel 4.
        self.assertEqual((stock["passes"], stock["trials"]), (27, 50))
        self.assertEqual(stock["pending"], ["echo", "juliet", "kilo"])
        self.assertEqual(stock["selected_from_round"], {1: 7})

    def test_complete_subset(self) -> None:
        report = S.score([self.run_dir, self.recovery_dir], expected=COMPLETE, reference=None)
        self.assertTrue(report["complete"])
        self.assertIsNone(report["rerun"])
        head = report["headline"]
        self.assertEqual((head["passes"], head["trials"], head["solved_tasks"]), (16, 20, 4))
        self.assertEqual((head["mean_at_3_passes"], head["mean_at_3_trials"]), (9, 12))
        self.assertEqual(report["readings"]["stock"]["passes"], 15)   # charlie from round 1: 4 instead of 5

    # --- rounds from run dirs -------------------------------------------------------------------------------
    def test_run_dir_job_order(self) -> None:
        run = self.tmp / "ordered"
        early, late, nostart = run / "jobs" / "zz_early", run / "jobs" / "aa_late", run / "jobs" / "mm_nostart"
        make_job(late, "aa_late", started_s=3600)
        make_job(early, "zz_early", started_s=0)
        make_job(nostart, "mm_nostart", started_s=None)
        (run / "jobs" / "empty").mkdir()
        for i in range(1, 6):
            make_trial(early, f"alpha__E{i}zzzzz", reward=1.0, ctrf="missing" if i == 1 else "valid")
            make_trial(late, f"alpha__L{i}zzzzz", reward=0.0)
            make_trial(nostart, f"alpha__N{i}zzzzz", reward=1.0)
        report = S.score([run], expected=["alpha"], reference=None)
        self.assertEqual([r["job_name"] for r in report["rounds"]], ["zz_early", "aa_late", "mm_nostart"])
        self.assertEqual(report["per_task"]["alpha"]["round"], 2)     # the first infra-clean round in that order
        self.assertEqual(report["headline"]["passes"], 0)
        self.assertTrue(any("empty directory" in w for w in report["warnings"]))

    # --- comparison ---------------------------------------------------------------------------------------
    def test_comparison(self) -> None:
        reference = self.tmp / "reference.json"
        vectors = {"alpha": [1, 1, 1, 0, 0], "bravo": [1, 1, 1, 1, 0], "charlie": [1, 1, 1, 1, 1],
                   "delta": [0, 1, 1, 1, 1]}
        write_json(reference, vectors)
        same = S.score([self.run_dir, self.recovery_dir], expected=COMPLETE, reference=reference)["comparison"]
        self.assertEqual((same["delta_passes"], same["bootstrap95_pp"], same["tasks_equal"]), (0, [0.0, 0.0], 4))
        self.assertTrue(same["identical_vectors"])
        write_json(reference, {**vectors, "charlie": [0, 0, 0, 0, 0]})
        better = S.score([self.run_dir, self.recovery_dir], expected=COMPLETE, reference=reference)["comparison"]
        self.assertEqual((better["delta_passes"], better["tasks_up"]), (5, 1))
        self.assertAlmostEqual(better["delta_pp"], 25.0)
        lo, hi = better["bootstrap95_pp"]
        self.assertTrue(0.0 <= lo <= 25.0 <= hi <= 100.0)
        self.assertFalse(better["identical_vectors"])

    def test_external_reference_validation(self) -> None:
        reference = self.tmp / "synthetic_reference.json"
        vectors = {"alpha": [1, 0], "bravo": [0, 1]}
        write_json(reference, vectors)
        self.assertEqual(S.load_reference(reference), vectors)
        for invalid in ({}, [], {"alpha": []}, {"alpha": [True]}, {"alpha": [2]}):
            with self.subTest(invalid=invalid):
                write_json(reference, invalid)
                with self.assertRaises(S.ScoreError):
                    S.load_reference(reference)

    def test_reference_is_opt_in(self) -> None:
        report = S.score([self.run_dir, self.recovery_dir], expected=COMPLETE)
        self.assertIsNone(report["comparison"])
        self.assertIsNone(S.build_parser().parse_args([str(self.run_dir)]).reference)
        tasks = self.tmp / "default_reference_tasks.txt"
        tasks.write_text(" ".join(COMPLETE), encoding="utf-8")
        code, text = quiet_main([str(self.run_dir), str(self.recovery_dir), "--expected-tasks", str(tasks)])
        self.assertEqual(code, 0)
        self.assertNotIn("Paired comparison", text)

    # --- statistics ---------------------------------------------------------------------------------------
    def test_wilson_and_quantiles(self) -> None:
        # Synthetic boundary cases: no successes and all successes have mirrored intervals.
        for total in (1, 4, 10):
            lower, upper = S.wilson(0, total)
            mirrored_lower, mirrored_upper = S.wilson(total, total)
            self.assertAlmostEqual(lower, 0)
            self.assertAlmostEqual(mirrored_upper, 1)
            self.assertAlmostEqual(upper, 1 - mirrored_lower)
            self.assertAlmostEqual(upper, S.WILSON_Z ** 2 / (total + S.WILSON_Z ** 2))
        # Ordered synthetic observations make the nearest-rank answers explicit.
        data = [float(x) for x in range(1, 101)]
        self.assertEqual([S.nearest_rank(data, p) for p in (50, 90, 95, 100)], [50, 90, 95, 100])
        data = [float(x) for x in range(1, 18)]
        self.assertEqual((S.nearest_rank(data, 50), S.nearest_rank(data, 95)), (9, 17))

    # --- CLI ----------------------------------------------------------------------------------------------
    def test_cli(self) -> None:
        tasks = self.tmp / "tasks.txt"
        tasks.write_text("# synthetic\n" + "\n".join(EXPECTED) + "\n", encoding="utf-8")
        out_json = self.tmp / "out" / "score.json"
        code, text = quiet_main([str(self.run_dir), str(self.recovery_dir), "--expected-tasks", str(tasks),
                                 "--no-reference", "--json", str(out_json)])
        self.assertEqual(code, 1)
        self.assertIn("NOT FINAL (kit): 6 of 10 tasks PENDING", text)
        # make_job's layout: R/job.<name>.json -> R/jobs/<name>/, with R the run dir of the last round, so that
        # re-scoring the same inputs (R among them) picks the new job up as the last round.
        last = self.recovery_dir
        # Commands use the running interpreter's venv (the Quickstart never activates it) and fill in the main
        # job's --model/--work-dir from R/harbor.env and R/job.json (placeholders when those files are absent).
        py, harbor = sys.executable, str(Path(sys.executable).parent / "harbor")
        self.assertIn(f"{py} -m tb21_eval.make_job --model <model-dir> --work-dir <work-dir> --run-dir {last} "
                      "--tasks echo,foxtrot,golf,hotel,juliet,kilo --job-name main_r02", text)
        self.assertIn(f"{harbor} run -c {last}/job.main_r02.json --env-file {last}/harbor.env", text)
        self.assertIn(f"{py} -m tb21_eval.score {self.run_dir} {self.recovery_dir}\n", text)
        self.assertIn(f"harbor job resume --job-path {self.run_dir / 'jobs' / 'main'})", text)  # echo is unfinished
        saved = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(saved["rerun"]["unfinished_trials"], ["echo__E5zzzzz"])
        self.assertEqual(saved["per_task_vector"]["charlie"], [1, 1, 1, 1, 1])

        tasks.write_text(" ".join(COMPLETE), encoding="utf-8")
        code, text = quiet_main([str(self.run_dir), str(self.recovery_dir), "--expected-tasks", str(tasks),
                                 "--no-reference"])
        self.assertEqual(code, 0)
        self.assertIn("FINAL (kit): Mean@5 = 80.00% (16/20)", text)
        code, text = quiet_main([str(self.run_dir), "--expected-tasks", str(tasks), "--no-reference",
                                 "--policy", "stock"])
        self.assertEqual(code, 0)
        self.assertIn("FINAL (stock): Mean@5 = 75.00% (15/20)", text)

    def test_safety_stop_is_infra(self) -> None:
        # A trial the PSWF agent ended on a model-guard refusal is re-run (kit) but scored as recorded (stock).
        run = self.tmp / "stop_run"
        job = run / "jobs" / "stop"
        make_job(job, "stop", started_s=0)
        for i in range(1, 6):
            trial = make_trial(job, f"alpha__S{i}zzzzz", reward=0.0 if i == 1 else 1.0, start=100 * i)
            if i == 1:
                write_json(trial / "agent" / "dynwf_meta.json",
                           {"tb_mode": "dynamic", "safety_stop": {"kind": "tor", "immediate": True}})
        rep = S.score([run], expected=["alpha"], reference=None)
        self.assertEqual(rep["pending"], ["alpha"])
        self.assertEqual(rep["rerun"]["tasks"], ["alpha"])
        why = rep["per_task"]["alpha"]["notes"][0]
        self.assertIn("episode ended by a model-guard refusal (tor)", why)
        stock = S.score([run], expected=["alpha"], reference=None, policy="stock")
        self.assertTrue(stock["complete"])
        self.assertEqual(stock["headline"]["passes"], 4)

    def test_running_job_is_in_progress_not_infra(self) -> None:
        # While harbor runs, unfinished trials and not-yet-started attempts are "in progress": no recovery job.
        run = self.tmp / "live_run"
        job = run / "jobs" / "live"
        write_json(job / "config.json", {"job_name": "live", "n_attempts": 5})
        write_json(job / "result.json", {"id": "live", "started_at": stamp(0, naive=True), "finished_at": None})
        for i in (1, 2):
            make_trial(job, f"alpha__L{i}zzzzz", start=100 * i)
        make_trial(job, "alpha__L3zzzzz", finished=False)
        make_trial(job, "bravo__L1zzzzz", reward=0.0, ctrf="missing", stdout=SETUP_ONLY)  # a real infra trial
        rep = S.score([run], expected=["alpha", "bravo", "charlie"], reference=None)
        self.assertFalse(rep["complete"])
        self.assertEqual(rep["running_jobs"], ["live"])
        self.assertEqual(sorted(rep["in_progress"]), ["alpha", "charlie"])
        self.assertEqual(rep["rerun"]["tasks"], ["bravo"])
        self.assertEqual(rep["rerun"]["resume_commands"], [])          # never resume a job that is still running
        self.assertIn("IN PROGRESS (2 tasks)", S.render(rep))

    def test_rerun_plan_for_job_dirs(self) -> None:
        # Scoring an explicit job dir: the re-score command must name the new job dir as an extra round.
        main = self.run_dir / "jobs" / "main"
        plan = S.score([main], expected=["charlie"], reference=None)["rerun"]
        self.assertEqual(plan["job_name"], "main_r01")
        self.assertEqual(plan["commands"][2], f"{sys.executable} -m tb21_eval.score {main} {self.run_dir}/jobs/main_r01")
        self.assertEqual(plan["resume_commands"], [])

    def test_invalid_input(self) -> None:
        main = self.run_dir / "jobs" / "main"
        plain = self.tmp / "plain"
        plain.mkdir(exist_ok=True)
        (plain / "readme.txt").write_text("not a harbor dir\n", encoding="utf-8")
        for dirs, message in (([self.tmp / "missing"], "no such directory"),
                              ([main, self.run_dir], "given twice"),
                              ([main / "alpha__A1zzzzz"], "single trial dir"),
                              ([self.run_dir / "jobs"], "jobs_dir"),
                              ([plain], "neither")):
            with self.subTest(dirs=dirs), self.assertRaisesRegex(S.ScoreError, message):
                S.score(dirs, reference=None)
        with self.assertRaisesRegex(S.ScoreError, "listed twice"):
            S.score([self.run_dir], expected=["alpha", "alpha"], reference=None)
        # A just-started job whose name looks like a trial name is still a job dir (its result.json is job-level).
        fresh = self.tmp / "fresh" / "lar27b__2026a"
        make_job(fresh, "lar27b__2026a")
        report = S.score([fresh], expected=["alpha"], reference=None)
        self.assertEqual((report["rounds"][0]["trials"], report["pending"]), (0, ["alpha"]))
        self.assertEqual(quiet_main([str(self.run_dir), "--k", "0"])[0], 2)
        self.assertEqual(quiet_main([str(self.tmp / "missing")])[0], 2)

    @unittest.skipIf(os.geteuid() == 0, "root can read any file")
    def test_unreadable_evidence_aborts(self) -> None:
        # Unreadable files must stop the scorer, not turn every trial into INFRA (and a re-run of everything).
        job = self.tmp / "perm" / "jobs" / "main"
        make_job(job, "main")
        ctrf = make_trial(job, "alpha__P1zzzzz") / "verifier" / "ctrf.json"
        ctrf.chmod(0)
        try:
            with self.assertRaisesRegex(S.ScoreError, "permission denied"):
                S.score([job.parent.parent], expected=["alpha"], reference=None)
        finally:
            ctrf.chmod(0o644)


if __name__ == "__main__":
    unittest.main(verbosity=2)

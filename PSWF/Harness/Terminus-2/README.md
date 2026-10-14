# PSWF harness for Terminus-2

This package runs problem-specific workflows with Harbor's Terminus-2 agent.
Each task uses two phases in the same container:

1. **Explore and specify.** The agent inspects the task and writes a workflow with
   acceptance criteria, environment findings, a plan, execution checkpoints, and
   verification steps. Configurable time, turn, and context limits bound this phase.
2. **Solve.** The agent receives the workflow and completes the task. Optional controls
   refresh the plan before context compaction and remind the agent to finish its
   deliverables near the end of the budget.

`terminus2_dynwf/` contains the agent and prompt templates. `pswf_harbor_patches/`
contains the required Harbor patches and their release verification command.
`tests/` contains offline tests with handwritten synthetic fixtures.

## Requirements and installation

Use Python 3.12 and Harbor from commit
`f03db62fd2ed2ed1f79aefe024cfcbc68a0d759e`. The package version alone does not identify
that source revision. Install the pinned environment from the evaluation kit first:

```bash
pip install -r ../../EvaluationKit/TB-2.1/requirements/eval.txt
pip install .
python -m pswf_harbor_patches \
  --tokenizer-dir /path/to/model \
  --manifest-dir /path/outside/release-tree/harbor-patches
```

The tokenizer patch counts context with the served model's tokenizer. The summarizer
patch fits constructed requests within the configured input budget. The remaining
patches handle container cleanup, network configuration, pre-container hooks, and
bounded retries for transient verifier dependency errors.

The patch command accepts the pinned pristine tree or this release's patched tree.
It verifies the installed Harbor and harness against the release digests in
`pswf_harbor_patches/__init__.py`. These digests identify the cleaned release source,
including its comments and prompt files. `python -m pswf_harbor_patches --check`
verifies an existing installation without applying patches.

When upgrading from an earlier patched installation, recreate the environment or
reinstall the pinned pristine Harbor revision before applying this release. The
verifier rejects an older patched tree whose source digest differs from this release.

## Use

Configure Harbor with the import path
`terminus2_dynwf.dynamic_agent:Terminus2Dynamic`. Set `tb_mode="dynamic"` for the
two-phase workflow or `tb_mode="vanilla"` for stock Terminus-2 orchestration.
`../../EvaluationKit/TB-2.1` provides serving, task setup, agent kwargs, and scoring.

The agent expects an OpenAI-compatible endpoint with separate `reasoning_content`.
Set `Q38_TOKENIZER_JSON` to the served model's local `tokenizer.json`.
`interleaved_thinking` controls reasoning replay, and `stop_on_policy_violation`
ends a trial when the endpoint returns the configured policy refusal.

Logs, workflow dumps, telemetry, and test output are runtime artifacts. Keep their
directories outside the release tree. The class defaults disable optional profiling
and refreshed-plan dumps; the evaluation kit's agent kwargs enable those features.
The trusted output directory separates artifacts from the task container and does
not redact their contents.

## Tests

Run the standalone tests in the Harbor environment after applying the release
patches. The HTTP tests use a loopback server and a local tokenizer, with no model
inference or benchmark tasks. Store test output and temporary files outside the
release tree:

```bash
export PYTHONDONTWRITEBYTECODE=1
export Q38_TOKENIZER_JSON=/path/to/model/tokenizer.json
export TMPDIR=/tmp
for test_file in tests/test_*.py; do
  python -B "$test_file" || exit 1
done
```

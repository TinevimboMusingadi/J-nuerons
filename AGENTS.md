# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **CPU-friendly Python research codebase** (no web app / no GUI). It
holds two related projects that share one Python environment:

- **Root `J-Neurons`** — jailbreak-detection stack (`src/`, `tests/sanity_check.py`).
- **`secret-loyalties/` (L-Neurons)** — the primary project for this branch;
  sparse activation probes for detecting secret loyalties (`secret-loyalties/src/`,
  `secret-loyalties/tests/`).

### Environment notes
- Dependencies are installed into the system Python (`python3`, 3.12) via the
  startup update script; there is **no virtualenv** — just run `python3` directly.
- `torch` is installed as the **CPU-only** wheel (no GPU in this VM). The live
  pipeline auto-selects CPU and runs fine on the small Qwen models.
- The GPU helper `secret-loyalties/scripts/setup_gpu.sh` is for provisioning a
  fresh GCP GPU VM only — do not run it here.

### Lint / test / run
- Tests are plain `unittest`, run offline with no model weights:
  - `python3 secret-loyalties/tests/test_lneurons.py` (24 tests)
  - `python3 secret-loyalties/tests/test_rigor.py` (15 tests)
  - `python3 tests/sanity_check.py` (root J-Neurons mock end-to-end)
- Offline analysis pipeline (deterministic, no weights):
  - `python3 secret-loyalties/src/run_pipeline.py --mock`
  - `python3 secret-loyalties/src/analyze_results.py`
- Live pipeline (downloads an open-weight model from HF, CPU is fine for small
  ones). Keep smoke runs cheap with `--max-rows` / `--skip-jlens`:
  - `python3 secret-loyalties/src/run_pipeline.py --model Qwen/Qwen2.5-0.5B-Instruct --max-rows 6 --skip-jlens`
- No configured linter (no flake8/ruff/black config in the repo); the standard
  quality gate here is the `unittest` suites above.

### Gotchas
- `run_pipeline.py` / tests inject `secret-loyalties/src` and the repo root onto
  `sys.path` themselves, so run the scripts **directly by path** (as above)
  rather than importing modules from an arbitrary cwd.
- The live run prints many sklearn `ConvergenceWarning`s from the L1 probe on
  small `--max-rows` data — these are benign, not failures.
- Running the pipeline overwrites tracked JSON/PNG artifacts under `results/` and
  `secret-loyalties/results/`. These are committed sample outputs; revert them
  (`git checkout -- results secret-loyalties/results`) unless you intend to
  update them.
- The black-box audit uses a local HF model by default. The optional
  `AnthropicProvider` needs `ANTHROPIC_API_KEY`; Gemini dataset regeneration
  (`src/data/generate_data.py`) needs `GEMINI_API_KEY`. Neither is required for
  tests or the default pipeline (there are offline fallbacks).

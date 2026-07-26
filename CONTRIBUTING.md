# Contributing

Thank you for helping improve the project. Contributions should make experiments
clearer, safer, or easier to reproduce.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Before submitting a change, run:

```bash
python -m compileall model scm scripts benchmarking
python -m build
ruff check model scm scripts benchmarking
```

GPU experiments are not required for documentation-only contributions. For model
changes, describe the dataset split, configuration, hardware, seed, checkpoint,
and metrics used for validation.

## Pull requests

- Keep each pull request focused.
- Add or update documentation when behavior changes.
- Preserve copyright, license, citation, and adapted-code notices.
- Never commit credentials, protected datasets, subject data, checkpoints, or
  large generated artifacts.
- Explain whether results are reproduced, newly measured, or expected but not
  yet verified.

By contributing, you agree that your contribution is distributed under the
repository's MIT License.

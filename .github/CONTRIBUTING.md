# Contributing to TPS

First of all, **thank you for considering contributing!**
All contributions are welcome: bug reports, documentation improvements, examples, feature requests, or code changes.

This document provides a few guidelines to help keep the project consistent and easy to maintain.

---

## Development setup

This project uses **uv** to manage the Python environment and dependencies.

To install uv, please follow the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).


### 1. Clone the repository

```bash
git clone git@github.com:raphaelreme/tps.git  # OR https://github.com/raphaelreme/tps.git
cd tps
```

### 2. Create the development environment

```bash
uv sync
```

This will create a local virtual environment and install all required dependencies (including dev ones).

You can then activate the python environment with:
```bash
source .venv/bin/activate
```

or run commands through the `uv run` interface as showed below.


---

## Code quality & tooling

We rely on modern Python tooling to ensure code quality and consistency.

### Linting and formatting

- **ruff** is used for both linting and formatting
- **mypy** is used for static type checking

Run locally:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
```

Please make sure your changes pass these checks before submitting a PR.

---

## Tests

Tests are written using **pytest**.

Run the test suite with:

```bash
uv run pytest
# OR
uv run pytest --cov  # To see test coverage
```

We use **codecov** in CI to monitor test coverage.
New features should include appropriate tests, and bug fixes should ideally include a regression test.

---

## Commit hygiene

This repository uses **prek** to sanitize commits.

Before committing, make sure hooks are installed:

```bash
uv run prek install
```

Commits should:
- Be small and focused
- Have clear, descriptive commit messages
- Avoid mixing unrelated changes

---

## Pull requests

When opening a pull request:

1. Clearly describe **what** the change does and **why**
2. Reference relevant issues if applicable
3. Ensure:
   - Tests pass
   - Linting and formatting are clean
   - Type checking passes
4. Keep PRs reasonably scoped (large changes are easier to review when split)

Draft PRs are welcome if you want early feedback.

---

## Style & design guidelines

- Prefer clarity over cleverness
- Keep mathematical notation consistent with existing code and documentation
- Avoid unnecessary dependencies
- When adding numerical methods, consider numerical stability and document assumptions

---

## Reporting issues

If you find a bug or have a feature request:
- Open an issue with a minimal reproducible example when possible
- Clearly state expected vs actual behavior

---

## Questions & discussions

If you are unsure about an implementation or design choice, feel free to:
- Open an issue for discussion
- Ask questions in a draft PR

---

Thanks again for contributing — your help is greatly appreciated! 🚀

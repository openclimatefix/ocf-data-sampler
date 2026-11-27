.PHONY: lint format typecheck test check

# Use the latest tool releases, as CI does
RUFF := uvx ruff@latest
TY := uvx ty@latest

lint:
	$(RUFF) check --fix .

format:
	$(RUFF) format .

typecheck:
	$(TY) check .

test:
	uv run python -m pytest

# Run the CI checks without modifying files
check:
	$(RUFF) check .
	$(RUFF) format --check .
	$(TY) check .

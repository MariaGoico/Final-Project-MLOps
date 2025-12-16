# Install dependencies
install:
	pip install uv
	uv sync

# Run tests with coverage
test:
	uv run python -m pytest tests/ -vv --cov=logic --cov=api --cov=cli

# Format code
format:	
	uv run -- black logic/*.py cli/*.py api/*.py

# Lint code (ignore non-critical warnings for make)
lint:
	uv run -- pylint --disable=R,C --ignore-patterns=test_.*\.py logic/*.py api/*.py --exit-zero

# Run format + lint
refactor: format lint

# Run all tasks
all: install lint test format

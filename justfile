setup:
    uv sync
    uv run pre-commit install || true

lint:
    uv run ruff check --fix main.py
    uv run ruff format main.py
    uv run ty check main.py

run *args:
    uv run python main.py {{args}}

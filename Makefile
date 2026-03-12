.PHONY: install install-dev test test-cov lint format clean run

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=rag_studio --cov-report=html --cov-report=term-missing

lint:
	ruff check rag_studio/ tests/
	mypy rag_studio/

format:
	ruff format rag_studio/ tests/
	ruff check --fix rag_studio/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

run:
	uvicorn rag_studio.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t rag-studio:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

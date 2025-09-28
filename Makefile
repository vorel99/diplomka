VENV = venv
PYTHON = $(VENV)/bin/python
ACTIVATE = $(VENV)/bin/activate


install_dev:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	source $(ACTIVATE) && pre-commit install

compile:
	$(PYTHON) -m piptools compile --resolver=backtracking --extra dev --no-emit-index-url -o requirements-dev.txt pyproject.toml
	$(PYTHON) -m piptools compile --resolver=backtracking --no-emit-index-url -o requirements.txt pyproject.toml

sync:
	$(PYTHON) -m piptools sync requirements-dev.txt
	$(PYTHON) -m pip install -e ".[dev]"

up: compile sync

clean:
	find . -type d -name .pytest_cache | xargs --no-run-if-empty -t rm -r
	find . -type d -name __pycache__ | xargs --no-run-if-empty -t rm -r
	find . -type d -name dist | xargs --no-run-if-empty -t rm -r
	find . -type d -name build | xargs --no-run-if-empty -t rm -r
	find . -type d -regex ".*\.egg.*"  | xargs --no-run-if-empty -t rm -r
	rm -rf $(VENV)

.PHONY: install_dev compile sync up clean

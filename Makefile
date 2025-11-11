VENV = .venv
PYTHON = $(VENV)/bin/python

install_dev:
	# uv installed in devcontainer
	# setup venv and install package in editable mode with dev dependencies
	uv venv $(VENV) --clear --python 3.10

	# Install monorepo workspace with dev dependencies
	uv sync --all-packages

	# install pre-commit hooks
	$(PYTHON) -m pre_commit install

clean:
	find . -type d -name .pytest_cache | xargs --no-run-if-empty -t rm -r
	find . -type d -name __pycache__ | xargs --no-run-if-empty -t rm -r
	find . -type d -name dist | xargs --no-run-if-empty -t rm -r
	find . -type d -name build | xargs --no-run-if-empty -t rm -r
	find . -type d -regex ".*\.egg.*"  | xargs --no-run-if-empty -t rm -r
	rm -rf $(VENV)

.PHONY: install_dev compile sync up clean

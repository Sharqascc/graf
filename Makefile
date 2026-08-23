.RECIPEPREFIX = >

install:
> pip install -r requirements/base.txt -r requirements/dev.txt

test:
> pytest -q

coverage:
> pytest --cov=src --cov-report=term-missing

lint:
> ruff check .

format:
> ruff format .
> ruff check --fix .

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

clean-interim:
> rm -rf data/interim data/processed
> @echo "removed data/interim and data/processed (raw preserved)"

clean-all: clean-interim
> rm -rf data/external
> @echo "removed data/external"

freeze:
> python scripts/freeze_lock.py

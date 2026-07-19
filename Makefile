.PHONY: install train test app check

install:
	python -m pip install -e ".[dev]"

train:
	python -m hospital_pricing.train

test:
	pytest -q

app:
	streamlit run app/app.py

check: test train

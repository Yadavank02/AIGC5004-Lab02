# Making Training Work

Diagnosing and fixing a network that would not train, one change at a time.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run
python diagnose.py # reproduces the failure and reports the gradient check
python Lab02_changes.py # runs each fix stage and saves a comparison plot

## Diagnosis
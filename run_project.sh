#!/bin/sh
set -eu
python -m src.data.validation
python -m src.models.train
pytest -q

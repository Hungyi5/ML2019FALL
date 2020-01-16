#!/bin/bash
python3.7 src/make_test.py "${1}" --data_path data/
python3.7 src/predict.py models/best/ --out "${2}"
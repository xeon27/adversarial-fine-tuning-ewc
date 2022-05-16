#!/bin/bash
now=$(date +"%T")
python main.py --lr 0.0001 --optim "sgd" --alpha 0 --consolidate --lamda 40 > output_$now.txt

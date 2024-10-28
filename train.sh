#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=10:00:00
#$ -N train_molclip
#$ -o log/log.out
#$ -e log/log.err
#$ -m abe
#$ -M sakano@li.c.titech.ac.jp

. .venv/bin/activate
python src/scripts/train.py

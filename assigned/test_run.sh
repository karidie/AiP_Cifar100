#!/bin/bash

models=("resnet" "resnet18")
batch_sizes=(64 32)
learning_rates=(0.001 0.01 0.1)
epochs=(150 100 50)

for model in "${models[@]}"
do
  for batch_size in "${batch_sizes[@]}"
  do
    for lr in "${learning_rates[@]}"
    do
      for epoch in "${epochs[@]}"
      do
        python3 cifar_training.py --model $model --batch_size $batch_size --learning_rate $lr --num_epoch $epoch
      done
    done
  done
done
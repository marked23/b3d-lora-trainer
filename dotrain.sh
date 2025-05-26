#!/usr/bin/env bash

torchrun --standalone --nproc_per_node 2 train.py

# set --nproc_per_node to how many GPUs you have.
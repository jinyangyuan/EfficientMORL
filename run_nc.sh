#!/bin/bash

function run_model {
    data_path='../compositional-scene-representation-datasets/'$name'.h5'
    out_dir='experiments_nc/'$name
    if [[ ! -d "$out_dir/weights" ]]; then
        mkdir -p "$out_dir/weights"
    fi
    if [[ ! -d "$out_dir/runs" ]]; then
        mkdir -p "$out_dir/runs"
    fi
    if [[ ! -d "$out_dir/tb" ]]; then
        mkdir -p "$out_dir/tb"
    fi
    python main.py with 'config_'$name'_nc.json' data_path=$data_path training.out_dir=$out_dir
}

for name in 'mnist' 'dsprites' 'abstract' 'clevr' 'shop' 'gso'; do
    run_model
done

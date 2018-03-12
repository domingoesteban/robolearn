#!/usr/bin/env bash

# #### #
# VARS #
# #### #
scenario='dmdgps'

#seeds=(0 50 100)
#init_index=0
#end_index=3
#seeds=("${seeds[@]:${init_index}:${end_index}}")

default_seeds=(0 50 100)
seeds=("${@:-${default_seeds[@]}}")
total_seeds=${#seeds[@]}

echo "Reacher GPS"
echo "Total seeds: ${#seeds[@]}"
echo "Experiment seeds: ${seeds[@]}"
echo ""

for index in ${!seeds[@]}; do
    seed=${seeds[index]}
#    script_index=$((index+init_index))
    script_index=$((index))
    echo "************************************"
    echo "Running '${scenario}' $((script_index+1))/${total_seeds}: Seed: ${seed}"

    python main.py --scenario=${scenario} --seed=${seed} \
    --run_num=${script_index}

done

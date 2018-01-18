#!/usr/bin/env bash

# #### #
# VARS #
# #### #
scenario='test'
log_dir='normal_gps'

seeds=(0 50 100)
init_index=0
end_index=1

# #### #
# RUNS #
# #### #
total_seeds=${#seeds[@]}
seeds=("${seeds[@]:${init_index}:${end_index}}")

echo "Dualist GPS"
echo "Total seeds: ${#seeds[@]}"
echo "Experiment seeds: ${seeds[@]}"
echo ""

for index in ${!seeds[@]}; do
    seed=${seeds[index]}
    script_index=$((index+init_index))
    echo "************************************"
    echo "Running '${scenario}' $((script_index+1))/${total_seeds}: Seed: ${seed}"

    python main.py --scenario=${scenario} --seed=${seed} \
    --run_num=${script_index} --log_dir=${log_dir}

done

# ##### #
# PLOTS #
# ##### #

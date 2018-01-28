#!/usr/bin/env bash

# #### #
# VARS #
# #### #
scenario='reacher_gps2'
log_dir='gps_log2'

seeds=(0 50 100)
init_index=0
end_index=1

# #### #
# RUNS #
# #### #
seeds=("${seeds[@]:${init_index}:${end_index}}")
total_seeds=${#seeds[@]}

echo "Reacher GPS"
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

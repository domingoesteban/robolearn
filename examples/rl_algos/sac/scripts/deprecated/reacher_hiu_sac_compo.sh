#!/usr/bin/env bash

# #### #
# VARS #
# #### #
expt_name='objective_test'
env_name='reacher'
algo_name='hiu_sac_compo'
dir_prefix=${algo_name}X3_0sha_elu_mix
#dir_prefix=${algo_name}

python_script=${env_name}'_'${algo_name}
log_dir_path='/home/desteban/logs/'${expt_name}'/'${env_name}'/'

#default_seeds=(610 710 810 910 1010)
default_seeds=(1010 810)
seeds=("${@:-${default_seeds[@]}}")
total_seeds=${#seeds[@]}
#init_index=0
#end_index=3
#seeds=("${seeds[@]:${init_index}:${end_index}}")

#default_subtasks=(0 1 -1)
default_subtasks=(-1)
subtasks=("${@:-${default_subtasks[@]}}")
total_subtasks=${#subtasks[@]}


# Mix weights
#default_mix_weights=(1 5)
default_mix_weights=(0)
mix_weights=("${@:-${default_mix_weights[@]}}")
total_mix_weights=${#mix_weights[@]}


total_scripts=$(($total_seeds * $total_subtasks * $total_mix_weights))

echo "Robolearn DRL script"
echo "Total seeds: ${#seeds[@]}"
echo "Experiment seeds: ${seeds[@]}"
echo ""

for seed_idx in ${!seeds[@]}; do
for subtask_idx in ${!subtasks[@]}; do
for mix_weights_idx in ${!mix_weights[@]}; do
    seed=${seeds[seed_idx]}
    subtask=${subtasks[subtask_idx]}
#    script_index=$((index+init_index))
    script_index=$(((seed_idx)*total_subtasks + subtask_idx))
    echo "********************************************************"
    echo "Running '${python_script}.py' $((script_index+1))/${total_scripts} | Seed: ${seed} Subtask: ${subtask}"

    expt_name='sub'${subtask}_${algo_name}_${seed}
    echo "Log_dir '${log_dir_path}'"

    mix_weight=${mix_weights[mix_weights_idx]}
    log_dir=${log_dir_path}'sub'${subtask}'/'${dir_prefix}'_'${mix_weight}'_'${seed}

    python ../${python_script}.py --seed ${seed} --subtask ${subtask} \
    --log_dir ${log_dir} --expt_name ${env_name} --mix_weight ${mix_weight} --gpu
done
done
done

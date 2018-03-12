#!/usr/bin/env bash

gps_script_numbers=(1 2 3 4 5)

default_seeds=(0 50 100)
seeds=("${@:-${default_seeds[@]}}")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
for index in ${!gps_script_numbers[@]}; do
    number=${gps_script_numbers[index]}
    echo "************************************"
    echo "Running gps_'${number}'"

    ${DIR}/reacher_gps${number}.sh "${seeds[@]}"

done

#!/usr/bin/env bash

gps_numbers=(1 2 3 4 5 6)


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
for index in ${!gps_numbers[@]}; do
    number=${gps_numbers[index]}
    echo "************************************"
    echo "Running gps_'${number}'"

    ${DIR}/reacher_trajopt${number}.sh

done

#!/bin/bash
exp=$1
num_trials=$2

scripts/replica_mono.sh $exp $num_trials
scripts/replica_rgbd.sh $exp $num_trials

scripts/tum_mono.sh $exp $num_trials
scripts/tum_rgbd.sh $exp $num_trials

#!/bin/bash

# This script should do two things
psf=$1
dcd=$2
round=$3
idx=$4
target=$5

# 1. MD simulation given input arguments of round and idx
echo "`date` Running simulations, round $round and idx $idx"
jsrun -n1 -g1 -c1 $NAMDHOME_A13/namd3 +p1 +devices 0 +setcpuaffinity ../Simulations/$round/$idx/BBA.conf > ../Simulations/$round/$idx/BBA.log
echo "`date` simulation done for round $round and idx $idx"

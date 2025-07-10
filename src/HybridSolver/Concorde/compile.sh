#!/bin/bash
cd build
rm -r ./*
cmake -DCMAKE_BUILD_TYPE=Release -DCPLEX_ROOT_DIR=/home/sciandra/opt/ibm/ILOG/CPLEX_Studio2211 ..
make -j4

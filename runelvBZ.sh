#! /bin/bash

SOFT_PATH=/home/commlab/electrophys/ElviraBZ

mpirun -n $1 $SOFT_PATH/bin/mainelv_openmpi_gcc_bz -i $2 -o $3 > runelv.out
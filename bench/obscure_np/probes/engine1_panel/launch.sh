#!/usr/bin/env bash
# usage: launch.sh TAG SCEN [ENV=VAL ...] ; runs panel_dump in background, writes pid to run_<TAG>_<SCEN>.pid
S=${S:?set S to the scratch dir holding e1/}; cd $S/e1/run
TAG=$1; SCEN=$2; shift 2
env "$@" TAG=$TAG nohup python -u panel_dump.py $S/e1/v4b $S/e1/pool $S/train/train.parquet $S/dl $S/e1/run/work $S/e1/run/out $SCEN > ${SCEN}${TAG}.log 2>&1 &
echo $! > run${TAG}_${SCEN}.pid

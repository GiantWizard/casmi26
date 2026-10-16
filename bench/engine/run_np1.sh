#!/bin/bash
# NP-like held-out harness rows for one FP bank: run_np1.sh NAME CKPT [CKPT...]
cd ~/casmi26/rk
NAME=$1; shift
mkdir -p np1_$NAME
C="--v4code $HOME/casmi26/inputs/v4b/code --v4mod $HOME/casmi26/inputs/v4b --work $HOME/casmi26/rk/work --train $HOME/casmi26/data/train.parquet --split $HOME/casmi26/inputs/foldsafe/split.parquet --fold hold --sids-file $HOME/casmi26/rk/hold_np1_sids.parquet --n 4000 --cpu --no-selftest --nshards 4"
for s in 0 1 2 3; do
  $HOME/casmi26/.venv/bin/python -I gen_rows.py $C --shard $s --fpnet "$@" --out np1_$NAME > np1_$NAME/log$s.txt 2>&1 &
done
wait
echo done $NAME

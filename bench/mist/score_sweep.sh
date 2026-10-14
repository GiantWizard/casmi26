#!/bin/bash
export ONP_KEY_CACHE=/Users/b/casmi26/mist/keycache
B=/Users/b/casmi26-repo/bench/obscure_np; P=~/casmi26/.venv/bin/python
DIR=$1; SOL=$2; TEST=$3; BASE=$4
files=($DIR/*.csv); rm -f $DIR/score_all.txt
for ((i=0;i<${#files[@]};i+=12)); do
  SUBS="base=$BASE"
  for f in "${files[@]:i:12}"; do n=$(basename $f .csv); SUBS="$SUBS ${n//./p}=$f"; done
  $P $B/mrr_eval.py --solution $SOL --test $TEST --sub $SUBS --baseline base 2>&1 | grep -E "^  (all|cls=2|cls=3) " >> $DIR/score_all.txt
done

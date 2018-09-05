#!/usr/bin/env bash

if [ -f 'pid.txt' ]
then
    kill -9 `cat pid.txt`
    rm -f pid.txt
fi

nohup python -u tests/train_triplet_network_v2.py &>out&
pid=$!
echo $pid > pid.txt

echo 'start successfully'
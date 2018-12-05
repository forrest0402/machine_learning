#!/usr/bin/env bash

if [ -f 'pid.txt' ]
then
    kill -9 `cat pid.txt`
    rm -f pid.txt
fi

nohup python -u tests/tripletnetwork/train_cosine.py &>out.log&
pid=$!
echo $pid > pid.txt

echo 'start successfully'


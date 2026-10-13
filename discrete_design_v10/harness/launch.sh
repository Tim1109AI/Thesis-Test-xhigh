#!/bin/bash
# usage: launch.sh NAME cmd...   (records the PID in NAME.pid)
name=$1; shift
setsid nohup "$@" > $name.log 2>&1 &
echo $! > $name.pid

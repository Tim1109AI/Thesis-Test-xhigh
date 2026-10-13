#!/bin/bash
# usage: stop.sh NAME  (kills the process group started by launch.sh)
pid=$(cat $1.pid); kill -- -$pid 2>/dev/null || kill $pid

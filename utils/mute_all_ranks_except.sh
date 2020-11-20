#!/bin/bash
# https://stackoverflow.com/a/20569097/4179419

UNMUTE=$1
shift 1

if [ "$OMPI_COMM_WORLD_RANK" == "$UNMUTE" ]; then
  exec $*
else
  exec $* >/dev/null 2>&1
fi

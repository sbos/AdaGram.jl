#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [[ `uname` == 'Darwin' ]]; then
	DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$DIR/lib julia "$@"
elif [[ `uname` == 'Linux' ]]; then
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DIR/lib julia "$@"
fi


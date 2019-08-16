#!/bin/bash

export CC="clang"

command_exists () {
  type "$1" >/dev/null 2>&1;
}

if  ! command_exists clang ; then
	echo "clang is not installed, switching to gcc"
	export CC="gcc"
fi

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
echo "Current DIR: $DIR"
echo "Using CC: $CC"
cd $DIR
mkdir -p ./lib

if [[ `uname` == 'Darwin' ]]; then
	$CC -std=c99 -march=native -dynamiclib ./csrc/learn.c -O3 -lm -o ./lib/superlib.dylib
elif [[ `uname` == 'Linux' ]]; then
	$CC -std=c99 -march=native -shared ./csrc/learn.c -O3 -lm -fPIC -o ./lib/superlib.so
fi

#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

$DIR"/run.sh" $DIR"/classify.jl" "$@"
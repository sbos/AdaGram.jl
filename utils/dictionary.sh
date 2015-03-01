#!/bin/bash

tr ' ' '\n' < $1 | sort -S 10G | uniq -c | awk '{print $2" "$1}' > $2
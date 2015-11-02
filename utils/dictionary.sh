#!/bin/bash

cut -f1 -d$'\t' < $1 | sort -S 10G | uniq -c | awk '{print $2" "$1}' > $2

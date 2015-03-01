#!/bin/bash

tr '[:upper:]' '[:lower:]' < $1 | tr '[:punct:]' ' ' | tr '[:space:]' ' ' | tr -cd 'a-z ' | tr -s ' ' > $2
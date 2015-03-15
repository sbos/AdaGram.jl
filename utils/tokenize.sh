#!/bin/bash

tr '[:upper:]' '[:lower:]' < $1 | tr '[:punct:]' ' ' | tr '[:space:]' ' ' | tr -cd '[:alnum:] ' | tr -s ' ' > $2

#!/bin/bash

java edu.stanford.nlp.process.PTBTokenizer < $1 | sed '/[[:alnum:]]/!d' | tr '[:space:]' ' ' | tr '[:upper:]' '[:lower:]' > $2
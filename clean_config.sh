#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <config.json file>"
    exit 1
fi

cp $1 $1.full
jq 'walk(if type == "object" then del(.scale_dtype, .zp_dtype) else . end)' $1 > tmp.json
mv tmp.json $1
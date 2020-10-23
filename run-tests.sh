#!/usr/bin/env bash
set -e
if [[ -z "${BASH_SOURCE[0]}" ]]; then
    if [[ $SHELL =~ "zsh" ]] || [[ ! -z $ZSH ]]; then
        export SOURCE_DIR=$(realpath $(dirname ${(%):-%N}))
    fi
else
    export SOURCE_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
fi

ROOT=`realpath "$SOURCE_DIR"`
TESTS_ROOT="$ROOT/tests"
echo $TESTS_ROOT

for path in $(find "$TESTS_ROOT" -name '*.py'); do
    echo "Run $path"
    python3 "$path"
done

echo "=========================="
echo "      ALL TESTS OK!"
echo "=========================="

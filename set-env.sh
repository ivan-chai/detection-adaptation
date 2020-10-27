# Use it from shell as ". set-env.sh".
if [[ -z "${BASH_SOURCE[0]}" ]]; then
    if [[ $SHELL =~ "zsh" ]] || [[ ! -z $ZSH ]]; then
        export SOURCE_DIR=$(realpath $(dirname ${(%):-%N}))
    fi
else
    export SOURCE_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
fi

ROOT=`realpath "$SOURCE_DIR"`
export PYTHONPATH="$PYTHONPATH:$ROOT/src"

#!/usr/bin/env bash

export TRAKR_RL_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if ! [[ -z "${CONDA_PREFIX}" ]]; then
    python_exe=${CONDA_PREFIX}/bin/python3.11
else
    echo "[Error] No conda environment activated. Please activate the conda environment first."
    # exit 1
fi


# task env name autocomplete
_ut_rl_lab_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${TRAKR_RL_PATH}/scripts/rsl_rl/train.py 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./trakr_rl.sh"


_ut_setup_conda_env() {

    # copied from isaaclab/_isaac_sim/setup_conda_env.sh
    # add source trakr_rl.sh to conda activate.d
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# show icon if not running headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' \
        '# for trakr_rl' \
        'source '${TRAKR_RL_PATH}'/trakr_rl.sh' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    # we need to setup conda variables to load the binaries
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi
}

# pass the arguments
case "$1" in
    -i|--install)
        git lfs install # ensure git lfs is installed
        pip install -e ${TRAKR_RL_PATH}/source/trakr_rl/
        _ut_setup_conda_env
        activate-global-python-argcomplete
        ;;
    -l|--list)
        shift
        ${python_exe} ${TRAKR_RL_PATH}/scripts/list_envs.py "$@"
        ;;
    -p|--play)
        shift
        ${python_exe} ${TRAKR_RL_PATH}/scripts/rsl_rl/play.py "$@"
        ;;
    -t|--train)
        shift
        ${python_exe} ${TRAKR_RL_PATH}/scripts/rsl_rl/train.py --headless "$@"
        ;;
    *) # unknown option
        ;;
esac

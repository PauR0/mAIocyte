#!/usr/bin/env bash

#Check if the script is running in a venv and deactivate it
INVENV=$(python -c 'import sys; print ("1" if hasattr(sys, "real_prefix") else "0")')
if [[ INVENV -eq 1 ]];then
    echo "exiting current venv"
    deactivate
fi

#The path to the dir where the script is located (IT MUST BE CENTERLINE PATH)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

build_venv () {
    echo "building $1"
    python3 -m venv $1
    source $1/bin/activate
    pip3 install --upgrade pip
    pip3 install wheel
    pip3 install -r "$SCRIPTPATH/requirements.txt"
    deactivate
}

set_act_alias() {

    if alias "act_$1" > /dev/null 2>&1
        then
            echo "Alias act_$1 already exists. The activate alias should be set manually."
        else
            case $SHELL in

                "/bin/zsh")
                    echo "\n alias act_$1='source $SCRIPTPATH/$1/bin/activate'" >> "$HOME/.zshrc"
                    ;;

                "/bin/bash")
                    echo "\n alias act_$1='source $SCRIPTPATH/$1/bin/activate'" >> "$HOME/.bashrc"
                    ;;

                *)
                    echo "Could not set alias to activate. Only zsh and bash are supported"
                    ;;

            esac
    fi

}

if [[ ( $# -eq 1 || ( $# -eq 2 && $1 == -f )) ]];then
    if [[ $# -gt 1 ]];then
        if [[ -d $2 ]];then
            rm -r $2
        fi
            build_venv $2
            set_act_alias $2

    elif [[ ! -d $1 ]];then
        build_venv $1
        set_act_alias $1

    else
        echo "Virtual Environment $1 already exists. Try with -f option to rebuild it from default values."
    fi

else
    echo "Bad usage. Options are install_env [-f] <venv> "
    exit 5
fi
exit 0

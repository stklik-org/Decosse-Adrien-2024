#!/bin/bash

# Variables
PI_USER="donald"
PI_HOST="10.78.42.20"
REMOTE_PATH="~/photo.jpg"
LOCAL_PATH="/home/adrien/Bureau"

if [ $# -gt 2 ]; then
    echo "Usage: $0 <nom_du_raspberry> -c"
    exit 1
else
    if [ "$1" == "mickey" ]; then
        PI_HOST="10.78.42.20"
    else
        if [ "$1" == "minnie" ]; then
            PI_HOST="10.78.42.21"
        else
            echo "Raspberry name should be 'minnie' or 'mickey', but recieved $1"
            exit 1
        fi
    fi
fi


if ! ssh -o ConnectTimeout=5 -q ${PI_USER}@${PI_HOST} exit; then
    echo "Failed to connect to $1"
    exit 1
fi

if [[ $# -eq 2 && "$2" == "-c" ]]; then
    echo "Taking picture in"
    for ((i=10; i>0; i--))
    do
	    echo "$i"
	    sleep 1
    done
    echo "Cheese !"

    ssh ${PI_USER}@${PI_HOST} 'raspistill -o ~/photo.jpg -t 1'

    echo "Picture made, now exporting photo.jpg"

    rsync ${PI_USER}@${PI_HOST}:${REMOTE_PATH} ${LOCAL_PATH}
else
    ssh ${PI_USER}@${PI_HOST} 'raspistill -o ~/photo.jpg -t 1'

    rsync ${PI_USER}@${PI_HOST}:${REMOTE_PATH} ${LOCAL_PATH}
fi


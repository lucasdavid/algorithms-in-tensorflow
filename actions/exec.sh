#!/bin/bash

TAG=${TAG:-latest-gpu}

docker run --gpus all -it --rm \
    -v $(pwd):/tf \
    -v $(pwd)/datasets:/tf/datasets \
    -v $(pwd)/data:/data \
    -v $(pwd)/data/cached/keras:/root/.keras \
    -v $(pwd)/secrets/kaggle:/root/.kaggle \
    -w /tf \
    ldavid/experiments \
    python $1

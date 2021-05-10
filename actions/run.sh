#!/bin/bash -ex

run()
{
  CMD=${1:-up}

  if [[ "$CMD" == "up" ]]; then
    docker-compose -f config/docker-compose.yml up
  elif [[ "$CMD" == "tensorboard" ]]; then
    docker-compose -f config/docker-compose.yml \
                 exec experiments \
                 tensorboard --logdir "${2:-/tf/logs/}" --bind_all
  else
    docker-compose -f config/docker-compose.yml ${@:1}
  fi
}

run $@

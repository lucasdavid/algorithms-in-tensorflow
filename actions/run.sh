#!/bin/bash

run()
{
  CMD=${1:-up}

  if [[ "$CMD" == "up" ]]; then
    docker-compose --env-file config/.env -f config/docker-compose.yml up -d
  elif [[ "$CMD" == "tensorboard" ]]; then
    docker-compose --env-file config/.env -f config/docker-compose.yml \
                 exec experiments \
                 tensorboard --logdir "${2:-/tf/logs/}" --bind_all
  else
    docker-compose --env-file config/.env -f config/docker-compose.yml ${@:1}
  fi
}

run $@

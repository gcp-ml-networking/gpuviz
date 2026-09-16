#!/bin/bash
# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -xe

arg_copy=( "$@" )
build_debug_version=false
arm64=false
while [ $# -gt 0 ]; do
  case $1 in
    --cpu=arm | --cpu=arm64 | --arm)
    arm64=true
    echo "Cross-compiling for arm64"
    ;;
  esac
  shift
done

if ! groups | grep -qw docker; then
  sudo usermod -aG docker $USER
fi
docker build -t gpuviz-container -f Dockerfile .
docker run --rm --detach  --name GPUViz --volume $(pwd)/..:/depbuild -it gpuviz-container:latest
if $arm64; then
  docker exec GPUViz bash scripts/install_arm64_cross_compilation_tools.sh
fi
docker exec GPUViz bash scripts/build_gpuviz_within_container.sh "${arg_copy[@]}"

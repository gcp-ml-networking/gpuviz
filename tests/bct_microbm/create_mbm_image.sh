#!/bin/bash
# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


# Function to display usage
usage() {
  echo "Builds and optionally uploads a container for the micro benchmark."
  echo "Container is called micro_bm gpuviz-mbm-container:VERSION"
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo " -h, --help           Display this message"
  echo " -u, --upload         Tag and upload the container"
  echo " -d, --dont_create    Skip the creation step"
  echo " -v, --version_tag    VERSION Specify what version string should be used"
  echo "                      defaults to \"1.0\""
}

# Default variable values
version=1.0
create_mode=true
upload_mode=false

while [ $# -gt 0 ]; do
  case $1 in
    -h | --help)
      usage
      exit 0
      ;;
    -u | --upload)
      upload_mode=true
      ;;
    -d | --dont_create)
      create_mode=false
      ;;
    -v | --version_tag)
      shift
      version=$1
      ;;
  esac
  shift
done

if ! groups | grep -qw docker; then
  sudo usermod -aG docker $USER
fi

if $create_mode; then
  docker build -t gpuviz-mbm-container:$version -f Dockerfile .
  docker run --name micro_bm --volume $(pwd)/../..:/depbuild gpuviz-mbm-container:$version tests/bct_microbm/run_within_mbm.sh
  docker commit micro_bm gpuviz-mbm-container:$version
  docker rm micro_bm
fi

if $upload_mode; then
  docker tag gpuviz-mbm-container:$version us-docker.pkg.dev/kernel-net-team/gpudirect-tcpx/gpuviz-mbm-container:$version
  docker push us-docker.pkg.dev/kernel-net-team/gpudirect-tcpx/gpuviz-mbm-container:$version
fi

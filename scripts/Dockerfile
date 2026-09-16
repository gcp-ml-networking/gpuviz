# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND='noninteractive'

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y autoremove

RUN apt-get install -y --no-install-recommends \
    git openssh-server wget iproute2 vim libopenmpi-dev build-essential \
    cmake gdb python3 \
    protobuf-compiler libprotobuf-dev rsync libssl-dev libcurl4-openssl-dev

Run apt-get install -y --no-install-recommends openjdk-21-jdk zip unzip

RUN apt-get install -y --reinstall ca-certificates
RUN update-ca-certificates

# Download the proper version of bazelisk for our architecture.
WORKDIR /third_party
RUN wget -O bazelisk-linux https://github.com/bazelbuild/bazelisk/releases/download/v1.24.1/bazelisk-linux-$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/amd64/') && \
    chmod 755 bazelisk-linux && mv bazelisk-linux /usr/local/bin/bazel

WORKDIR /depbuild

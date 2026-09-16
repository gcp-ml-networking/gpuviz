# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND='noninteractive'

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y autoremove

RUN apt-get install -y --no-install-recommends \
        build-essential wget openssl git python3
RUN apt-get install -y --reinstall ca-certificates
RUN update-ca-certificates

# Download the proper version of bazelisk for our architecture.
WORKDIR /third_party
RUN wget -O bazelisk-linux https://github.com/bazelbuild/bazelisk/releases/download/v1.24.1/bazelisk-linux-$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/amd64/') && \
    chmod 755 bazelisk-linux && mv bazelisk-linux /usr/local/bin/bazel

WORKDIR /depbuild

# COPY nccl_micro_benchmark .
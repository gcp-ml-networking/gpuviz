#!/bin/bash
# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# All the other packages (loader, libc, etc.) turn
# out to be dependencies of g++-aarch64-linux-gnu,
# but we include them to be on the safe side.
apt-get install -y g++-aarch64-linux-gnu gcc-aarch64-linux-gnu \
  cpp-aarch64-linux-gnu libc6-dev-arm64-cross binutils-aarch64-linux-gnu

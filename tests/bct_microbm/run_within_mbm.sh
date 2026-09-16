#!/bin/bash
# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

: "${BAZEL:=bazel}"

USE_BAZEL_VERSION=7.4.1

# build GPUViz library.
"$BAZEL" build --config=release //src:GPUViz
# build the micro benchmark
"$BAZEL" build --config=release //tests:gpuviz_micro_benchmark

libNcclMicroBM=$("$BAZEL" cquery //tests:gpuviz_micro_benchmark --output=files | grep "^bazel.*gpuviz_micro_benchmark$")

cp $libNcclMicroBM /usr/local/bin/gpuviz_micro_benchmark

libGPUViz_path=$("$BAZEL" cquery //src:GPUViz --output=files | grep "^bazel.*libGPUViz.so$")

cp $libGPUViz_path /usr/local/lib/libGPUViz.so

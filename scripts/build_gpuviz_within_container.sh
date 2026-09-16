#!/bin/bash
# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -xe

: "${BAZEL:=bazel}"
: "${PROJECT_ROOT:=../}"

USE_BAZEL_VERSION=7.4.1

build_debug_version=false
while [ $# -gt 0 ]; do
  case $1 in
    -d | --debug | --dbg | --build_debug | --build_dbg)
      build_debug_version=true
      echo "Debug version will built"
      ;;
  esac
  case $1 in
    --cpu=arm | --cpu=arm64 | --arm)
    cpu_flags="--cpu=arm64 --platforms=//platforms:linux_arm64 --copt=-fPIC"
    echo "Building for arm64"
    ;;
  esac
  shift
done

# Change into the directory of the parent directory of where this script is located.
cd "$(dirname "$0")"
# Change into the parent directory, which should be project root.
cd "$PROJECT_ROOT"

# build GPUViz library.
"$BAZEL" build --config=release $cpu_flags //src:GPUViz
libGPUViz_path=$("$BAZEL" cquery --config=release $cpu_flags //src:GPUViz --output=files | grep "^bazel.*libGPUViz.so$")

if $build_debug_version; then
  "$BAZEL" build --config=debug $cpu_flags //src:GPUViz
  libGPUViz_path_debug=$("$BAZEL" cquery --config=debug $cpu_flags //src:GPUViz --output=files | grep "^bazel.*libGPUViz.so$")
fi

# locate the built artifact directory of libGPUViz.
libGPUViz_path=$("$BAZEL" cquery $cpu_flags //src:GPUViz --output=files | grep "^bazel.*libGPUViz.so$")

# build python generated code of proto.
"$BAZEL" build --config=release //src/proto:millisecond_bw_file_format_py_proto
"$BAZEL" build --config=release //src/proto:nccl_telemetry_py_proto
"$BAZEL" build --config=release //src/proto:nccl_telemetry_cc_proto

# locate the built artifact directory of python generated code.
libMsBwPyProto=$("$BAZEL" cquery //src/proto:millisecond_bw_file_format_py_proto --output=files | grep "^bazel.*millisecond_bw_file_format_pb2.py$")
libNcclPyProto=$("$BAZEL" cquery //src/proto:nccl_telemetry_py_proto --output=files | grep "^bazel.*nccl_telemetry_proto_pb2.py$")
libNcclHProto=$("$BAZEL" cquery //src/proto:nccl_telemetry_cc_proto --output=files | grep "^bazel.*nccl_telemetry_proto.pb.h$")
libNcclCCProto=$("$BAZEL" cquery //src/proto:nccl_telemetry_cc_proto --output=files | grep "^bazel.*nccl_telemetry_proto.pb.cc$")
mkdir -p build/src/proto/

cp $libGPUViz_path build/libGPUViz.so
cp $libMsBwPyProto build/src/proto/millisecond_bw_file_format_pb2.py
cp $libNcclPyProto build/src/proto/nccl_telemetry_proto_pb2.py
cp $libNcclCCProto build/src/proto/nccl_telemetry_proto.pb.cc
cp $libNcclHProto build/src/proto/nccl_telemetry_proto.pb.h
($build_debug_version && cp $libGPUViz_path_debug build/libGPUViz-dbg.so) || true

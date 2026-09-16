#!/bin/bash
# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

git config --global --add safe.directory /depbuild
GIT_VER=$(git describe --tag)
echo '#define GIT_VER "'${GIT_VER}'"' > src/git_tag.h

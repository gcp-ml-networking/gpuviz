// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <dlfcn.h>
#include "gtest/gtest.h"

TEST(GPUVizDlTest, VerifyExports) {
  // Load the locally compiled GPUViz shared library from Bazel data runfiles path
  const char* path = "src/libGPUViz.so";
  void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    // Fallback paths for direct local execution outside bazel sandbox
    handle = dlopen("../build/libGPUViz.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (!handle) {
    handle = dlopen("build/libGPUViz.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (!handle) {
    handle = dlopen("libGPUViz.so", RTLD_NOW | RTLD_LOCAL);
  }

  ASSERT_NE(handle, nullptr)
      << "Failed to load libGPUViz.so. dlerror: "
      << (dlerror() ? dlerror() : "unknown");

  // Verify that the new V4 symbol exists and is exported globally
  void* v4_ptr = dlsym(handle, "nccl_telemetry_stats_plugin_v4");
  EXPECT_NE(v4_ptr, nullptr)
      << "Failed to resolve nccl_telemetry_stats_plugin_v4. dlerror: "
      << (dlerror() ? dlerror() : "none");

  // Verify backward compatibility version symbol entrypoints
  EXPECT_NE(
      dlsym(handle, "nccl_telemetry_stats_plugin_v3"),
      nullptr)
      << "dlerror: " << (dlerror() ? dlerror() : "none");
  EXPECT_NE(
      dlsym(handle, "nccl_telemetry_stats_plugin_v2"),
      nullptr)
      << "dlerror: " << (dlerror() ? dlerror() : "none");
  EXPECT_NE(
      dlsym(handle, "nccl_telemetry_stats_plugin_v1"),
      nullptr)
      << "dlerror: " << (dlerror() ? dlerror() : "none");

  dlclose(handle);
}

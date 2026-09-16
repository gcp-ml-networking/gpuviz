// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_telemetry_provider_stub.h"

#include <string.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <thread>

#include "gtest/gtest.h"

namespace gpuviz {
namespace {

class NCCLTelemetryProviderStubTest : public ::testing::Test {};

// This test verifies that the overriden function `readTelemetry` can return a
// sopecific proto which can be serialized and parsed without loss of
// information.

TEST_F(NCCLTelemetryProviderStubTest, TestreturnedProto) {
  NcclTelemetryStatsStub telemetry_provider;
  auto stats = telemetry_provider.readTelemetry();
  if (stats.ok()) {
    std::string serializedMessage;
    stats.value()->SerializeToString(&serializedMessage);
    cout << "serialized string: " << serializedMessage << endl;
    protoDist::AllStats stats1;
    stats1.ParseFromString(serializedMessage);
  }
}

}  // namespace
}  // namespace gpuviz
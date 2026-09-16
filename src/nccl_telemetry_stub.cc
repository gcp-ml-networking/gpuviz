// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_telemetry_provider_stub.h"

namespace gpuviz {

// With extern “C” block, C++ compiler ensures that the function
// names are un-mangled – that the compiler emits a binary file
// with their names unchanged, as a C compiler would do
extern "C" {
// The volatile keyword is used to ensure that the compiler does not
// optimize away the plugin.
volatile ncclStatsPlugin_v1_t nccl_telemetry_stats_plugin_v1 = {
    "NcclTelemetryStatsStub_v1",
    NcclTelemetryStatsStub::init,
    NcclTelemetryStatsStub::destroy,
    NcclTelemetryStatsStub::addConnection,
    NcclTelemetryStatsStub::deleteConnection,
    NcclTelemetryStatsStub::notifyOperationMeasurement};
};

}  // namespace gpuviz
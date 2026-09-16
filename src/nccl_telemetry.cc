// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_container.h"

namespace gpuviz {

// With extern “C” block, C++ compiler ensures that the function
// names are un-mangled – that the compiler emits a binary file
// with their names unchanged, as a C compiler would do
extern "C" {
// The volatile keyword is used to ensure that the compiler does not
// optimize away the plugin.
__attribute__((visibility("default"))) volatile ncclStatsPlugin_v1_t
    nccl_telemetry_stats_plugin_v1 = {
        "NcclTelemetryStats_v1",
        NcclStatsContainer::init_v1,
        NcclStatsContainer::destroy,
        NcclStatsContainer::addConnection,
        NcclStatsContainer::deleteConnection,
        NcclStatsContainer::notifyOperationMeasurement};

// The volatile keyword is used to ensure that the compiler does not
// optimize away the plugin.
__attribute__((visibility("default"))) volatile ncclStatsPlugin_v2_t
    nccl_telemetry_stats_plugin_v2 = {
        "NcclTelemetryStats_v2",
        NcclStatsContainer::init_v2,
        NcclStatsContainer::destroy,
        NcclStatsContainer::addConnection,
        NcclStatsContainer::deleteConnection,
        NcclStatsContainer::notifyOperationMeasurement,
        NcclStatsContainer::notifyProfilerEvent};

// The volatile keyword is used to ensure that the compiler does not
// optimize away the plugin.
__attribute__((visibility("default"))) volatile ncclStatsPlugin_v3_t
    nccl_telemetry_stats_plugin_v3 = {
        "NcclTelemetryStats_v3",
        NcclStatsContainer::init,
        NcclStatsContainer::destroy,
        NcclStatsContainer::addConnection,
        NcclStatsContainer::deleteConnection,
        NcclStatsContainer::notifyOperationMeasurement,
        NcclStatsContainer::notifyProfilerEvent};

// The volatile keyword is used to ensure that the compiler does not
// optimize away the plugin.
__attribute__((visibility("default"))) volatile ncclStatsPlugin_v4_t
    nccl_telemetry_stats_plugin_v4 = {
        "NcclTelemetryStats_v4",
        NcclStatsContainer::init,
        NcclStatsContainer::destroy,
        NcclStatsContainer::addConnection,
        NcclStatsContainer::deleteConnection,
        NcclStatsContainer::notifyOperationMeasurement,
        NcclStatsContainer::notifyProfilerEvent};
};

}  // namespace gpuviz
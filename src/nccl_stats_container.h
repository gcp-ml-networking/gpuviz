/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_STATS_CONTAINER_H_
#define NCCL_STATS_CONTAINER_H_

#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "nccl_stats_aggregator.h"
#include "nccl_stats_reporter.h"

namespace gpuviz {

class NcclStatsContainerSingleton;

class NcclStatsContainer {
 public:
  static ncclResult_t init(ncclDebugLogger_t logFunction,
                           uint64_t distributionCollectorBitmap,
                           ncclTelemetryMode defaultTelemetryMode,
                           const char* callerIdentifier, int32_t major,
                           int32_t minor,
                           uintptr_t* statsGlobalHandle /* Out */
  );
  static ncclResult_t init_v2(ncclDebugLogger_t logFunction,
                              uint64_t distributionCollectorBitmap,
                              ncclTelemetryMode defaultTelemetryMode,
                              const char* callerIdentifier,
                              uintptr_t* statsGlobalHandle /* Out */
  );
  static ncclResult_t init_v1(ncclDebugLogger_t logFunction,
                              uint64_t distributionCollectorBitmap,
                              uintptr_t* statsGlobalHandle /* Out */);
  static ncclResult_t destroy(uintptr_t statsGlobalHandle);
  static ncclResult_t addConnection(
      uintptr_t statsGlobalHandle,
      const ncclStatsConnectionIdentifier* connectionIdentifier,
      uintptr_t* statsConnectionHandle);
  static ncclResult_t addConnection(
      uintptr_t statsGlobalHandle,
      const ncclStatsConnectionIdentifier_v4* connectionIdentifier,
      uintptr_t* statsConnectionHandle);
  static ncclResult_t deleteConnection(uintptr_t statsConnectionHandle,
                                       ncclStatsConnectionCloseType closeType,
                                       const char* verboseReason);
  static ncclResult_t notifyOperationMeasurement(
      uintptr_t statsConnectionHandle,
      const ncclStatsOperationMetric* measurement);
  static ncclResult_t notifyProfilerEvent(uintptr_t statsConnectionHandle,
                                          const uint8_t* event_data,
                                          size_t len);
  // Causes the call to absl::InitializeLog to be skipped
  // This method should be called in unit tests when the test framcework
  // is already calling absl::InitializeLog so that we can avoid calling
  // it a second time.  Does not need to be used in integration tests
  // which use GPUViz as a .so
  static void skipInitializeLog() { run_initialize_log_ = false; };

 private:
  static NcclStatsContainerSingleton& getNcclStatsContainerSingletonInstance();
  static bool run_initialize_log_;
};

}  // namespace gpuviz

#endif  // NCCL_STATS_CONTAINER_H_
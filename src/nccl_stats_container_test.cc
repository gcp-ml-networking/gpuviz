// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_container.h"

#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <thread>

#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"

namespace gpuviz {
namespace {

static void dummyDebugLog(ncclDebugLogLevel level, uint64_t flags,
                          const char* filefunc, int line, const char* fmt,
                          ...) {
  (void)level;
  (void)flags;
  printf("%s:%d ", filefunc, line);
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
  printf("\n");
}

void SetDefaultEnvironmentVariables() {
  setenv(
      "NCCL_NET_PLUGIN_TELEMETRY_MODE",
      absl::StrCat(int(ncclTelemetryMode::kUploadOnlyControlledByMds)).c_str(),
      1 /* overwrite */);
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1",
         1 /* overwrite */);
}

class NcclStatsContainerTest : public ::testing::Test {
  void SetUp() {
    SetDefaultEnvironmentVariables();
    NcclStatsContainer::skipInitializeLog();
  }
};

TEST_F(NcclStatsContainerTest, TestNcclStatsContainer) {
  uintptr_t statsGlobalHandle;
  ncclDebugLogger_t logFunction = dummyDebugLog;
  NcclStatsContainer::init_v2(
      /*logFunction=*/logFunction,
      /*distributionCollectorBitmap=*/(RecvLatencySW | SendLatencySW),
      /*defaultTelemetryMode=*/ncclTelemetryMode::kUploadOnlyControlledByMds,
      /*callerName=*/"test",
      /*statsGlobalHandle=*/&statsGlobalHandle);
  uintptr_t statsGlobalHandle2;
  NcclStatsContainer::init_v2(
      /*logFunction=*/logFunction,
      /*distributionCollectorBitmap=*/(RecvLatencySW | SendLatencySW),
      /*defaultTelemetryMode=*/ncclTelemetryMode::kUploadOnlyControlledByMds,
      /*callerName=*/"test",
      /*statsGlobalHandle=*/&statsGlobalHandle2);

  // Assert that both the previous init got the same statsGlobalHandle because
  // of singleton implementation of init.
  ASSERT_EQ(statsGlobalHandle2, statsGlobalHandle);

  // Call to destroy shouldn't destroy statsGlobalHandle
  NcclStatsContainer::destroy(statsGlobalHandle);
  // Adding connection returns success
  uintptr_t statsConnectionHandle;
  ncclStatsConnectionIdentifier connectionIdentifier;
  connectionIdentifier.nccl_plugin_name = "StubPlugin";
  connectionIdentifier.nccl_plugin_type = NetPlugin;
  connectionIdentifier.conn_type = EntityTCPConnection;
  connectionIdentifier.gpu_pci_addr = "0000:8C:00.0";
  ASSERT_EQ(NcclStatsContainer::addConnection(
                statsGlobalHandle,
                /*connectionIdentifier=*/&connectionIdentifier,
                /*statsConnectionHandle=*/&statsConnectionHandle),
            ncclSuccess);
  // Adding connection to statsGlobalHandle2 returns success
  uintptr_t statsConnectionHandle2;
  ASSERT_EQ(NcclStatsContainer::addConnection(
                statsGlobalHandle2,
                /*connectionIdentifier=*/&connectionIdentifier,
                /*statsConnectionHandle=*/&statsConnectionHandle2),
            ncclSuccess);
  NcclStatsContainer::deleteConnection(
      statsConnectionHandle, ConnectionCloseLocalTerminate, "Normal terminal");
  NcclStatsContainer::deleteConnection(
      statsConnectionHandle2, ConnectionCloseLocalTerminate, "Normal terminal");
  // Call to destroy will destruct statsGlobalHandle2
  NcclStatsContainer::destroy(statsGlobalHandle2);
}

TEST_F(NcclStatsContainerTest, MultiThreadInitialization) {
  uintptr_t statsGlobalHandle[10];
  std::thread thread_arr[10];
  ncclDebugLogger_t logFunction = dummyDebugLog;
  for (int i = 0; i < 10; i++) {
    std::thread thread([&, i] {
      EXPECT_EQ(NcclStatsContainer::init_v2(
                    logFunction,
                    (RecvLatencySW | SendLatencySW | SendMessageSize |
                     RecvMessageSize),
                    /*defaultTelemetryMode=*/
                    ncclTelemetryMode::kUploadOnlyControlledByMds,
                    /*callerName=*/"test", &statsGlobalHandle[i]),
                ncclSuccess);
    });
    thread_arr[i] = std::move(thread);
  }
  for (int i = 0; i < 10; i++) {
    thread_arr[i].join();
  }
  for (int i = 1; i < 10; i++) {
    ASSERT_EQ(statsGlobalHandle[i], statsGlobalHandle[i - 1]);
  }
  for (int i = 0; i < 10; i++) {
    std::thread thread(NcclStatsContainer::destroy, statsGlobalHandle[i]);
    thread_arr[i] = std::move(thread);
  }
  for (int i = 0; i < 10; i++) {
    thread_arr[i].join();
  }
}

TEST_F(NcclStatsContainerTest,
       TestNcclStatsInitializationWithDifferentArguments) {
  uintptr_t statsGlobalHandle;
  ncclDebugLogger_t logFunction = dummyDebugLog;
  EXPECT_EQ(NcclStatsContainer::init_v2(
                /*logFunction=*/logFunction,
                /*distributionCollectorBitmap=*/(RecvLatencySW | SendLatencySW),
                /*defaultTelemetryMode=*/
                ncclTelemetryMode::kUploadOnlyControlledByMds,
                /*callerName=*/"test",
                /*statsGlobalHandle=*/&statsGlobalHandle),
            ncclSuccess);
  uintptr_t statsGlobalHandle2;
  EXPECT_EQ(
      NcclStatsContainer::init_v2(/*logFunction=*/logFunction,
                               /*distributionCollectorBitmap=*/(RecvLatencySW),
                               /*defaultTelemetryMode=*/
                               ncclTelemetryMode::kUploadOnlyControlledByMds,
                               /*callerName=*/"test",
                               /*statsGlobalHandle=*/&statsGlobalHandle2),
      ncclInvalidArgument);
  EXPECT_EQ(NcclStatsContainer::destroy(statsGlobalHandle), ncclSuccess);
}

TEST_F(NcclStatsContainerTest, MultiThreadDestroy) {
  std::thread thread_arr[10];
  for (int i = 0; i < 10; i++) {
    std::thread thread([] {
      uintptr_t statsGlobalHandle;
      ncclDebugLogger_t logFunction = dummyDebugLog;
      ASSERT_EQ(NcclStatsContainer::init_v2(
                    logFunction,
                    (RecvLatencySW | SendLatencySW | SendMessageSize |
                     RecvMessageSize),
                    /*defaultTelemetryMode=*/
                    ncclTelemetryMode::kUploadOnlyControlledByMds,
                    /*callerName=*/"test", &statsGlobalHandle),
                ncclSuccess);
      uintptr_t statsConnectionHandle;
      ncclStatsConnectionIdentifier connectionIdentifier;
      connectionIdentifier.nccl_plugin_name = "StubPlugin";
      connectionIdentifier.nccl_plugin_type = NetPlugin;
      connectionIdentifier.conn_type = EntityTCPConnection;
      connectionIdentifier.gpu_pci_addr = "0000:8C:00.0";
      ASSERT_EQ(
          NcclStatsContainer::addConnection(
              statsGlobalHandle, &connectionIdentifier, &statsConnectionHandle),
          ncclSuccess);

      ncclStatsOperationMetric operation_metric;
      ncclStatsLatencyMeasurement latency_measurement = {
          .latency_type = LatencySoftware, .latency_in_nanoseconds = 24};
      operation_metric.measurements = &latency_measurement;
      operation_metric.num_measurements = 1;
      operation_metric.op_sz = 2;
      operation_metric.type = OperationTypeChunkSend;

      ASSERT_EQ(NcclStatsContainer::notifyOperationMeasurement(
                    statsConnectionHandle, &operation_metric),
                ncclSuccess);

      ASSERT_EQ(NcclStatsContainer::deleteConnection(
                    statsConnectionHandle, ConnectionCloseLocalTerminate,
                    "Normal terminal"),
                ncclSuccess);

      ASSERT_EQ(NcclStatsContainer::destroy(statsGlobalHandle), ncclSuccess);
    });
    thread_arr[i] = std::move(thread);
  }
  for (int i = 0; i < 10; i++) {
    thread_arr[i].join();
  }
}

TEST_F(NcclStatsContainerTest, TestAddConnectionV4WithGpuUuid) {
  uintptr_t statsGlobalHandle;
  ncclDebugLogger_t logFunction = dummyDebugLog;
  ASSERT_EQ(NcclStatsContainer::init_v2(
                logFunction, (SendLatencySW | SendMessageSize),
                ncclTelemetryMode::kUploadOnlyControlledByMds, "test",
                &statsGlobalHandle),
            ncclSuccess);

  uintptr_t statsConnectionHandle;
  ncclStatsConnectionIdentifier_v4 conn_v4 = {};
  conn_v4.conn_type = EntityProfilerPluginConnection;
  conn_v4.nccl_plugin_type = ProfilerPlugin;
  conn_v4.nccl_plugin_name = "profiler";
  conn_v4.gpu_pci_addr = "0000:90:00.0";
  conn_v4.gpu_uuid = "GPU-41d4f771-166f-c831-5d9e-1f3671e1439d";
  conn_v4.connection.profiler_conn.is_nvl_telemetry = true;
  conn_v4.connection.profiler_conn.channel_id = 2;

  ASSERT_EQ(NcclStatsContainer::addConnection(
                statsGlobalHandle, &conn_v4, &statsConnectionHandle),
            ncclSuccess);
  ASSERT_NE(statsConnectionHandle, 0);

  ASSERT_EQ(NcclStatsContainer::deleteConnection(
                statsConnectionHandle, ConnectionCloseLocalTerminate,
                "Normal termination"),
            ncclSuccess);
  ASSERT_EQ(NcclStatsContainer::destroy(statsGlobalHandle), ncclSuccess);
}

TEST_F(NcclStatsContainerTest, TestInitV3WithPluginVersion) {
  uintptr_t statsGlobalHandle;
  ncclDebugLogger_t logFunction = dummyDebugLog;
  ASSERT_EQ(NcclStatsContainer::init(
                /*logFunction=*/logFunction,
                /*distributionCollectorBitmap=*/(SendLatencySW | SendMessageSize),
                /*defaultTelemetryMode=*/ncclTelemetryMode::kUploadOnlyControlledByMds,
                /*callerIdentifier=*/"test_v3",
                /*major=*/1, /*minor=*/0,
                /*statsGlobalHandle=*/&statsGlobalHandle),
            ncclSuccess);
  ASSERT_NE(statsGlobalHandle, 0);
  ASSERT_EQ(NcclStatsContainer::destroy(statsGlobalHandle), ncclSuccess);
}

}  // namespace
}  // namespace gpuviz
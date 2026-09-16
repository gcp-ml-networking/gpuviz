// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_aggregator.h"

#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <thread>

#include "absl/status/status_matchers.h"
#include "google/protobuf/message_lite.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"
#include "params.h"
#include "utils.h"

using absl_testing::IsOk;

namespace gpuviz {
namespace {

constexpr uint64_t kMicroSecondsPerSecond = 1000000;
constexpr uint64_t kNanoSecondsPerSecond = 1000000000;

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
std::string splitStringFromLastColon(const std::string& input) {
  size_t colonPos = input.rfind(':');  // Find the last colon

  if (colonPos != std::string::npos) {
    return input.substr(0, colonPos);  // Return substring before the last colon
  } else {
    return input;  // Return the original string if ":" is not found
  }
}
void verifyHistogram(const protoDist::StatsDistribution& hist,
                     double max_value_hist, double min_value_hist,
                     const std::unordered_map<double, int>& hist_map) {
  ASSERT_EQ(hist.min(), min_value_hist);
  ASSERT_EQ(hist.max(), max_value_hist);
  double lower_bound = 0;
  for (int i = 0; i < hist.bucket_counts().size(); i++) {
    double upper_bound = std::pow(1.2, i) * 1;
    for (auto e : hist_map) {
      if (lower_bound <= e.first && upper_bound >= e.first) {
        ASSERT_EQ(hist.bucket_counts()[i], e.second);
      }
    }

    lower_bound = upper_bound;
  }
}
void verifyReadTelemetry(NcclStatsAggregator* aggregator, std::string ip_port,
                         double max_value_hist, double min_value_hist,
                         const std::unordered_map<double, int>& hist_map) {
  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<protoDist::AllStats> all_stats = *std::move(stats);
  int totalSendBWDistribution = 0;
  for (auto conn : all_stats->conn_stats()) {
    for (auto hist : conn.histogram()) {
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              ip_port) {
        totalSendBWDistribution++;
        verifyHistogram(hist, max_value_hist, min_value_hist, hist_map);
      }
      std::string ip_addr = splitStringFromLastColon(ip_port);
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              ip_addr) {
        totalSendBWDistribution++;
        verifyHistogram(hist, max_value_hist, min_value_hist, hist_map);
      }
    }
  }
  EXPECT_EQ(totalSendBWDistribution, 2);  // Connection level and NIC level
}

class NCCLStatsAggregatorTest : public ::testing::Test {
 protected:
  NCCLStatsAggregatorTest() {}
  ~NCCLStatsAggregatorTest() override {}
  void SetUp() override {}
  void TearDown() override { clearenv(); }
};

TEST_F(NCCLStatsAggregatorTest, TestLogging) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator =
      std::make_unique<NcclStatsAggregator>(logFunction, RecvLatencyNetHW);
  std::string test_msg = "Test log message";
  aggregator->addLogMsg(test_msg, NCCL_LOG_INFO, __PRETTY_FUNCTION__, __LINE__);

  auto stats = aggregator->readTelemetry();
  std::unique_ptr<google::protobuf::Message> stats_val =
      std::move(stats.value());

  google::protobuf::Message* msg = stats_val.get();
  auto all_stats = google::protobuf::DynamicCastMessage<protoDist::AllStats>(msg);

  for (auto log_entry : all_stats->events_log()) {
    ASSERT_EQ(log_entry.event_message(), "TELEMETRY/GPUViz: " + test_msg);
  }
}

// Test case for successfully adding an event
TEST_F(NCCLStatsAggregatorTest, AddEventSuccess) {
  auto aggregator =
      std::make_unique<NcclStatsAggregator>(dummyDebugLog, RecvLatencyNetHW);
  // Create a sample Event protobuf object.
  protoDist::Event test_event;
  test_event.set_telemetry_type("TestType");
  test_event.mutable_heartbeat();  // Add a heartbeat to make it non-empty

  // Serialize the event to a byte vector.
  std::vector<uint8_t> serialized_event(test_event.ByteSizeLong());
  ASSERT_TRUE(test_event.SerializeToArray(serialized_event.data(),
                                          serialized_event.size()));

  // Act: Call the addEvent function.
  absl::Status status =
      aggregator->addEvent(serialized_event.data(), serialized_event.size());
  ASSERT_TRUE(status.ok());

  // Assert: Read the telemetry and verify the event was added correctly.
  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<google::protobuf::Message> stats_val =
      std::move(stats.value());
  auto all_stats = google::protobuf::DynamicCastMessage<protoDist::AllStats>(stats_val.get());

  ASSERT_EQ(all_stats->events_size(), 1);
  const auto& result_event = all_stats->events(0);
  ASSERT_EQ(result_event.telemetry_type(), "TestType");
  ASSERT_TRUE(result_event.has_heartbeat());
}

TEST_F(NCCLStatsAggregatorTest, TestInitialization) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;
  // Adding connection on initialized aggregator should succeed.
  EXPECT_THAT(aggregator->init(), IsOk());
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  EXPECT_TRUE(statsConnectionHandle.ok());
  EXPECT_THAT(aggregator->deleteConnection(statsConnectionHandle.value(),
                                           ConnectionCloseLocalTerminate,
                                           "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestNicSpeedInitializationFastrak) {
  clearenv();

  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);

  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "FasTrak";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);

  std::pair<double, double> BWPerNIC;
  BWPerNIC = gpuviz::params::GetMaxAndTargetBWPerNICInBitsPerSec(
      aggregator->GetNicSpeeds().first, aggregator->GetNicSpeeds().second);
  EXPECT_TRUE(BWPerNIC.first == 2e+11);
  EXPECT_TRUE(BWPerNIC.second == 2e+11);

  params::BucketerParams bucketer_params =
      params::GetBucketerParamsForNicBWHistogram(
          aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 2e+11);

  bucketer_params =
      params::GetBucketerParamsForBWHistogram(aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 2e+11);

  bucketer_params = params::GetBucketerParamsForNicBWHistogram(
      aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 2e+11);

  bucketer_params = params::GetBucketerParamsForNicBWVarianceHistogram(
      aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 2e+11);

  EXPECT_THAT(aggregator->deleteConnection(statsConnectionHandle.value(),
                                           ConnectionCloseLocalTerminate,
                                           "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestNicSpeedInitializationNetIb) {
  clearenv();

  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);

  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "net-ib";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);

  std::pair<double, double> BWPerNIC;
  BWPerNIC = gpuviz::params::GetMaxAndTargetBWPerNICInBitsPerSec(
      aggregator->GetNicSpeeds().first, aggregator->GetNicSpeeds().second);
  EXPECT_TRUE(BWPerNIC.first == 4e+11);
  EXPECT_TRUE(BWPerNIC.second == 3.85e+11);

  params::BucketerParams bucketer_params =
      params::GetBucketerParamsForNicBWHistogram(
          aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 4e+11);

  bucketer_params =
      params::GetBucketerParamsForBWHistogram(aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 4e+11);

  bucketer_params = params::GetBucketerParamsForNicBWHistogram(
      aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 4e+11);

  bucketer_params = params::GetBucketerParamsForNicBWVarianceHistogram(
      aggregator->GetNicSpeeds().first);
  EXPECT_EQ(bucketer_params.max_buckets, 4e+11);

  EXPECT_THAT(aggregator->deleteConnection(statsConnectionHandle.value(),
                                           ConnectionCloseLocalTerminate,
                                           "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestLocalIpv4) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;

  ncclStatsTCPConnection connection;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("127.0.0.1");

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("128.0.0.2");

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;
  connection_identifier.connection.tcp_conn = connection;

  // Adding connection on initialized aggregator should succeed.
  ASSERT_THAT(aggregator->init(), IsOk());
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());
  auto connectionHandle = statsConnectionHandle.value();
  EXPECT_EQ(connectionHandle->getLocalIp(), "127.0.0.1");
  EXPECT_THAT(aggregator->deleteConnection(statsConnectionHandle.value(),
                                           ConnectionCloseLocalTerminate,
                                           "Normal terminal"),
              IsOk());
}
TEST_F(NCCLStatsAggregatorTest, TestLocalIpv6) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;

  ncclStatsTCPConnection connection;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  sockaddr_in6* ipv6Addr = reinterpret_cast<sockaddr_in6*>(&local_addr);
  ipv6Addr->sin6_family = AF_INET6;
  inet_pton(AF_INET6, "2001:db8::10", &(ipv6Addr->sin6_addr));
  ipv6Addr->sin6_port = htons(90);  // Set the port number in network byte order

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("128.0.0.2");

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;
  connection_identifier.connection.tcp_conn = connection;

  // Adding connection on initialized aggregator should succeed.
  ASSERT_THAT(aggregator->init(), IsOk());
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());
  auto connectionHandle = statsConnectionHandle.value();
  EXPECT_EQ(connectionHandle->getLocalIp(), "2001:db8::10");
  EXPECT_THAT(aggregator->deleteConnection(statsConnectionHandle.value(),
                                           ConnectionCloseLocalTerminate,
                                           "Normal terminal"),
              IsOk());
}
TEST_F(NCCLStatsAggregatorTest, ThreadMultiThreadedConnectionManagement) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvLatencyNetHW | RecvLatencySW | SendLatencySW | SendMessageSize);

  EXPECT_THAT(aggregator->init(), IsOk());

  std::thread thread_arr[10];
  for (int i = 0; i < 10; i++) {
    std::thread thread([&] {
      ncclStatsConnectionIdentifier connection_identifier;
      connection_identifier.nccl_plugin_name = "StubPlugin";
      connection_identifier.conn_type = EntityTCPConnection;
      connection_identifier.gpu_pci_addr = "0000:8C:00.0";
      connection_identifier.nccl_plugin_type = NetPlugin;
      auto statsConnectionHandle =
          aggregator->addConnection(&connection_identifier);
      ASSERT_TRUE(statsConnectionHandle.ok());

      ncclStatsOperationMetric operation_metric;
      ncclStatsLatencyMeasurement latency_measurement = {
          .latency_type = LatencySoftware, .latency_in_nanoseconds = 24};
      operation_metric.measurements = &latency_measurement;
      operation_metric.num_measurements = 1;
      operation_metric.op_sz = 2;
      operation_metric.type = OperationTypeChunkSend;

      auto connectionHandle = statsConnectionHandle.value();
      ASSERT_THAT(
          connectionHandle->notifyOperationMeasurement(&operation_metric),
          IsOk());

      ASSERT_THAT(
          aggregator->deleteConnection(
              (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
              ConnectionCloseLocalTerminate, "Normal terminal"),
          IsOk());
    });
    thread_arr[i] = std::move(thread);
  }
  for (int i = 0; i < 10; i++) {
    thread_arr[i].join();
  }
}

TEST_F(NCCLStatsAggregatorTest, TestReadBeforeWrite) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvLatencyNetHW | RecvLatencySW | SendLatencySW | SendMessageSize);

  EXPECT_THAT(aggregator->init(), IsOk());
  auto lastTelemetryStats = aggregator->readTelemetry();
  ASSERT_THAT(lastTelemetryStats.status(), IsOk());

  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "FasTrak";
  connection_identifier.nccl_plugin_type = NetPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());

  ncclStatsOperationMetric operation_metric;
  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware, .latency_in_nanoseconds = 24};
  operation_metric.measurements = &latency_measurement;
  operation_metric.num_measurements = 1;
  operation_metric.op_sz = 2;
  operation_metric.type = OperationTypeChunkSend;

  auto connectionHandle = statsConnectionHandle.value();
  ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
              IsOk());
  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());

  lastTelemetryStats = aggregator->readTelemetry();
  ASSERT_THAT(lastTelemetryStats.status(), IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestMultiThreadReadAndWrite) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvLatencyNetHW | RecvLatencySW | SendLatencySW | SendMessageSize);

  EXPECT_THAT(aggregator->init(), IsOk());

  bool exit_thread = false;
  std::thread thread([&] {
    while (!exit_thread) {
      auto lastTelemetryStats = aggregator->readTelemetry();
      ASSERT_THAT(lastTelemetryStats.status(), IsOk());
    }
  });

  std::thread thread_arr[10];
  for (int i = 0; i < 10; i++) {
    std::thread thread([&] {
      for (int j = 0; j < 100; j++) {
        ncclStatsConnectionIdentifier connection_identifier;
        connection_identifier.nccl_plugin_name = "StubPlugin";
        connection_identifier.nccl_plugin_type = NetPlugin;
        connection_identifier.conn_type = EntityTCPConnection;
        connection_identifier.gpu_pci_addr = "0000:8C:00.0";
        connection_identifier.nccl_plugin_type = NetPlugin;
        auto statsConnectionHandle =
            aggregator->addConnection(&connection_identifier);
        ASSERT_TRUE(statsConnectionHandle.ok());

        ncclStatsOperationMetric operation_metric;
        ncclStatsLatencyMeasurement latency_measurement = {
            .latency_type = LatencySoftware, .latency_in_nanoseconds = 24};
        operation_metric.measurements = &latency_measurement;
        operation_metric.num_measurements = 1;
        operation_metric.op_sz = 2;
        operation_metric.type = OperationTypeChunkSend;

        auto connectionHandle = statsConnectionHandle.value();
        ASSERT_THAT(
            connectionHandle->notifyOperationMeasurement(&operation_metric),
            IsOk());
        ASSERT_THAT(
            aggregator->deleteConnection(
                (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                ConnectionCloseLocalTerminate, "Normal terminal"),
            IsOk());
      }
    });
    thread_arr[i] = std::move(thread);
  }
  for (int i = 0; i < 10; i++) {
    thread_arr[i].join();
  }
  exit_thread = true;

  thread.join();
}

TEST_F(NCCLStatsAggregatorTest, DestructionWithOpenConnections) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);

  EXPECT_THAT(aggregator->init(), IsOk());

  std::thread thread_arr[10];
  for (int i = 0; i < 10; i++) {
    std::thread thread([&] {
      ncclStatsConnectionIdentifier connection_identifier;
      connection_identifier.nccl_plugin_name = "StubPlugin";
      connection_identifier.nccl_plugin_type = NetPlugin;
      connection_identifier.conn_type = EntityTCPConnection;
      connection_identifier.gpu_pci_addr = "0000:8C:00.0";
      connection_identifier.nccl_plugin_type = NetPlugin;
      auto statsConnectionHandle =
          aggregator->addConnection(&connection_identifier);
      ASSERT_TRUE(statsConnectionHandle.ok());
    });
    thread_arr[i] = std::move(thread);
  }
  for (int i = 0; i < 10; i++) {
    thread_arr[i].join();
  }
}

TEST_F(NCCLStatsAggregatorTest, TestAggregateBWStats) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  setenv("NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT", "5", 2);              // 5 bucket
  setenv("NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS", "1000", 2);  // 1 second
  setenv("NCCL_GPUVIZ_JITTER_BW_BUCKET_COUNT", "0", 2);             // No Jitter
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize);

  ASSERT_THAT(aggregator->init(), IsOk());
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "FasTrak";
  connection_identifier.nccl_plugin_type = NetPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;

  ncclStatsTCPConnection connection;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("127.0.0.1");

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("128.0.0.2");

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;
  connection_identifier.connection.tcp_conn = connection;
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());
  /* The first processHighFrequencyTelemetry is dummy and BW bucket which is
  configured to be of size 5 will look like: 0 0 0 0 0
  */
  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  /*
  This loop call first notifyOperationMeasurement with size 10 bytes in first
  second with latency of 1 second & type OperationTypeChunkSend, and then after
  4 second calls notifyOperationMeasurement of 10 bytes with latency of 5 second
  & type OperationTypeChunkSend. And then finally calls
  processHighFrequencyTelemetry in each loop. This loop runs for 4 times. After
  each loop BW bucket of Send direction which is configured to be of size 5 and
  each bucket is of size 1 second is expected to look like this:

  12 2 2 2 2

  Converting the above size in bytes to BW in bits/second will look like this:

  96000 16000 16000 16000 16000

  The first notifyOperationMeasurement add 10 bytes to 0th index.
  The second notifyOperationMeasurement adds 2 bytes to all 5 index.

  Aggregation into histogram of type protoDist::protoSendBandwidth after 4
  iterations should look like this:

  Min: 0
  Max: 96000
  Avg: 25600
  Bucket 0-1 Count: 5
  Bucket 1-15725.6 Count: 0
  Bucket 15725.6-18870.7 Count: 16
  Bucket 18870.7-81140.4 Count: 0
  Bucket 81140.4-97368.5 Count: 4
  Bucket 97368.5-2.96752e+08 Count: 0


  */
  for (int i = 0; i < 4; i++) {
    ncclStatsOperationMetric operation_metric;
    ncclStatsLatencyMeasurement latency_measurement = {
        .latency_type = LatencySoftware,
        .latency_in_nanoseconds = kNanoSecondsPerSecond};
    operation_metric.measurements = &latency_measurement;
    operation_metric.num_measurements = 1;
    operation_metric.op_sz = 10000;
    operation_metric.type = OperationTypeChunkSend;
    auto connectionHandle = statsConnectionHandle.value();
    ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
                IsOk());
    usleep(4 * kMicroSecondsPerSecond);

    latency_measurement.latency_in_nanoseconds = 5 * kNanoSecondsPerSecond;
    ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
                IsOk());
    ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  }
  auto stats = aggregator->readTelemetry();
  std::unique_ptr<protoDist::AllStats> all_stats = std::move(stats.value());
  int totalSendBWDistribution = 0;
  for (auto conn : all_stats->conn_stats()) {
    for (auto hist : conn.histogram()) {
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              "127.0.0.1:80") {
        totalSendBWDistribution++;
        ASSERT_EQ(hist.min(), 16000);
        ASSERT_EQ(hist.max(), 96000);
        double lower_bound = 0;
        for (int i = 0; i < hist.bucket_counts().size(); i++) {
          double upper_bound = std::pow(1.2, i) * 1;
          if (lower_bound < 16000 && upper_bound > 16000) {
            ASSERT_EQ(hist.bucket_counts()[i], 16);
          }
          if (lower_bound < 96000 && upper_bound > 96000) {
            ASSERT_EQ(hist.bucket_counts()[i], 4);
          }
          lower_bound = upper_bound;
        }
      }
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              "127.0.0.1") {
        totalSendBWDistribution++;
        ASSERT_EQ(hist.min(), 16000);
        ASSERT_EQ(hist.max(), 96000);
        double lower_bound = 0;
        for (int i = 0; i < hist.bucket_counts().size(); i++) {
          double upper_bound = std::pow(1.2, i) * 1;
          if (lower_bound < 16000 && upper_bound > 16000) {
            ASSERT_EQ(hist.bucket_counts()[i], 16);
          }
          if (lower_bound < 96000 && upper_bound > 96000) {
            ASSERT_EQ(hist.bucket_counts()[i], 4);
          }
          lower_bound = upper_bound;
        }
      }
    }
  }
  EXPECT_EQ(totalSendBWDistribution, 2);  // Connection level and NIC level

  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestAggregateBWStatsWithCarryOverBuckets) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  setenv("NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT", "10", 2);
  setenv("NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS", "1000", 2);
  setenv("NCCL_GPUVIZ_JITTER_BW_BUCKET_COUNT", "0", 2);
  setenv("NCCL_GPUVIZ_MAX_EXPECTED_LATENCY_PER_TRANSACTION_IN_MILLISECONDS",
         "0", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize);

  ASSERT_THAT(aggregator->init(), IsOk());
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "net-ib";
  connection_identifier.nccl_plugin_type = NetPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;

  ncclStatsTCPConnection connection;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("127.0.0.1");

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("128.0.0.2");

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;
  connection_identifier.connection.tcp_conn = connection;
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());
  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  // generate a random size for operation less than 1 million
  uint64_t operation_size = rand() % 1000000;
  uint64_t bw_bucket_time =
      1;  // 1 second, same as NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS
  uint64_t operation_latency = 2;  // 2 seconds

  for (int i = 0; i < 4; i++) {
    ncclStatsOperationMetric operation_metric;
    ncclStatsLatencyMeasurement latency_measurement = {
        .latency_type = LatencySoftware,
        .latency_in_nanoseconds = operation_latency * kNanoSecondsPerSecond};
    operation_metric.measurements = &latency_measurement;
    operation_metric.num_measurements = 1;
    operation_metric.op_sz = operation_size;
    operation_metric.type = OperationTypeChunkSend;
    auto connectionHandle = statsConnectionHandle.value();
    ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
                IsOk());
    // adding sleep ensures that all 10 buckets are valid buckets and not
    // skipped as invalid because of because they remained unused.
    usleep(10 * kMicroSecondsPerSecond);
    ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  }
  // The value is not divided in multiple buckets because there is carry over
  // range.
  double bucket_value = (double)(operation_size * 8);
  // every iteration entire bucket_value is occupied only by 0th idex and
  // remaining 9 buckets are 0. Total 4 iterations aggregated in histogram so 36
  // 0s and 4 bucket_value in histogram.
  std::unordered_map<double, int> hist_map = {{0, 36}, {bucket_value, 4}};
  verifyReadTelemetry(aggregator.get(), "127.0.0.1:80", bucket_value, 0,
                      hist_map);
  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());
  aggregator.reset();

  // Now if we add carry over of 2 sec
  setenv("NCCL_GPUVIZ_MAX_EXPECTED_LATENCY_PER_TRANSACTION_IN_MILLISECONDS",
         "2000", 2);
  aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize);

  ASSERT_THAT(aggregator->init(), IsOk());
  statsConnectionHandle = aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());
  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());

  for (int i = 0; i < 4; i++) {
    ncclStatsOperationMetric operation_metric;
    ncclStatsLatencyMeasurement latency_measurement = {
        .latency_type = LatencySoftware,
        .latency_in_nanoseconds = operation_latency * kNanoSecondsPerSecond};
    operation_metric.measurements = &latency_measurement;
    operation_metric.num_measurements = 1;
    operation_metric.op_sz = operation_size;
    operation_metric.type = OperationTypeChunkSend;
    auto connectionHandle = statsConnectionHandle.value();
    ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
                IsOk());
    // adding sleep ensures that all 10 buckets are valid buckets and not
    // skipped as invalid because of because they remained unused.
    usleep(10 * kMicroSecondsPerSecond);
    ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  }
  hist_map.clear();
  // The value is divided in multiple buckets because there is carry over range.
  double buckets_range = operation_latency / bw_bucket_time;
  bucket_value = (double)(operation_size * 8) / (double)(buckets_range + 1);
  // every iteration entire bucket_value is occupied only by first 3 index and
  // remaining 5 buckets are 0 because of 2 carry over bucketa. Total 4
  // iterations aggregated in histogram so 20 0s and 12 bucket_value in
  // histogram.
  hist_map = {{0, 20}, {bucket_value, 12}};
  verifyReadTelemetry(aggregator.get(), "127.0.0.1:80", bucket_value, 0,
                      hist_map);

  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestAggregateNicBWStats) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  setenv("NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT", "5", 2);              // 5 bucket
  setenv("NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS", "1000", 2);  // 1 second
  setenv("NCCL_GPUVIZ_JITTER_BW_BUCKET_COUNT", "0", 2);             // No Jitter
  setenv("NCCL_GPUVIZ_TARGET_BW_PER_NIC_IN_BITS_PER_SEC", "40000000", 2);
  setenv("NCCL_GPUVIZ_BASELINE_LATENCY_IN_NANOSECOND", "2000000000", 2);
  setenv("NCCL_GPUVIZ_VARIANCE_HISTOGRAM_COLLECTION_ENABLE", "1", 2);

  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize);

  ASSERT_THAT(aggregator->init(), IsOk());
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "net-ib";
  connection_identifier.nccl_plugin_type = NetPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;

  ncclStatsTCPConnection connection;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("127.0.0.1");

  std::vector<int> ports = {60, 70, 80, 90};
  std::vector<absl::StatusOr<NcclStatsConnectionStatistics*>>
      statsConnectionHandle(4);
  for (int i = 0; i < 4; i++) {
    // Remote Endpoint Creation
    struct sockaddr_storage remote_addr;
    memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.ss_family = AF_INET;
    ((struct sockaddr_in*)&remote_addr)->sin_port = htons(ports[i]);
    ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr =
        inet_addr("128.0.0.2");

    connection.local_endpoint = local_addr;
    connection.remote_endpoint = remote_addr;
    connection_identifier.connection.tcp_conn = connection;
    statsConnectionHandle[i] =
        aggregator->addConnection(&connection_identifier);
    ASSERT_TRUE(statsConnectionHandle[i].ok());
  }

  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  /*
  This loop call first notifyOperationMeasurement with size 10 bytes in first
  second with latency of 1 second & type OperationTypeChunkSend, and then after
  4 second calls notifyOperationMeasurement of 10 bytes with latency of 5 second
  & type OperationTypeChunkSend. And then finally calls
  processHighFrequencyTelemetry in each loop. This loop runs for 4 times. After
  each loop BW bucket of Send direction which is configured to be of size 5 and
  each bucket is of size 1 second is expected to look like this:

  12 2 2 2 2

  Converting the above size in bytes to BW in bits/second will look like this:

  96000 16000 16000 16000 16000

  The first notifyOperationMeasurement add 10 bytes to 0th index.
  The second notifyOperationMeasurement adds 2 bytes to all 5 index.

  Aggregation into histogram of type protoDist::protoSendBandwidth after 4
  iterations should look like this:

  Min: 0
  Max: 96000
  Avg: 25600
  Bucket 0-1 Count: 5
  Bucket 1-15725.6 Count: 0
  Bucket 15725.6-18870.7 Count: 16
  Bucket 18870.7-81140.4 Count: 0
  Bucket 81140.4-97368.5 Count: 4
  Bucket 97368.5-2.96752e+08 Count: 0

  */
  uint8_t number_of_iterations = 4;
  uint64_t operation_size = rand() % 1000000;
  operation_size = operation_size * 5;
  uint64_t bw_buckets_count = 5;  // same as NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT
  double min_value_hist_per_connection =
      (operation_size / bw_buckets_count) * 8;
  double min_value_hist_per_connection_times_in_hist =
      number_of_iterations * (bw_buckets_count - 1);
  double max_value_hist_per_connection =
      operation_size * 8 + min_value_hist_per_connection;
  double max_value_hist_per_connection_times_in_hist = number_of_iterations;
  uint8_t num_of_connections = 4;
  double min_value_hist_per_nic =
      min_value_hist_per_connection * num_of_connections;
  double min_value_hist_per_nic_times_in_hist =
      min_value_hist_per_connection_times_in_hist;
  double max_value_hist_per_nic =
      max_value_hist_per_connection * num_of_connections;
  double max_value_hist_per_nic_times_in_hist =
      max_value_hist_per_connection_times_in_hist;
  double target_bw =
      40000000;  // same as NCCL_GPUVIZ_TARGET_BW_PER_NIC_IN_BITS_PER_SEC
  uint8_t expected_variance_hist = 0;
  double bw_variance_hist = target_bw - min_value_hist_per_nic;
  if (bw_variance_hist > 0) {
    expected_variance_hist += 1;  // BW variance hist is per NIC
  }
  double bw_variance_hist_times = min_value_hist_per_connection_times_in_hist;
  double max_operation_latency = 5 * kNanoSecondsPerSecond;
  double baseline_latency =
      2000000000;  // 2 ns, same as NCCL_GPUVIZ_BASELINE_LATENCY_IN_NANOSECOND
  uint64_t max_offered_load_in_bits =
      operation_size * 8 * 2 *
      num_of_connections;  // 8 to convert bits to bytes, 2 from from 2
                           // operations
  double expected_latency =
      (((double)(max_offered_load_in_bits * SECOND_TO_NANOSECOND_MULTIPLIER)) /
       (double)target_bw) +
      baseline_latency;
  double latency_variance_hist = max_operation_latency - expected_latency;
  if (latency_variance_hist > 0) {
    expected_variance_hist +=
        5;  // Latency variance hist is per NIC and 4 connections
  }
  uint64_t latency_variance_hist_times =
      max_value_hist_per_connection_times_in_hist;
  ncclStatsOperationMetric operation_metric;
  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = kNanoSecondsPerSecond};
  operation_metric.measurements = &latency_measurement;
  operation_metric.num_measurements = 1;
  operation_metric.op_sz = operation_size;
  operation_metric.type = OperationTypeChunkSend;
  for (uint8_t i = 0; i < number_of_iterations; i++) {
    for (uint8_t j = 0; j < num_of_connections; j++) {
      latency_measurement.latency_in_nanoseconds = kNanoSecondsPerSecond;
      auto connectionHandle = statsConnectionHandle[j].value();
      ASSERT_THAT(
          connectionHandle->notifyOperationMeasurement(&operation_metric),
          IsOk());
    }
    usleep(4 * kMicroSecondsPerSecond);
    for (uint8_t j = 0; j < num_of_connections; j++) {
      auto connectionHandle = statsConnectionHandle[j].value();
      latency_measurement.latency_in_nanoseconds = 5 * kNanoSecondsPerSecond;
      ASSERT_THAT(
          connectionHandle->notifyOperationMeasurement(&operation_metric),
          IsOk());
    }
    ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  }
  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<protoDist::AllStats> all_stats = *std::move(stats);
  int totalSendBWDistribution = 0;
  int totalVarianceHistograms = 0;
  for (auto conn : all_stats->conn_stats()) {
    for (auto hist : conn.histogram()) {
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              "127.0.0.1:80") {
        totalSendBWDistribution++;
        std::unordered_map<double, int> hist_map = {
            {min_value_hist_per_connection,
             min_value_hist_per_connection_times_in_hist},
            {max_value_hist_per_connection,
             max_value_hist_per_connection_times_in_hist}};
        verifyHistogram(hist, max_value_hist_per_connection,
                        min_value_hist_per_connection, hist_map);
      }
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              "127.0.0.1") {
        totalSendBWDistribution++;
        std::unordered_map<double, int> hist_map = {
            {min_value_hist_per_nic, min_value_hist_per_nic_times_in_hist},
            {max_value_hist_per_nic, max_value_hist_per_nic_times_in_hist}};
        verifyHistogram(hist, max_value_hist_per_nic, min_value_hist_per_nic,
                        hist_map);
      }

      if (hist.dist_type() == protoDist::protoBandwidthVariance) {
        totalVarianceHistograms++;
        std::unordered_map<double, int> hist_map = {
            {bw_variance_hist, bw_variance_hist_times}};
        verifyHistogram(hist, bw_variance_hist, bw_variance_hist, hist_map);
      }
      if (hist.dist_type() == protoDist::protoLatencyVariance) {
        totalVarianceHistograms++;
        std::unordered_map<double, int> hist_map = {
            {latency_variance_hist, latency_variance_hist_times}};
        verifyHistogram(hist, latency_variance_hist, latency_variance_hist,
                        hist_map);
      }
    }
  }
  EXPECT_EQ(totalVarianceHistograms, expected_variance_hist);
  EXPECT_EQ(totalSendBWDistribution, 5);  // Connection level and NIC level

  for (int j = 0; j < 4; j++) {
    ASSERT_THAT(
        aggregator->deleteConnection(
            (NcclStatsConnectionStatistics*)statsConnectionHandle[j].value(),
            ConnectionCloseLocalTerminate, "Normal terminal"),
        IsOk());
  }
}

TEST_F(NCCLStatsAggregatorTest, TestCarryOverNicBW) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  setenv("NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT", "10", 2);
  setenv("NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS", "1000", 2);
  setenv("NCCL_GPUVIZ_JITTER_BW_BUCKET_COUNT", "0", 2);
  setenv("NCCL_GPUVIZ_MAX_BW_PER_NIC_IN_BITS_PER_SEC", "16000", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize);

  ASSERT_THAT(aggregator->init(), IsOk());
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "net-ib";
  connection_identifier.nccl_plugin_type = NetPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;

  ncclStatsTCPConnection connection;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("127.0.0.1");

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("128.0.0.2");

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;
  connection_identifier.connection.tcp_conn = connection;
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());
  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  // generate a random size for operation less than 1 million
  uint64_t operation_size = rand() % 1000000;
  uint64_t operation_latency = 2;  // 2 seconds

  for (int i = 0; i < 4; i++) {
    ncclStatsOperationMetric operation_metric;
    ncclStatsLatencyMeasurement latency_measurement = {
        .latency_type = LatencySoftware,
        .latency_in_nanoseconds = operation_latency * kNanoSecondsPerSecond};
    operation_metric.measurements = &latency_measurement;
    operation_metric.num_measurements = 1;
    operation_metric.op_sz = operation_size;
    operation_metric.type = OperationTypeChunkSend;
    auto connectionHandle = statsConnectionHandle.value();
    ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
                IsOk());
    // NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT is 10 buckets and adding sleep ensures
    // that all 10 buckets are valid buckets and not skipped as invalid because
    // of because they remained unused.
    usleep(10 * kMicroSecondsPerSecond);
    ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  }
  double bucket_value = (double)(operation_size * 8);

  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<protoDist::AllStats> all_stats = *std::move(stats);
  int totalSendBWDistribution = 0;
  for (auto conn : all_stats->conn_stats()) {
    for (auto hist : conn.histogram()) {
      // verifying for connection which has no carry over logic
      std::unordered_map<double, int> hist_map_connection = {{0, 36},
                                                             {bucket_value, 4}};
      double max_value_hist_connection = bucket_value;
      double min_value_hist_connection = 0;
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              "127.0.0.1:80") {
        totalSendBWDistribution++;
        verifyHistogram(hist, max_value_hist_connection,
                        min_value_hist_connection, hist_map_connection);
      }

      // verifying for NIC which has carry over logic
      double max_bw =
          16000;  // same as NCCL_GPUVIZ_MAX_BW_PER_BUCKET_IN_BITS_PER_SEC
      uint64_t bucket_range = std::min(bucket_value / max_bw, (double)10);
      uint64_t num_of_zeros = 10 - bucket_range;
      std::unordered_map<double, int> hist_map_nic = {
          {0, 4 * num_of_zeros}, {max_bw, 4 * bucket_range}};
      double max_value_hist_nic = max_bw;
      double min_value_hist_nic = max_bw;
      if (hist.dist_type() == protoDist::protoSendBandwidth &&
          conn.connection_id().connection().tcp_conn().local_endpoint() ==
              "127.0.0.1") {
        totalSendBWDistribution++;
        verifyHistogram(hist, max_value_hist_nic, min_value_hist_nic,
                        hist_map_nic);
      }
    }
  }
  EXPECT_EQ(totalSendBWDistribution, 2);  // Connection level and NIC level

  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestBWDistributionBucketInitialization) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;
  // distribution bucket is not created for BW collection because
  // SendMessageSize and SendLatencySW is missing while initializing
  // distributionCollectorBitmap
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | RecvMessageSize);
  ASSERT_THAT(aggregator->init(), IsOk());
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.nccl_plugin_type = NetPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  connection_identifier.nccl_plugin_type = NetPlugin;
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());

  ncclStatsOperationMetric operation_metric;
  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = kNanoSecondsPerSecond};
  operation_metric.measurements = &latency_measurement;
  operation_metric.num_measurements = 1;
  operation_metric.op_sz = 1;
  operation_metric.type = OperationTypeChunkRecv;
  auto connectionHandle = statsConnectionHandle.value();
  ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
              IsOk());
  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestGetConnectionStatsProtoFromReadBucket) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize);
  ASSERT_THAT(aggregator->init(), IsOk());
  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.nccl_plugin_type = ProfilerPlugin;
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.gpu_pci_addr = "0000:8C:00.0";
  auto statsConnectionHandle =
      aggregator->addConnection(&connection_identifier);
  ASSERT_TRUE(statsConnectionHandle.ok());

  ncclStatsOperationMetric operation_metric;
  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = kNanoSecondsPerSecond};
  operation_metric.measurements = &latency_measurement;
  operation_metric.num_measurements = 1;
  // first message
  operation_metric.op_sz = 100;
  operation_metric.type = OperationTypeChunkSend;
  auto connectionHandle = statsConnectionHandle.value();
  ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
              IsOk());
  // second message of bigger size
  operation_metric.op_sz = 200;
  ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
              IsOk());

  // process the telemetry & check the generated proto
  ASSERT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());
  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<protoDist::AllStats> all_stats = *std::move(stats);
  for (const auto& conn_stat : all_stats->conn_stats()) {
    for (const auto& hist : conn_stat.histogram()) {
      if (hist.dist_type() == protoDist::protoSendLatencySW) {
        // validate that boundaries are set correctly
        ASSERT_EQ(hist.min_msg_size_bucket(), 100);
        ASSERT_EQ(hist.max_msg_size_bucket(), 200);
      }
    }
  }
  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)statsConnectionHandle.value(),
                  ConnectionCloseLocalTerminate, "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest,
       TestV4ConnectionWithMsBWAndHighFrequencyTelemetry) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 1);
  setenv("NCCL_GPUVIZ_ENABLE_MILLISECOND_BANDWIDTH_OUTPUT", "1", 1);

  ncclDebugLogger_t logFunction = dummyDebugLog;
  gpuviz::MillisecondBandwidthOutput msec_bw;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction,
      RecvMessageSize | RecvLatencySW | SendLatencySW | SendMessageSize,
      &msec_bw);
  ASSERT_THAT(aggregator->init(), IsOk());

  ncclStatsConnectionIdentifier_v4 conn_id_v4 = {};
  conn_id_v4.nccl_plugin_name = "profiler";
  conn_id_v4.nccl_plugin_type = ProfilerPlugin;
  conn_id_v4.conn_type = EntityTCPConnection;
  conn_id_v4.gpu_pci_addr = "0000:8C:00.0";

  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("10.0.0.1");

  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("10.0.0.2");

  conn_id_v4.connection.tcp_conn.local_endpoint = local_addr;
  conn_id_v4.connection.tcp_conn.remote_endpoint = remote_addr;

  auto statsConnectionHandle = aggregator->addConnectionV4(&conn_id_v4);
  ASSERT_TRUE(statsConnectionHandle.ok());

  ncclStatsOperationMetric operation_metric;
  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = kNanoSecondsPerSecond};
  operation_metric.measurements = &latency_measurement;
  operation_metric.num_measurements = 1;
  operation_metric.op_sz = 1024;
  operation_metric.type = OperationTypeChunkSend;

  auto connectionHandle = statsConnectionHandle.value();
  ASSERT_THAT(connectionHandle->notifyOperationMeasurement(&operation_metric),
              IsOk());

  // processHighFrequencyTelemetry calls calculateAndUpdateHistogram and CopyOverBandwidthStats.
  // Validates that msec_bw buffer is allocated for "10.0.0.1" and both Send/Recv bucketers are allocated.
  EXPECT_THAT(aggregator->processHighFrequencyTelemetry(), IsOk());

  EXPECT_THAT(aggregator->deleteConnection(
                  statsConnectionHandle.value(), ConnectionCloseLocalTerminate,
                  "Normal terminal"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestV4ConnectionIdentifierGpuUuidPopulation) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, SendLatencySW | SendMessageSize);
  ASSERT_THAT(aggregator->init(), IsOk());

  ncclStatsConnectionIdentifier_v4 conn_v4 = {};
  conn_v4.conn_type = EntityProfilerPluginConnection;
  conn_v4.nccl_plugin_type = ProfilerPlugin;
  conn_v4.nccl_plugin_name = "profiler";
  conn_v4.gpu_pci_addr = "0000:8F:00.0";
  conn_v4.gpu_uuid = "GPU-c2140440-ada5-a6c6-1c16-1aae1b184382";
  conn_v4.connection.profiler_conn.is_nvl_telemetry = true;
  conn_v4.connection.profiler_conn.channel_id = 14;
  conn_v4.connection.profiler_conn.profiler_conn.local_rank = 0;
  conn_v4.connection.profiler_conn.profiler_conn.remote_or_root_rank = 7;
  conn_v4.connection.profiler_conn.profiler_conn.collective_type = "NvlWaitPeer";

  auto statsConnectionHandle = aggregator->addConnectionV4(&conn_v4);
  ASSERT_TRUE(statsConnectionHandle.ok());

  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = 3500};
  ncclStatsOperationMetric op_metric = {};
  op_metric.type = OperationTypeChunkSend;
  op_metric.num_measurements = 1;
  op_metric.measurements = &latency_measurement;
  op_metric.op_sz = 0;
  auto connHandle = statsConnectionHandle.value();
  ASSERT_THAT(connHandle->notifyOperationMeasurement(&op_metric), IsOk());

  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<protoDist::AllStats> all_stats = *std::move(stats);

  bool found_nvl_conn = false;
  for (const auto& conn_stat : all_stats->conn_stats()) {
    const auto& cid = conn_stat.connection_id();
    if (cid.has_gpu_uuid() &&
        cid.gpu_uuid() == "GPU-c2140440-ada5-a6c6-1c16-1aae1b184382") {
      found_nvl_conn = true;
      EXPECT_EQ(cid.gpu_pci_addr(), "0000:8F:00.0");
      EXPECT_EQ(cid.nccl_plugin_name(), "profiler");
      EXPECT_TRUE(cid.connection().profiler_conn().is_nvl_telemetry());
      EXPECT_EQ(cid.connection().profiler_conn().channel_id(), 14);
      EXPECT_EQ(cid.connection().profiler_conn().local_rank(), 0);
      EXPECT_EQ(cid.connection().profiler_conn().remote_or_root_rank(), 7);
      EXPECT_EQ(cid.connection().profiler_conn().collective_type(),
                "NvlWaitPeer");
    }
  }
  EXPECT_TRUE(found_nvl_conn);

  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)connHandle,
                  ConnectionCloseLocalTerminate, "Normal termination"),
              IsOk());
}

TEST_F(NCCLStatsAggregatorTest, TestV4ConnectionIdentifierNullGpuUuid) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_unique<NcclStatsAggregator>(
      logFunction, SendLatencySW | SendMessageSize);
  ASSERT_THAT(aggregator->init(), IsOk());

  ncclStatsConnectionIdentifier_v4 conn_v4 = {};
  conn_v4.conn_type = EntityProfilerPluginConnection;
  conn_v4.nccl_plugin_type = ProfilerPlugin;
  conn_v4.nccl_plugin_name = "profiler";
  conn_v4.gpu_pci_addr = "0000:8F:00.0";
  conn_v4.gpu_uuid = nullptr;
  conn_v4.connection.profiler_conn.is_nvl_telemetry = true;
  conn_v4.connection.profiler_conn.channel_id = 0;

  auto statsConnectionHandle = aggregator->addConnectionV4(&conn_v4);
  ASSERT_TRUE(statsConnectionHandle.ok());

  ncclStatsLatencyMeasurement latency_measurement = {
      .latency_type = LatencySoftware,
      .latency_in_nanoseconds = 2000};
  ncclStatsOperationMetric op_metric = {};
  op_metric.type = OperationTypeChunkSend;
  op_metric.num_measurements = 1;
  op_metric.measurements = &latency_measurement;
  auto connHandle = statsConnectionHandle.value();
  ASSERT_THAT(connHandle->notifyOperationMeasurement(&op_metric), IsOk());

  auto stats = aggregator->readTelemetry();
  ASSERT_TRUE(stats.ok());
  std::unique_ptr<protoDist::AllStats> all_stats = *std::move(stats);

  ASSERT_EQ(all_stats->conn_stats_size(), 1);
  const auto& cid = all_stats->conn_stats(0).connection_id();
  EXPECT_FALSE(cid.has_gpu_uuid());

  ASSERT_THAT(aggregator->deleteConnection(
                  (NcclStatsConnectionStatistics*)connHandle,
                  ConnectionCloseLocalTerminate, "Normal termination"),
              IsOk());
}

}  // namespace
}  // namespace gpuviz

// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_reporter.h"

#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"
#include "params.h"
#include "test_utils.h"
#include "utils.h"

namespace gpuviz {
namespace {

using namespace ::testing;
using namespace ::gpuviz::utils;
using ::testing::Test;

// Generates fake proto to upload.
static absl::StatusOr<std::unique_ptr<protoDist::AllStats>>
GenerateFakeProtobuf(std::string message = "1", std::string category = "2") {
  auto all_stats = std::make_unique<protoDist::AllStats>();
  protoDist::LogEntry* log = all_stats->add_events_log();
  log->set_event_message(std::move(message));
  log->set_category_name(std::move(category));
  return all_stats;
}

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

class MockTelemetryProvider : public TelemetryProvider {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<protoDist::AllStats>>,
              readTelemetry, (), (override));
  MOCK_METHOD(absl::Status, processHighFrequencyTelemetry, (), (override));
  MOCK_METHOD(void, addLogMsg,
              (absl::string_view message, ncclDebugLogLevel level,
               const char* pretty_func, uint32_t line,
               ncclDebugLogSubSys subsys, std::string category_name,
               int32_t category, int32_t msg_id),
              (override));
  MOCK_METHOD(void, addLogMsg,
              (absl::string_view message, ncclDebugLogLevel level,
               const char* pretty_func, uint32_t line),
              (override));
};

class MockUploader : public UploaderInterface {
 public:
  MockUploader() {}
  ~MockUploader() = default;
  MOCK_METHOD(absl::Status, Upload, (const google::protobuf::Message&, bool),
              (override));
  MOCK_METHOD(absl::Time, NextUploadTime, (), (override));
  MOCK_METHOD(bool, IsDead, (), (override));
  MOCK_METHOD(size_t, GetMaxMessageByteSize, (), (override));
};

void SetDefaultEnvironmentVariables() {
  setenv(
      "NCCL_NET_PLUGIN_TELEMETRY_MODE",
      absl::StrCat(int(ncclTelemetryMode::kUploadOnlyControlledByMds)).c_str(),
      1 /* overwrite */);
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1",
         1 /* overwrite */);
}

class NCCLStatsReporterTest : public ::testing::Test {
  void SetUp() { SetDefaultEnvironmentVariables(); }
};

TEST_F(NCCLStatsReporterTest, TestInitialization) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_shared<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);
  EXPECT_EQ(aggregator->init(), absl::OkStatus());
  auto reporter = std::make_unique<NcclStatsReporter>(aggregator, logFunction,
                                                      nullptr, "off");
  EXPECT_EQ(reporter->init(), absl::OkStatus());
}

TEST_F(NCCLStatsReporterTest, TestInvokeCallbacks) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "1", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "1000000", 2);
  setenv("NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT", "100", 2);
  setenv("NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS", "1", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, nullptr, "off");
  // 9 or 10 times because readTelemetry is configured to be called every 1
  // seconds and the test sleeps for 10 seconds, + 1 to be called after exiting
  // loop.
  EXPECT_CALL(*TelemetryProvider, readTelemetry).Times(testing::AtLeast(10));
  // 99 or 100 times because processHighFrequencyTelemetry is configured to be
  // called every 100 ms, + 1 to be called after exiting loop.
  EXPECT_CALL(*TelemetryProvider, processHighFrequencyTelemetry)
      .Times(testing::AtLeast(100));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  usleep(10000000);  // 10 Seconds
}

TEST_F(NCCLStatsReporterTest, TestInvokeCallbacksWithHighBWOptOut) {
  // opt out the high bandwidth collection.
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "0", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "1000000", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, nullptr, "off");
  EXPECT_CALL(*TelemetryProvider, readTelemetry).Times(testing::AtLeast(1));
  EXPECT_CALL(*TelemetryProvider, processHighFrequencyTelemetry).Times(0);
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(0.2));
}

TEST_F(NCCLStatsReporterTest, TestLogRotation) {
  ncclDebugLogger_t logFunction = dummyDebugLog;
  auto aggregator = std::make_shared<NcclStatsAggregator>(
      logFunction, RecvLatencyNetHW | RecvLatencySW | SendLatencySW);
  EXPECT_EQ(aggregator->init(), absl::OkStatus());
  auto reporter = std::make_unique<NcclStatsReporter>(aggregator, logFunction,
                                                      nullptr, "off");
  EXPECT_EQ(reporter->init(), absl::OkStatus());

  reporter->flushTelemetry();
  usleep(10000000);  // 10 Seconds
  reporter->flushTelemetry();
  setenv("NCCL_GPUVIZ_FILE_ROTATION_INTERVAL_IN_SECONDS", "2", 2);
  reporter->flushTelemetry();
  usleep(3000000);  // 3 Seconds
}

TEST_F(NCCLStatsReporterTest, TestInvokeCallbacksWithUploader) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "0", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "1000000", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();
  absl::Time start_test_time = get_current_monotonic_time();

  EXPECT_CALL(*uploader, IsDead())
      .Times(AtLeast(1))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, Upload(_, _))
      .Times(AtLeast(2))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*uploader, NextUploadTime())
      .Times(AtLeast(3))
      // happy case when the upload is not a blocker.
      .WillRepeatedly(Return(start_test_time));

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");
  EXPECT_CALL(*TelemetryProvider, readTelemetry)
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
}

TEST_F(NCCLStatsReporterTest, TestUploaderSlowDownReadTelemetry) {
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "0", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "1000000", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();
  absl::Time start_test_time = get_current_monotonic_time();

  EXPECT_CALL(*uploader, IsDead())
      .Times(AtLeast(1))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, Upload(_, _))
      .WillOnce(Return(absl::InternalError("error")));
  EXPECT_CALL(*uploader, NextUploadTime())
      .Times(2)
      // Called in the beginning.
      .WillOnce(Return(start_test_time))
      // The first upload fails, the next retry is 10 seconds later.
      .WillOnce(Return(start_test_time + absl::Seconds(10)));

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");

  // readTelemetry is only called once. 1st time is at the beginning.
  // Then the uploader will fail to upload, the queue is not cleared.
  // So, the readTelemetry() will be blocked from being called.
  EXPECT_CALL(*TelemetryProvider, readTelemetry)
      .Times(1)
      .WillOnce(Return(GenerateFakeProtobuf()));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
}

TEST_F(NCCLStatsReporterTest, TestInitFailureWhenEnvVarIndicatesOff) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "0", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead()).Times(0);
  EXPECT_CALL(*uploader, Upload(_, _)).Times(0);
  EXPECT_CALL(*uploader, NextUploadTime()).Times(0);

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");

  EXPECT_CALL(*TelemetryProvider, readTelemetry).Times(0);
  absl::Status init_status = reporter->init();
  EXPECT_EQ(init_status.code(), absl::StatusCode::kFailedPrecondition);
  absl::SleepFor(absl::Seconds(0.2));
}

TEST_F(NCCLStatsReporterTest, TestWriteOnLocalDiskOnly) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "1", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead()).Times(0);
  EXPECT_CALL(*uploader, Upload(_, _)).Times(0);
  EXPECT_CALL(*uploader, NextUploadTime()).Times(0);

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");

  EXPECT_CALL(*TelemetryProvider, readTelemetry).Times(AtLeast(1));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
}

TEST_F(NCCLStatsReporterTest, TestUploadOnly) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "3", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "1000000", 2);
  setenv("NCCL_GPUVIZ_OUTPUT_FILES_PATH", "/tmp", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead()).Times(AtLeast(1));
  EXPECT_CALL(*uploader, Upload(_, _)).Times(AtLeast(1));
  absl::Time start_test_time = get_current_monotonic_time();

  EXPECT_CALL(*uploader, NextUploadTime())
      .WillRepeatedly(Return(start_test_time));

  DeleteFilesMatchingRegex("/tmp", "^exporter_.*log$");
  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");

  EXPECT_CALL(*TelemetryProvider, readTelemetry)
      .Times(3)
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
  EXPECT_TRUE(!NonEmptyFilesMatchingRegexExists("/tmp", "^exporter_.*log$"));
}

TEST_F(NCCLStatsReporterTest, TestUploadOnlyWithBogusValue) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "2", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "1000000", 2);
  setenv("NCCL_GPUVIZ_OUTPUT_FILES_PATH", "/tmp", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead()).Times(AtLeast(1));
  EXPECT_CALL(*uploader, Upload(_, _)).Times(AtLeast(1));
  absl::Time start_test_time = get_current_monotonic_time();

  EXPECT_CALL(*uploader, NextUploadTime())
      .WillRepeatedly(Return(start_test_time));

  DeleteFilesMatchingRegex("/tmp", "^exporter_.*log$");
  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");

  EXPECT_CALL(*TelemetryProvider, readTelemetry)
      .Times(3)
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
  EXPECT_TRUE(!NonEmptyFilesMatchingRegexExists("/tmp", "^exporter_.*log$"));
}

TEST_F(NCCLStatsReporterTest, TestWriteAndUploadBothOptIn) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "4", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead()).Times(AtLeast(1));
  EXPECT_CALL(*uploader, Upload(_, _)).Times(AtLeast(1));
  absl::Time start_test_time = get_current_monotonic_time();
  EXPECT_CALL(*uploader, NextUploadTime())
      .Times(AtLeast(1))
      .WillRepeatedly(Return(start_test_time));

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");

  EXPECT_CALL(*TelemetryProvider, readTelemetry)
      .Times(3)
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()))
      .WillOnce(Return(GenerateFakeProtobuf()));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
}

TEST_F(NCCLStatsReporterTest,
       TestWriteAndUploadBothOptInWithEnvVarButUploadOffOverWrittenByMds) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "4", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead()).Times(0);
  EXPECT_CALL(*uploader, Upload(_, _)).Times(0);
  EXPECT_CALL(*uploader, NextUploadTime()).Times(0);

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "off");

  EXPECT_CALL(*TelemetryProvider, readTelemetry).Times(AtLeast(1));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(2.2));
}

TEST_F(NCCLStatsReporterTest, TestUploadOnShutdown) {
  setenv("NCCL_NET_PLUGIN_TELEMETRY_MODE", "3", 2);
  setenv("NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE", "0", 2);
  setenv("NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS", "10000000", 2);
  ncclDebugLogger_t logFunction = dummyDebugLog;

  auto TelemetryProvider = std::make_shared<MockTelemetryProvider>();
  auto uploader = std::make_unique<MockUploader>();

  EXPECT_CALL(*uploader, IsDead())
      .Times(AtLeast(1))
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, Upload(_, _)).Times(1).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*uploader, NextUploadTime())
      .Times(AtLeast(1))
      // uploader should never be called during normal operation.
      // However, the shutdown upload would not be blocked by this.
      .WillRepeatedly(Return(absl::InfiniteFuture()));

  auto reporter = std::make_unique<NcclStatsReporter>(
      TelemetryProvider, logFunction, std::move(uploader), "all");
  // only called once during shutdown.
  EXPECT_CALL(*TelemetryProvider, readTelemetry)
      .WillOnce(Return(GenerateFakeProtobuf()));
  EXPECT_EQ(reporter->init(), absl::OkStatus());
  absl::SleepFor(absl::Seconds(0.2));
}

}  // namespace
}  // namespace gpuviz
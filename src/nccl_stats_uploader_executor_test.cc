// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_uploader_executor.h"

#include <cstdarg>
#include <cstdio>
#include <queue>
#include <vector>

#include "absl/strings/str_split.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "test_utils.h"
#include "uploader_interface.h"
#include "utils.h"

using namespace ::testing;
using namespace ::gpuviz::utils;
using namespace ::protoDist;
using ::testing::Test;

namespace gpuviz {
namespace {
// Stores the addresses of all fields of original AllStats proto.
struct AllStatsSubFieldAddr {
  AllStatsSubFieldAddr(AllStats* msg) {
    for (int i = msg->events_log_size() - 1; i >= 0; --i) {
      log_event_addrs.push_back(msg->mutable_events_log()->Mutable(i));
    }
    for (int i = msg->conn_stats_size() - 1; i >= 0; --i) {
      conn_stats_addrs.push_back(msg->mutable_conn_stats()->Mutable(i));
    }
  }
  std::vector<LogEntry*> log_event_addrs;
  std::vector<ConnectionStats*> conn_stats_addrs;
};

// Tests all sub-fields of AllStats memory location do not change after
// splitAllStatsMessageIntoQ() function splits the original messages into q of
// AllStats.
static void TestQueueOfAllStatsAreExpected(
    const AllStatsSubFieldAddr& msg_addr,
    std::queue<std::unique_ptr<AllStats>>& q) {
  size_t num_log_events = msg_addr.log_event_addrs.size();
  size_t num_conn_stats = msg_addr.conn_stats_addrs.size();
  if (num_conn_stats == 0 && num_log_events == 0) {
    EXPECT_TRUE(q.empty());
    return;
  }
  std::string process_identifier = getProcessIdentifier();
  size_t ind_log_events = 0, ind_conn_stats = 0;
  std::optional<uint64_t> prev_upload_index = std::nullopt;
  while (!q.empty()) {
    std::unique_ptr<AllStats> sub_m = std::move(q.front());
    q.pop();

    // The process identifier now contains a random number and more than
    // one call to getProcessIdentifier() produces two different random
    // numbers.  This isn't a problem in practice, since the function is
    // only called once and the result is then stored in a variable,
    // but it does mean that this test can't do a full string match.
    // So we check the first two parts (PID and process start time)
    std::vector<std::string> our_pid_parts =
        absl::StrSplit(process_identifier, ':');
    std::vector<std::string> sub_pid_parts =
        absl::StrSplit(sub_m->source_process_identifier(), ':');
    EXPECT_EQ(our_pid_parts[0], sub_pid_parts[0]);
    EXPECT_EQ(our_pid_parts[1], sub_pid_parts[1]);

    // upload_index should increment continuously by 1.
    if (prev_upload_index.has_value()) {
      EXPECT_EQ(sub_m->upload_index() - *prev_upload_index, 1);
    }
    prev_upload_index = sub_m->upload_index();

    for (LogEntry& log_entry : *(sub_m->mutable_events_log())) {
      if (log_entry.category_name() == "Uploader_Start") {
        // Uploader_start is added as part of the UpdateQueue, skipping.
        continue;
      }
      ASSERT_GT(num_log_events, ind_log_events);
      EXPECT_EQ(&log_entry, msg_addr.log_event_addrs[ind_log_events++]);
    }
    for (ConnectionStats& conn_stats : *(sub_m->mutable_conn_stats())) {
      ASSERT_GT(num_conn_stats, ind_conn_stats);
      EXPECT_EQ(&conn_stats, msg_addr.conn_stats_addrs[ind_conn_stats++]);
    }
  }
  EXPECT_EQ(num_log_events, ind_log_events);
  EXPECT_EQ(num_conn_stats, ind_conn_stats);
}

// Calculates expected number of AllStats after the split that contain pure
// LogEvent and pure ConnectionStats.
static int CalculateExpectedChunkSize(AllStats* msg,
                                      const size_t msg_size_quota) {
  if (msg->ByteSizeLong() <= msg_size_quota &&
      (msg->conn_stats_size() > 0 || msg->events_log_size() > 0)) {
    return 1;
  }
  int num_chunks = 0;
  size_t current_cumullative_size = 0;
  for (const LogEntry& log : msg->events_log()) {
    if (current_cumullative_size + log.ByteSizeLong() > msg_size_quota) {
      if (current_cumullative_size != 0) {
        num_chunks++;
        current_cumullative_size = 0;
      }
    }
    current_cumullative_size += log.ByteSizeLong();
  }
  for (const ConnectionStats& conn_stats : msg->conn_stats()) {
    if (current_cumullative_size + conn_stats.ByteSizeLong() > msg_size_quota) {
      if (current_cumullative_size != 0) {
        num_chunks++;
        current_cumullative_size = 0;
      }
    }
    current_cumullative_size += conn_stats.ByteSizeLong();
  }
  if (current_cumullative_size != 0) {
    num_chunks++;
    current_cumullative_size = 0;
  }
  return num_chunks;
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

class NcclStatsUploaderExecutorWrapper : public NcclStatsUploaderExecutor {
 public:
  NcclStatsUploaderExecutorWrapper(std::unique_ptr<UploaderInterface> uploader,
                                   ncclDebugLogger_t logFunction)
      : NcclStatsUploaderExecutor(std::move(uploader), logFunction) {}
  void TestQueueOfAllStatsIsSplitRight(const AllStatsSubFieldAddr& msg_addr,
                                       const int expected_chunk_size) {
    test_q([&](std::queue<std::unique_ptr<AllStats>>& q) {
      EXPECT_EQ(expected_chunk_size, q.size());
      TestQueueOfAllStatsAreExpected(msg_addr, q);
    });
  }
};

class NcclStatsUploaderExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST(NcclStatsUploaderExecutorTest, NoSplitWithBigChunkSize) {
  // Generate a tiny AllStats, and attempt to split with a gigantic chunk
  // size.
  auto msg = std::make_unique<AllStats>(GenerateAllStats(
      {DistType::protoBandwidthVariance, DistType::protoLatencyVariance}, 1, 1,
      1));
  AllStatsSubFieldAddr addrs(msg.get());
  size_t msg_size_quota = static_cast<size_t>(1E10);
  ASSERT_GT(msg_size_quota, msg->ByteSizeLong());
  int expected_chunk_size =
      CalculateExpectedChunkSize(msg.get(), msg_size_quota);

  auto uploader = std::make_unique<MockUploader>();
  // turn off uploader to test split functionality. We keep uploader alive, but
  // let the next upload time to be infinity future, so that no clean up of the
  // queue will be executed before we test the queue content.
  EXPECT_CALL(*uploader, IsDead()).WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, NextUploadTime())
      .WillRepeatedly(Return(absl::InfiniteFuture()));
  EXPECT_CALL(*uploader, GetMaxMessageByteSize())
      .WillOnce(Return(msg_size_quota));

  auto executor_test = std::make_unique<NcclStatsUploaderExecutorWrapper>(
      std::move(uploader), dummyDebugLog);
  EXPECT_TRUE(executor_test->UpdateQueue(std::move(msg)).ok());
  executor_test->TestQueueOfAllStatsIsSplitRight(addrs, expected_chunk_size);
}

TEST(NcclStatsUploaderExecutorTest, SplitWithSmallChunkSize) {
  // Generate a normal size AllStats message, with 10 LogEntry message, 10
  // ConnectionStats, each of which has 2 histogram, each histogram has 100
  // buckets. For a single LogEntry, the memory size should be less than 100
  // bytes, the wire format size should be even smaller. For a single
  // connectionstats, the size should be dominated by the bucket, which is 8
  // bytes per int64, so, each connection stats should be around 2000 bytes.
  // We choose 150 bytes as the chunk size, so that it is larger than 1
  // LogEntry, and smaller than a connection stats.
  auto msg = std::make_unique<AllStats>(GenerateAllStats(
      {DistType::protoBandwidthVariance, DistType::protoLatencyVariance}, 10,
      10, 100));
  AllStatsSubFieldAddr addrs(msg.get());
  size_t msg_size_quota = static_cast<size_t>(150);
  int expected_chunk_size =
      CalculateExpectedChunkSize(msg.get(), msg_size_quota);

  auto uploader = std::make_unique<MockUploader>();
  // turn off uploader to test split functionality. We keep uploader alive, but
  // let the next upload time to be infinity future, so that no clean up of the
  // queue will be executed before we test the queue content.
  EXPECT_CALL(*uploader, IsDead()).WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, NextUploadTime())
      .WillRepeatedly(Return(absl::InfiniteFuture()));
  EXPECT_CALL(*uploader, GetMaxMessageByteSize())
      .WillOnce(Return(msg_size_quota));

  auto executor_test = std::make_unique<NcclStatsUploaderExecutorWrapper>(
      std::move(uploader), dummyDebugLog);
  EXPECT_TRUE(executor_test->UpdateQueue(std::move(msg)).ok());
  executor_test->TestQueueOfAllStatsIsSplitRight(addrs, expected_chunk_size);
}

TEST(NcclStatsUploaderExecutorTest, SplitWithTypicalChunkSize) {
  // Generate a normal size AllStats message, with 10 LogEntry message, 100
  // ConnectionStats, each of which has 8 histograms, each histogram has 200
  // buckets. For a single connectionstats, the size should be dominated by
  // the bucket, which is 8 bytes per int64, so, each connection stats should
  // be around 200 * 8 * 8 = 10800 bytes. We choose 20000 bytes as the chunk
  // size, which is larger than a connection stats.

  std::vector<DistType> types(8, DistType::protoBandwidthVariance);
  auto msg = std::make_unique<AllStats>(GenerateAllStats(types, 100, 10, 200));
  AllStatsSubFieldAddr addrs(msg.get());
  size_t msg_size_quota = static_cast<size_t>(20000);
  int expected_chunk_size =
      CalculateExpectedChunkSize(msg.get(), msg_size_quota);

  auto uploader = std::make_unique<MockUploader>();
  // turn off uploader to test split functionality. We keep uploader alive, but
  // let the next upload time to be infinity future, so that no clean up of the
  // queue will be executed before we test the queue content.
  EXPECT_CALL(*uploader, IsDead()).WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, NextUploadTime())
      .WillRepeatedly(Return(absl::InfiniteFuture()));
  EXPECT_CALL(*uploader, GetMaxMessageByteSize())
      .WillOnce(Return(msg_size_quota));

  auto executor_test = std::make_unique<NcclStatsUploaderExecutorWrapper>(
      std::move(uploader), dummyDebugLog);
  ABSL_LOG(INFO) << "byte size is " << msg->ByteSizeLong();
  absl::Time current_time = get_current_monotonic_time();
  EXPECT_TRUE(executor_test->UpdateQueue(std::move(msg)).ok());
  absl::Duration duration = get_current_monotonic_time() - current_time;
  ABSL_LOG(INFO) << "It takes this duration to split the queue: " << duration;
  executor_test->TestQueueOfAllStatsIsSplitRight(addrs, expected_chunk_size);
}

TEST(NcclStatsUploaderExecutorTest, SplitWithLargeConnStatsOnly) {
  // Generate a normal size AllStats message, with 0 LogEntry, 1000
  // ConnectionStats, each of which has 8 histograms, each histogram has 200
  // buckets. For a single connectionstats, the size should be dominated by
  // the bucket, which is 8 bytes per int64, so, each connection stats should
  // be around 10800 bytes. We choose 1E7 bytes as the chunk size, which is
  // larger than a connection stats.
  std::vector<DistType> types(8, DistType::protoBandwidthVariance);
  auto msg = std::make_unique<AllStats>(GenerateAllStats(types, 1000, 0, 200));
  AllStatsSubFieldAddr addrs(msg.get());
  size_t msg_size_quota = static_cast<size_t>(1E7);
  int expected_chunk_size =
      CalculateExpectedChunkSize(msg.get(), msg_size_quota);

  auto uploader = std::make_unique<MockUploader>();
  // turn off uploader to test split functionality. We keep uploader alive, but
  // let the next upload time to be infinity future, so that no clean up of the
  // queue will be executed before we test the queue content.
  EXPECT_CALL(*uploader, IsDead()).WillRepeatedly(Return(false));
  EXPECT_CALL(*uploader, NextUploadTime())
      .WillRepeatedly(Return(absl::InfiniteFuture()));
  EXPECT_CALL(*uploader, GetMaxMessageByteSize())
      .WillOnce(Return(msg_size_quota));

  auto executor_test = std::make_unique<NcclStatsUploaderExecutorWrapper>(
      std::move(uploader), dummyDebugLog);
  ABSL_LOG(INFO) << "byte size is " << msg->ByteSizeLong();
  absl::Time current_time = get_current_monotonic_time();
  EXPECT_TRUE(executor_test->UpdateQueue(std::move(msg)).ok());
  absl::Duration duration = get_current_monotonic_time() - current_time;
  ABSL_LOG(INFO) << "It takes this duration to split the queue: " << duration;
  executor_test->TestQueueOfAllStatsIsSplitRight(addrs, expected_chunk_size);
}

TEST(NcclStatsUploaderExecutorTest, UpdateQueueFailsWhenNoUploader) {
  std::unique_ptr<UploaderInterface> null_uploader;
  NcclStatsUploaderExecutor executor(std::move(null_uploader), dummyDebugLog);
  std::vector<DistType> types(8, DistType::protoBandwidthVariance);
  auto msg = std::make_unique<AllStats>(GenerateAllStats(types, 1000, 0, 200));
  absl::Status result = executor.UpdateQueue(std::move(msg));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(result.message(), "No uploader exists.");
}

}  // namespace
}  // namespace gpuviz

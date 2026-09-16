/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_STATS_UPLOADER_EXECUTOR_H_
#define NCCL_STATS_UPLOADER_EXECUTOR_H_

#include <queue>
#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "google/protobuf/message.h"
#include "include/nccl_net.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "uploader_interface.h"

namespace gpuviz {

// This class is designed to execute the uploader in NcclStatsReporter.
class NcclStatsUploaderExecutor {
 public:
  NcclStatsUploaderExecutor(std::unique_ptr<UploaderInterface> uploader,
                            ncclDebugLogger_t logFunction);

  // Returns if the executor is shutdown or finishes uploading.
  bool UploadFinished() ABSL_LOCKS_EXCLUDED(mtx_);

  // Uploads the data passed in by UpdateQueue().
  // This method should be called in a dedicated thread.
  void RunUploadExecutor() ABSL_LOCKS_EXCLUDED(mtx_);

  // Produces the data passed in by caller.
  // When the passed in data m, will be split and overwrite the buffer, q_,
  // stored in this class. In the future, we will consider more sophisticated
  // algorithm, like merging queue, etc.
  // Returns success if the queue has been updated and failure if the queue
  // still had items in it.
  absl::Status UpdateQueue(std::unique_ptr<protoDist::AllStats> m)
      ABSL_LOCKS_EXCLUDED(mtx_);

  ~NcclStatsUploaderExecutor();

 protected:
  // Unit test use only.
  void test_q(absl::AnyInvocable<
              void(std::queue<std::unique_ptr<protoDist::AllStats>>&)>
                  test_function) ABSL_LOCKS_EXCLUDED(mtx_) {
    absl::MutexLock lock(&mtx_);
    test_function(q_);
  }

 private:
  using Msg = std::unique_ptr<protoDist::AllStats>;
  using MsgQ = std::queue<std::unique_ptr<protoDist::AllStats>>;

  std::thread uploader_thread_;

  // Splits the msg of AllStats into chunks and store in a queue, with
  // each chunk having wire format size <= bytes_per_message. The current
  // smallest unit of split is single LogEntry field and single
  // ConnectionStats field. If any single field of LogEntry or
  // ConnectionStats has a byte size greater than bytes_per_message, which
  // is possible in theory but not likely in reality, we should further
  // split the LogEntry and single ConnectionStats.
  // In addition to the split, we also included source_process_identifier and
  // upload_index in each msg.
  std::queue<std::unique_ptr<protoDist::AllStats>> SplitAllStatsMessageIntoQ(
      std::unique_ptr<protoDist::AllStats> msg, const size_t bytes_per_message);

  void Shutdown() ABSL_LOCKS_EXCLUDED(mtx_);

  // Determines whether the RunUploadExecutor() should wake up to upload the
  // data. Returns true if the q_ becomes non-empty or we want to shutdown.
  bool ShouldWakeUp() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mtx_);

  // Waits until ShouldWakeUp() returns true.
  // Returns whether user determines to shutdown.
  bool WaitForWorkOrShutdown() ABSL_LOCKS_EXCLUDED(mtx_);

  // Sleeps until next_execute_time. If we want to shutdown, the sleep will
  // be interrupted and return true. Otherwise, return false when we reach
  // next_execute_time.
  bool WaitForTimeoutOrShutdown(const absl::Time next_execute_time)
      ABSL_LOCKS_EXCLUDED(mtx_);

  // Upload the remaining q_ when shutdown happens.
  // This function will be executed when RunUploadExecutor() returns.
  void UploadAndCleanupRemainingTelemetryOnShutdown() ABSL_LOCKS_EXCLUDED(mtx_);

  protoDist::LogEntry CreateShutdownMessage();
  protoDist::LogEntry CreateUploadStartMessage();
  void CleanUpQ() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mtx_);

  std::unique_ptr<UploaderInterface> uploader_;
  ncclDebugLogger_t log_function_;

  absl::Mutex mtx_;

  bool last_upload_success_ ABSL_GUARDED_BY(mtx_) = true;

  // buffer to store the messages that are ready to upload.
  MsgQ q_ ABSL_GUARDED_BY(mtx_);

  // flag to indicate whether the caller decides to shutdown.
  bool shutdown_ ABSL_GUARDED_BY(mtx_) = false;

  // The process identifier to include with our messages
  const std::string source_process_identifier_;

  // Index to include in each messages being uploaded.
  uint64_t index_ = 0;
};

}  // namespace gpuviz

#endif  // NCCL_STATS_UPLOADER_EXECUTOR_H_

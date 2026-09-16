/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_STATS_REPORTER_H_
#define NCCL_STATS_REPORTER_H_
#include <atomic>
#include <ctime>
#include <fstream>
#include <queue>
#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "nccl_stats_aggregator.h"
#include "nccl_stats_uploader_executor.h"
#include "params.h"
#include "src/proto/nccl_telemetry_proto.pb.h"

namespace gpuviz {

class NcclStatsReporter {
 public:
  NcclStatsReporter(std::shared_ptr<TelemetryProvider> provider,
                    ncclDebugLogger_t logFunction,
                    std::unique_ptr<UploaderInterface> uploader = nullptr,
                    std::string uploader_opt_in = "",
                    gpuviz::MillisecondBandwidthOutput* msec_bw = nullptr)
      : exit_thread_(false),
        telemetry_provider_(provider),
        log_function_(logFunction),
        uploader_(std::move(uploader)),
        // Test only input parameter.
        uploader_opt_in_override_(std::move(uploader_opt_in)),
        msec_bw_(msec_bw) {}
  ~NcclStatsReporter();
  // This method should only be called by reporter_thread_ while exiting, or
  // from NcclStats Singleton when all the connections are deleted.
  void flushTelemetry();
  absl::Status init();
  std::string getFileName();

 private:
  absl::Mutex mtx_;
  absl::Notification exit_thread_notify_;
  std::atomic<bool> exit_thread_;
  std::thread reporter_thread_;
  std::shared_ptr<TelemetryProvider> telemetry_provider_;
  std::ofstream out_;
  ncclDebugLogger_t log_function_;
  struct callback_struct {
    absl::Time time_to_invoke;
    std::function<void(void)> callback;
    absl::Duration invoke_interval;
  };
  std::list<callback_struct> time_to_callback_list_;
  absl::Time log_file_time_;
  uint32_t log_file_num_;
  std::string file_name_;

  std::unique_ptr<UploaderInterface> uploader_;
  std::unique_ptr<NcclStatsUploaderExecutor> upload_executor_ = nullptr;

  absl::Status clean();
  void registerCallbackStruct(absl::Time current_time,
                              absl::Duration time_interval,
                              std::function<void(void)> callback);
  void reportStatsDistribution();
  absl::Duration invokeCallbacksAndReturnDurationtoNextCallback();

  // Determines whether to proceed to call
  // telemetry_provider_->readTelemetry() in readAndReportTelemetry().
  bool ShouldReadTelemetry() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mtx_);

  struct ReportMode {
    bool write_local_disk = false;
    std::unordered_set<protoDist::DistType> dist_types_to_upload;
  } report_mode_;
  std::string uploader_opt_in_override_;
  absl::Status DetermineWriteOnDiskBehavior();
  void DetermineUploadBehavior();
  absl::Status DetermineReportMode();

  void readAndReportTelemetry();
  void processHighFrequencyTelemetry();
  void outputMillisecondBandwidthStatsToFile();
  absl::Status rotateLogs();

  gpuviz::MillisecondBandwidthOutput* msec_bw_;
};

}  // namespace gpuviz

#endif  // NCCL_STATS_REPORTER_H_
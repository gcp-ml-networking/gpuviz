// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_reporter.h"

#include <unistd.h>

#include <ctime>
#include <queue>

#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "google/protobuf/message.h"
#include "nccl_stats.h"
#include "nccl_stats_uploader_executor.h"
#include "params.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "utils.h"

namespace gpuviz {

using Msg = std::unique_ptr<google::protobuf::Message>;
using MsgQ = std::queue<std::unique_ptr<google::protobuf::Message>>;

NcclStatsReporter::~NcclStatsReporter() {
  if (!clean().ok()) {
    std::string log_msg =
        "Failed to clean the reporter while destructing NcclStatsReporter.";
    telemetry_provider_->addLogMsg(log_msg, NCCL_LOG_WARN, __PRETTY_FUNCTION__,
                                   __LINE__);
  }
}

absl::Status NcclStatsReporter::init() {
  absl::Status report_mode_init_status = DetermineReportMode();
  if (!report_mode_init_status.ok()) {
    return report_mode_init_status;
  }
  if (!report_mode_.dist_types_to_upload.empty()) {
    upload_executor_ = std::make_unique<NcclStatsUploaderExecutor>(
        std::move(uploader_), log_function_);
  } else {
    upload_executor_ = nullptr;
  }
  absl::Status create_file_status = rotateLogs();
  if (!create_file_status.ok()) {
    return create_file_status;
  }
  absl::Time current_time = gpuviz::utils::get_current_monotonic_time();

  /* Add the callback function to the list in the same order in which it is
   * expected to be invoked */
  if (gpuviz::params::IsBandwidthHistogramCollectionEnabled()) {
    int64_t bw_burst_bucket_count =
        gpuviz::params::GetBandwidthBurstBucketCount();
    int64_t bw_bucket_time_in_milliseconds =
        gpuviz::params::GetBandwidthBucketTimeInMilliSeconds();
    int64_t bw_process_interval_in_milliseconds =
        bw_burst_bucket_count * bw_bucket_time_in_milliseconds;
    auto callback_function1 =
        std::bind(&NcclStatsReporter::processHighFrequencyTelemetry, this);
    registerCallbackStruct(
        current_time, absl::Milliseconds(bw_process_interval_in_milliseconds),
        callback_function1);
  }

  if (gpuviz::params::IsMillisecondBandwidthOutputEnabled()) {
    int64_t output_time_in_milliseconds =
        gpuviz::params::GetMillisecondBandwidthOutputIntervalInMilliSeconds();
    auto callback_function3 = absl::bind_front(
        &gpuviz::MillisecondBandwidthOutput::SaveBandwidthStats, msec_bw_,
        false);
    registerCallbackStruct(current_time,
                           absl::Milliseconds(output_time_in_milliseconds),
                           callback_function3);
  }

  int64_t read_interval_in_microseconds =
      gpuviz::params::GetReadIntervalInMicroSeconds();
  auto callback_function2 =
      std::bind(&NcclStatsReporter::readAndReportTelemetry, this);
  registerCallbackStruct(current_time,
                         absl::Microseconds(read_interval_in_microseconds),
                         callback_function2);
  std::thread thread(&NcclStatsReporter::reportStatsDistribution, this);
  reporter_thread_ = std::move(thread);

  return absl::OkStatus();
}

std::string NcclStatsReporter::getFileName() {
  if (!out_.is_open()) {
    // Instantiate file name with time we start collecting telemetry.
    absl::Time log_file_time_ = absl::Now();
    std::string time_str = absl::FormatTime("%Y-%m-%d_%H:%M:%S", log_file_time_,
                                            absl::UTCTimeZone());

    std::string directory_path =
        gpuviz::params::GetDirectoryPathForGPUVizFiles();
    file_name_ = directory_path + "/exporter_" + std::to_string(getpid()) +
                 "_" + time_str;

    log_file_num_ = 1;
    return file_name_ + ".log";
  }
  // All log files have the same name plus a number to indicate how many have
  // been rotated. This is to keep track of files from a single run.
  std::string new_file_name = file_name_;
  new_file_name.append("_" + std::to_string(log_file_num_));

  return new_file_name + ".log";
}

absl::Status NcclStatsReporter::rotateLogs() {
  if (!report_mode_.write_local_disk) {
    return absl::OkStatus();
  }
  std::ofstream rot_in;
  std::string result = getFileName();

  rot_in.open(result, std::ios_base::app);
  if (!rot_in) {
    std::string log_msg = "Failed to open or rotate file with error " +
                          std::string(std::strerror(errno));
    telemetry_provider_->addLogMsg(log_msg, NCCL_LOG_WARN, __PRETTY_FUNCTION__,
                                   __LINE__);
    return absl::InternalError("Failed to open file.");
  }
  log_file_num_ += 1;
  out_.close();
  out_.swap(rot_in);

  return absl::OkStatus();
}

absl::Status NcclStatsReporter::clean() {
  exit_thread_.store(true, std::memory_order_release);
  if (reporter_thread_.joinable()) {
    exit_thread_notify_.Notify();
    reporter_thread_.join();
  }
  if (upload_executor_ != nullptr) {
    upload_executor_ = nullptr;
  }
  out_.close();
  return absl::OkStatus();
}

void NcclStatsReporter::registerCallbackStruct(
    absl::Time current_time, absl::Duration time_interval,
    std::function<void(void)> callback) {
  callback_struct cb_struct;
  cb_struct.time_to_invoke = current_time + time_interval,
  cb_struct.callback = callback, cb_struct.invoke_interval = time_interval;
  time_to_callback_list_.push_back(cb_struct);
}

void NcclStatsReporter::reportStatsDistribution() {
  absl::Duration read_interval_duration = absl::ZeroDuration();
  while (!exit_thread_.load(std::memory_order_acquire)) {
    exit_thread_notify_.WaitForNotificationWithTimeout(read_interval_duration);
    read_interval_duration = invokeCallbacksAndReturnDurationtoNextCallback();
  }
  flushTelemetry();
}

bool NcclStatsReporter::ShouldReadTelemetry() {
  // If customer opt out upload, this function should always return true
  // because upload_executor == nullptr.
  return upload_executor_ == nullptr || upload_executor_->UploadFinished();
}

void NcclStatsReporter::readAndReportTelemetry() {
  absl::MutexLock lock(&mtx_);
  if (!ShouldReadTelemetry()) {
    return;
  }
  absl::StatusOr<std::unique_ptr<protoDist::AllStats>> telemetry =
      telemetry_provider_->readTelemetry();

  if (telemetry.ok()) {
    if (report_mode_.write_local_disk) {
      google::protobuf::Message& stats = *telemetry.value();
      const std::string delimiter = "==END==";

      absl::Duration file_rot_interval =
          absl::Seconds(gpuviz::params::GetFileRotationIntervalInSeconds());
      if (file_rot_interval > absl::Seconds(0) &&
          (absl::Now() - log_file_time_) > file_rot_interval) {
        log_file_time_ = absl::Now();
        absl::Status create_file_status = rotateLogs();

        if (!create_file_status.ok()) {
          telemetry_provider_->addLogMsg(create_file_status.ToString(),
                                         NCCL_LOG_WARN, __PRETTY_FUNCTION__,
                                         __LINE__);
        }
      }

      stats.SerializeToOstream(&out_);
      out_ << delimiter;
      out_.flush();
    }

    if (upload_executor_ != nullptr) {
      // uploader_executor_ not null means upload is turned on.

      // See https://abseil.io/tips/181 as a reference for the recommended way
      // of moving content held by statusor: *std::move(telemetry).
      std::unique_ptr<protoDist::AllStats> msg = *std::move(telemetry);
      gpuviz::utils::filterHistogramsByDistType(
          msg, report_mode_.dist_types_to_upload);
      absl::Status update_status =
          upload_executor_->UpdateQueue(std::move(msg));
      if (!update_status.ok()) {
        telemetry_provider_->addLogMsg(update_status.message(), NCCL_LOG_WARN,
                                       __PRETTY_FUNCTION__, __LINE__);
      }
    }
  } else {
    telemetry_provider_->addLogMsg(telemetry.status().ToString(), NCCL_LOG_WARN,
                                   __PRETTY_FUNCTION__, __LINE__);
  }
}

void NcclStatsReporter::processHighFrequencyTelemetry() {
  absl::MutexLock lock(&mtx_);
  absl::Status telemetry = telemetry_provider_->processHighFrequencyTelemetry();
}

absl::Duration
NcclStatsReporter::invokeCallbacksAndReturnDurationtoNextCallback() {
  std::vector<std::function<void(void)>> local_callback_list;
  absl::Time min_so_far = absl::InfiniteFuture();
  // Functions will get called in the same order in which they were added to
  // time_to_callback_list_.
  for (callback_struct& it : time_to_callback_list_) {
    absl::Time current_time = gpuviz::utils::get_current_monotonic_time();
    if (it.time_to_invoke <= current_time) {
      (it.callback)();
      // This will ensure that next invoke time for this function is after
      // invoke_interval from the current time, thus everytime maintaining
      // similar invoke_interval from the last invoke call of the function.
      it.time_to_invoke = current_time + it.invoke_interval;
    }
    min_so_far = std::min(min_so_far, it.time_to_invoke);
  }
  absl::Time time_now = gpuviz::utils::get_current_monotonic_time();
  absl::Duration duration_to_next_call =
      (min_so_far < time_now) ? absl::ZeroDuration() : (min_so_far - time_now);
  return duration_to_next_call;
}

void NcclStatsReporter::flushTelemetry() {
  if (gpuviz::params::IsBandwidthHistogramCollectionEnabled()) {
    processHighFrequencyTelemetry();
  }
  readAndReportTelemetry();
  if (upload_executor_ != nullptr) {
    upload_executor_ = nullptr;
  }
}

absl::Status NcclStatsReporter::DetermineWriteOnDiskBehavior() {
  ncclTelemetryMode telemetry_mode = gpuviz::params::GetPluginTelemetryMode();
  if (telemetry_mode == ncclTelemetryMode::kOff) {
    std::string log_msg =
        "Telemetry client is set with off, "
        "Telemetry client initialization intentionally returns a failed init "
        "status and "
        "not collecting data.";
    telemetry_provider_->addLogMsg(log_msg, NCCL_LOG_WARN, __PRETTY_FUNCTION__,
                                   __LINE__);
    return absl::FailedPreconditionError(
        "Customer wants to turn off NCCL telemetry data collection "
        "completely.");
  }
  if (telemetry_mode == ncclTelemetryMode::kReserved ||
      telemetry_mode == ncclTelemetryMode::kUndefined) {
    // all of these values are impossible to show in GPUViz. We now
    // default to be kUploadOnlyControlledByMds.
    std::string log_msg =
        "NCCL_NET_PLUGIN_TELEMETRY_MODE is set with bogus value "
        "or not set at all, "
        "we default the bogus value to turn off write on disk "
        "and upload the telemetry controlled by MDS.";
    telemetry_provider_->addLogMsg(log_msg, NCCL_LOG_WARN, __PRETTY_FUNCTION__,
                                   __LINE__);
    telemetry_mode = ncclTelemetryMode::kUploadOnlyControlledByMds;
  }
  // We turn off write on local disk iff telemetry_mode is
  // kUploadOnlyControlledByMds
  report_mode_.write_local_disk =
      (telemetry_mode != ncclTelemetryMode::kUploadOnlyControlledByMds);
  return absl::OkStatus();
}

void NcclStatsReporter::DetermineUploadBehavior() {
  report_mode_.dist_types_to_upload.clear();

  // Telemetry Mode env var takes precedence over individual Upload env vars if
  // telemetry mode indicates users want to shutdown upload to protect their
  // privacy. So, if user wants write on disk only, we will ignore the Upload
  // env vars.
  ncclTelemetryMode telemetry_mode = gpuviz::params::GetPluginTelemetryMode();
  if (telemetry_mode == ncclTelemetryMode::kWriteOnLocalDiskOnly) {
    return;
  }
  if (uploader_opt_in_override_ == "off") {
    // Only Unit test should execute this block.
    return;
  }

  if (params::isSendLatencyUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoSendLatencySW);
  }
  if (params::isRecvLatencyUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoRecvLatencySW);
  }
  if (params::isSendMessageSizeUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoSendMessageSize);
  }
  if (params::isRecvMessageSizeUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoRecvMessageSize);
  }
  if (params::isSendBandwidthUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoSendBandwidth);
  }
  if (params::isRecvBandwidthUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoRecvBandwidth);
  }
  if (params::isBandwidthVarianceUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoBandwidthVariance);
  }
  if (params::isLatencyVarianceUploadEnabled()) {
    report_mode_.dist_types_to_upload.insert(
        protoDist::DistType::protoLatencyVariance);
  }
}

absl::Status NcclStatsReporter::DetermineReportMode() {
  absl::Status status = DetermineWriteOnDiskBehavior();
  if (!status.ok()) {
    return status;
  }
  DetermineUploadBehavior();
  return absl::OkStatus();
}

}  // namespace gpuviz

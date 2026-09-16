// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_container.h"

#include <cstring>
#include <limits>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "client.h"
#include "git_tag.h"
#include "nccl_log_sink.h"
#include "nccl_uploader.h"
#include "params.h"
#include "uploader_interface.h"
#include "utils.h"

namespace gpuviz {

class NcclStatsContainerInternal {
 public:
  NcclStatsContainerInternal(ncclDebugLogger_t logFunction,
                             uint64_t distributionCollectorBitmap);
  // Wrapping classes ensure that init will only be called once during the class
  // lifetime, regardless of how many times the external "init" API is called.
  absl::Status init();
  NcclStatsAggregator* get_nccl_stats_aggregator();
  NcclStatsReporter* get_nccl_stats_reporter();
  gpuviz::MillisecondBandwidthOutput* get_millisecond_bandwidth_output();

 private:
  gpuviz::MillisecondBandwidthOutput msec_bw_;
  std::shared_ptr<NcclStatsAggregator> nccl_stats_aggregator_;
  std::unique_ptr<NcclLogSink> nccl_log_sink_;
  std::unique_ptr<NcclStatsReporter> nccl_stats_reporter_;
};

NcclStatsContainerInternal::NcclStatsContainerInternal(
    ncclDebugLogger_t logFunction, uint64_t distributionCollectorBitmap)
    : nccl_stats_aggregator_(std::make_shared<NcclStatsAggregator>(
          logFunction, distributionCollectorBitmap, &msec_bw_)),
      nccl_log_sink_(
          std::make_unique<NcclLogSink>(nccl_stats_aggregator_, logFunction)) {
  auto uploader = std::make_unique<NcclUploader>(
      logFunction, AcsAgentClientWrapper::Create, /* Client creator. */
      nullptr, /* backoff calculator, in prod, we use default*/
      // backoff parameter for send message. base_delay and cap in milliseconds,
      // base is unitless.
      NcclUploader::BackOffParameters{
          .base_delay = 500,
          .base = 2,
          .cap = 600'000 /* 10 minutes*/,
          .max_attempts = std::numeric_limits<int64_t>::max()},
      // backoff parameter for create stream.
      NcclUploader::BackOffParameters{
          .base_delay = 500,
          .base = 2,
          .cap = 600'000 /* 10 minutes*/,
          .max_attempts = std::numeric_limits<int64_t>::max()},
      // delay of create client override. This should be used in unit test only.
      // So, set it as nullopt.
      std::nullopt);
  nccl_stats_reporter_ = std::make_unique<NcclStatsReporter>(
      nccl_stats_aggregator_, logFunction, std::move(uploader), "", &msec_bw_);
}

absl::Status NcclStatsContainerInternal::init() {
  auto aggregator_init_status = nccl_stats_aggregator_->init();
  if (!aggregator_init_status.ok()) {
    return aggregator_init_status;
  }
  auto reporter_init_status = nccl_stats_reporter_->init();
  return reporter_init_status;
}

NcclStatsAggregator* NcclStatsContainerInternal::get_nccl_stats_aggregator() {
  return nccl_stats_aggregator_.get();
}

NcclStatsReporter* NcclStatsContainerInternal::get_nccl_stats_reporter() {
  return nccl_stats_reporter_.get();
}

gpuviz::MillisecondBandwidthOutput*
NcclStatsContainerInternal::get_millisecond_bandwidth_output() {
  return &msec_bw_;
}

/////////////////////////////////////////////////////////////////////////////

class NcclStatsContainerSingleton {
 public:
  NcclStatsContainerSingleton(bool callInitializeLog) {
    if (callInitializeLog) absl::InitializeLog();
    absl::SetGlobalVLogLevel(gpuviz::params::GetAbslLogVerbosity());
  }
  absl::StatusOr<NcclStatsContainerInternal*> getInstance(
      ncclDebugLogger_t logFunction, uint64_t distributionCollectorBitmap);
  void dropInstance();
  // should be called only for debug purposes
  NcclStatsContainerInternal* peekInstance();

 private:
  std::unique_ptr<NcclStatsContainerInternal> nccl_stats_container_internal_;
  absl::Mutex mx_;
  uint64_t ref_cnt_ = 0;
  ncclDebugLogger_t log_function_;
  uint64_t distribution_collector_bitmap_;
};

absl::StatusOr<NcclStatsContainerInternal*>
NcclStatsContainerSingleton::getInstance(ncclDebugLogger_t logFunction,
                                         uint64_t distributionCollectorBitmap) {
  {
    absl::MutexLock lock(&mx_);
    if (ref_cnt_ == 0) {
      nccl_stats_container_internal_ =
          std::make_unique<NcclStatsContainerInternal>(
              logFunction, distributionCollectorBitmap);
      auto nccl_stats_init_status = nccl_stats_container_internal_->init();
      if (!nccl_stats_init_status.ok()) {
        return nccl_stats_init_status;
      }
      distribution_collector_bitmap_ = distributionCollectorBitmap;
    }
    // allow GPUViz to initialization multiple times even if the log function
    // is different. Logging will continue with a new log function
    // (assumption: bitmap must be the same, checked below)
    log_function_ = logFunction;
    if (distribution_collector_bitmap_ != distributionCollectorBitmap) {
      return absl::InvalidArgumentError(
          "NCCL Stats previously \
                                            initialized with different parameters \
                                            for distributionCollectorBitmap.");
    }
    ref_cnt_++;
    return nccl_stats_container_internal_.get();
  }
}

NcclStatsContainerInternal* NcclStatsContainerSingleton::peekInstance() {
  {
    absl::MutexLock lock(&mx_);
    return nccl_stats_container_internal_.get();
  }
}

void NcclStatsContainerSingleton::dropInstance() {
  {
    absl::MutexLock lock(&mx_);
    ref_cnt_--;
    if (ref_cnt_ == 0) {
      nccl_stats_container_internal_.reset();
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

ncclResult_t NcclStatsContainer::init(
    ncclDebugLogger_t logFunction, uint64_t distributionCollectorBitmap,
    ncclTelemetryMode defaultTelemetryMode, const char* callerIdentifier,
    int32_t major, int32_t minor, uintptr_t* statsGlobalHandle /* Out */) {
  ncclResult_t res = init_v2(logFunction, distributionCollectorBitmap,
                             defaultTelemetryMode, callerIdentifier,
                             statsGlobalHandle);
  if (res == ncclSuccess) {
    ((NcclStatsAggregator*)(*statsGlobalHandle))->setPluginVersion(major, minor);
  }
  return res;
}

ncclResult_t NcclStatsContainer::init_v2(ncclDebugLogger_t logFunction,
                                         uint64_t distributionCollectorBitmap,
                                         ncclTelemetryMode defaultTelemetryMode,
                                         const char* callerIdentifier,
                                         uintptr_t* statsGlobalHandle /* Out */
) {
  params::nccl_telemetry_mode_param.default_value = defaultTelemetryMode;
  return init_v1(logFunction, distributionCollectorBitmap, statsGlobalHandle);
}

ncclResult_t NcclStatsContainer::init_v1(
    ncclDebugLogger_t logFunction, uint64_t distributionCollectorBitmap,
    uintptr_t* statsGlobalHandle /* Out */) {
  NcclStatsContainerSingleton& instance =
      getNcclStatsContainerSingletonInstance();
  auto internal_instance_status =
      instance.getInstance(logFunction, distributionCollectorBitmap);

  if (!internal_instance_status.ok()) {
    return ncclInvalidArgument;
  }
  NcclStatsAggregator* aggregator =
      internal_instance_status.value()->get_nccl_stats_aggregator();
  *statsGlobalHandle = (uintptr_t)(aggregator);

  std::string log_msg =
      absl::StrFormat("GPUViz initialized Successfully, version: %s.", GIT_VER);
  aggregator->addLogMsg(log_msg, NCCL_LOG_INFO, __PRETTY_FUNCTION__, __LINE__,
                        NCCL_INIT, "", 0, 0);
  return ncclSuccess;
}

ncclResult_t NcclStatsContainer::destroy(uintptr_t statsGlobalHandle) {
  (void)statsGlobalHandle;  // To suppress unused variable warning.
  // stop inf retry to send undrained data to ACS at shutdown time.
  NcclStatsContainerInternal* container_internal =
      getNcclStatsContainerSingletonInstance().peekInstance();
  if (container_internal != nullptr) {
    NcclStatsReporter* reporter = container_internal->get_nccl_stats_reporter();
    if (reporter != nullptr) {
      reporter->flushTelemetry();
    }
    getNcclStatsContainerSingletonInstance().dropInstance();
  }
  return ncclSuccess;
}

ncclResult_t NcclStatsContainer::addConnection(
    uintptr_t statsGlobalHandle,
    const ncclStatsConnectionIdentifier* connectionIdentifier,
    uintptr_t* statsConnectionHandle) {
  DCHECK(statsGlobalHandle ==
         (uintptr_t)getNcclStatsContainerSingletonInstance()
             .peekInstance()
             ->get_nccl_stats_aggregator());
  auto addConnectionStatus = ((NcclStatsAggregator*)statsGlobalHandle)
                                 ->addConnection(connectionIdentifier);

  if (!addConnectionStatus.ok()) {
    *statsConnectionHandle = 0;
    return ncclInternalError;
  } else {
    *statsConnectionHandle = (uintptr_t)addConnectionStatus.value();
  }
  return ncclSuccess;
}

ncclResult_t NcclStatsContainer::addConnection(
    uintptr_t statsGlobalHandle,
    const ncclStatsConnectionIdentifier_v4* connectionIdentifier,
    uintptr_t* statsConnectionHandle) {
  DCHECK(statsGlobalHandle ==
         (uintptr_t)getNcclStatsContainerSingletonInstance()
             .peekInstance()
             ->get_nccl_stats_aggregator());
  auto addConnectionStatus = ((NcclStatsAggregator*)statsGlobalHandle)
                                 ->addConnectionV4(connectionIdentifier);

  if (!addConnectionStatus.ok()) {
    *statsConnectionHandle = 0;
    return ncclInternalError;
  } else {
    *statsConnectionHandle = (uintptr_t)addConnectionStatus.value();
  }
  return ncclSuccess;
}

ncclResult_t NcclStatsContainer::deleteConnection(
    uintptr_t statsConnectionHandle, ncclStatsConnectionCloseType closeType,
    const char* verboseReason) {
  NcclStatsAggregator* aggregator =
      ((NcclStatsConnectionStatistics*)statsConnectionHandle)->getAggregator();

  DCHECK(aggregator == getNcclStatsContainerSingletonInstance()
                           .peekInstance()
                           ->get_nccl_stats_aggregator());

  if (!aggregator
           ->deleteConnection(
               (NcclStatsConnectionStatistics*)statsConnectionHandle, closeType,
               verboseReason)
           .ok()) {
    return ncclInternalError;
  }

  if (aggregator->getNumberOfConnections() == 0) {
    // Workaround for cleaning things up when destructor does not get called.
    // Needed on A3+.
    NcclStatsContainerInternal* container_internal =
        getNcclStatsContainerSingletonInstance().peekInstance();
    NcclStatsReporter* reporter = container_internal->get_nccl_stats_reporter();
    reporter->flushTelemetry();
    container_internal->get_millisecond_bandwidth_output()
        ->CloseOutBandwidthStats();
  }
  return ncclSuccess;
}

ncclResult_t NcclStatsContainer::notifyOperationMeasurement(
    uintptr_t statsConnectionHandle,
    const ncclStatsOperationMetric* measurement) {
  NcclStatsConnectionStatistics* connectionHandle =
      (NcclStatsConnectionStatistics*)statsConnectionHandle;
  NcclStatsAggregator* aggregator = connectionHandle->getAggregator();

  DCHECK(aggregator == getNcclStatsContainerSingletonInstance()
                           .peekInstance()
                           ->get_nccl_stats_aggregator());

  if (connectionHandle->notifyOperationMeasurement(measurement).ok()) {
    return ncclSuccess;
  }
  return ncclInternalError;
}

ncclResult_t NcclStatsContainer::notifyProfilerEvent(
    uintptr_t statsConnectionHandle, const uint8_t* event_data, size_t len) {
  NcclStatsConnectionStatistics* connectionHandle =
      (NcclStatsConnectionStatistics*)statsConnectionHandle;
  NcclStatsAggregator* aggregator = connectionHandle->getAggregator();

  DCHECK(aggregator == getNcclStatsContainerSingletonInstance()
                           .peekInstance()
                           ->get_nccl_stats_aggregator());

  if (connectionHandle->notifyProfilerEvent(event_data, len).ok()) {
    return ncclSuccess;
  }
  return ncclInternalError;
}

bool NcclStatsContainer::run_initialize_log_ = true;

NcclStatsContainerSingleton&
NcclStatsContainer::getNcclStatsContainerSingletonInstance() {
  static absl::NoDestructor<NcclStatsContainerSingleton>
      nccl_stats_container_singleton(run_initialize_log_);
  return *nccl_stats_container_singleton;  // provides pointer-like access
}

}  // namespace gpuviz

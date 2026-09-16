/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/functional/any_invocable.h"
#include "nccl_stats.h"

namespace gpuviz {
namespace params {

struct Param {
    absl::string_view env_name;
};

template <typename T>
struct NumericParam : public Param {
    T min;
    T max;
    T default_value;
    T wrong_input_value;
    absl::AnyInvocable<T()>* fallback = nullptr;
    T value() const;
};

struct StringParam : public Param {
    absl::string_view default_value;
    absl::AnyInvocable<absl::string_view()>* fallback = nullptr;
    bool trim = false;
    std::string value() const;
};

// Controls telemetry mode.  Off:0, Local Disk:2, Upload:3, Local Disk + Upload:4
extern NumericParam<int32_t> nccl_telemetry_mode_param;

// Controls net plugin telemetry mode.  Off:0, Local Disk:2, Upload:3, Local Disk + Upload:4
constexpr NumericParam<int32_t> kNetPluginTelemetryMode = {
    {
        .env_name = "NCCL_NET_PLUGIN_TELEMETRY_MODE",
    },
    /*min*/ 0,
    /*max*/ 5,  // No idea what mode 5 means here.
    /*default_value*/ 0,
    /*wrong_input_value*/ 5,
};

// Log verbosity level for GPUViz
constexpr NumericParam<int32_t> kAbslLogVerbosity = {
    {
        .env_name = "NCCL_GPUVIZ_ABSL_LOG_VERBOSITY",
    },
    /*min*/ std::numeric_limits<int32_t>::min(),
    /*max*/ std::numeric_limits<int32_t>::max(),
    /*default_value*/ 0,
    /*wrong_input_value*/ 0,
};

// How frequently telemetry histograms are read out to a buffer for
// future saving or uploading
constexpr NumericParam<int64_t> kReadIntervalInMicroseconds = {
    {
        .env_name = "NCCL_GPUVIZ_READ_INTERVAL_IN_MICROSECONDS",
    },
    /*min*/ 1000000,  // 1 second
    /*max*/ __INT_MAX__,
    /*default_value*/ 60000000,  // 60 seconds
    /*wrong_input_value*/ 60000000,  // 60 seconds
};

// GRPC Target string for the ACS Endpoint
constexpr StringParam kAcsEndPointOverride = {
    {
        .env_name = "NCCL_GPUVIZ_ACS_ENDPOINT",
    },
    /*default_value*/ "",
};

// Path to directory where local logs will be saved
constexpr StringParam kOutputFilesPath = {
    {
        .env_name = "NCCL_GPUVIZ_OUTPUT_FILES_PATH",
    },
    /*default_value*/ "/tmp",
    /*fallback*/ nullptr,
    /*trim*/ true,
};

// File rotation interval for primary output
// measured in seconds.  0 indicates disabled
// Millisecond bandwidth output also falls
// back to this if its specific parameters
// are not set.
constexpr NumericParam<int64_t> kFileRotationIntervalInSeconds = {
    {
        .env_name = "NCCL_GPUVIZ_FILE_ROTATION_INTERVAL_IN_SECONDS",
    },
    /*min*/ 0,  // File rotation disabled
    /*max*/ __INT_MAX__,
    /*default_value*/ 0,  // Disabled by default
    /*wrong_input_value*/ 0,  // Disabled on invalid value
};

// How many bandwidth buckets we allocate
// When multiplied by the size of each bucket in milliseconds
// this shows how much time we can handle in between
// calls to process the buckets into histograms
constexpr NumericParam<int64_t> kBwBurstBucketCount = {
    {
        .env_name = "NCCL_GPUVIZ_BW_BURST_BUCKET_COUNT",
    },
    /*min*/ 1,
    /*max*/ 1000,
    /*default_value*/ 100,
    /*wrong_input_value*/ 100,
};

// Number of xtra buckets added to account for jitter
// (by which we mean irregularity in the 
// scheduling of the process which reads
// the bandwidth information and makes
// histograms).
constexpr NumericParam<int64_t> kJitterBwBucketCount = {
    {
        .env_name = "NCCL_GPUVIZ_JITTER_BW_BUCKET_COUNT",
    },
    /*min*/ 0,
    /*max*/ 100,
    /*default_value*/ 10,
    /*wrong_input_value*/ 10,
};

// Enable collecting bandwidth histograms
constexpr NumericParam<int64_t> kBWHistogramCollectionEnable = {
    {
        .env_name = "NCCL_GPUVIZ_BANDWIDTH_HISTOGRAM_COLLECTION_ENABLE",
    },
    /*min*/ 0,
    /*max*/ 1,
    /*default_value*/ 0,
    /*wrong_input_value*/ 0,
};

// Do we also collect bandwidth variance histograms
constexpr NumericParam<int64_t> kVarianceHistCollectionEnable = {
    {
        .env_name = "NCCL_GPUVIZ_VARIANCE_HISTOGRAM_COLLECTION_ENABLE",
    },
    /*min*/ 0,
    /*max*/ 1,
    /*default_value*/ 1,
    /*wrong_input_value*/ 1,
};

ncclTelemetryMode GetPluginTelemetryMode();
int GetAbslLogVerbosity();
bool IsProfilerPluginEnabled();
std::string AcsServerEndPointOverwritten();
bool IsAcsEndPointRegional();
int64_t GetAcsServerAckTimeoutInMilliseconds();
std::string GetDirectoryPathForGPUVizFiles();
int64_t GetReadIntervalInMicroSeconds();
int64_t GetFileRotationIntervalInSeconds();
int64_t GetBandwidthBurstBucketCount();
int64_t GetBandwidthBucketTimeInMilliSeconds();
int64_t GetMaxExpectedLatencyPerTxInMilliSeconds();
int64_t GetJitterBandwidthBucketCount();
std::pair<double, double> GetMaxAndTargetBWPerNICInBitsPerSec(double nic_speed, double empr_nic_speed);
double GetBaselineLatencyInNanoSec();
bool IsBandwidthHistogramCollectionEnabled();
bool IsVarianceDetectionEnabled();
bool isSendLatencyUploadEnabled();
bool isRecvLatencyUploadEnabled();
bool isSendMessageSizeUploadEnabled();
bool isRecvMessageSizeUploadEnabled();
bool isSendBandwidthUploadEnabled();
bool isRecvBandwidthUploadEnabled();
bool isBandwidthVarianceUploadEnabled();
bool isLatencyVarianceUploadEnabled();

struct BucketerParams {
    double max_buckets;
    double scale;
    double base;
};

BucketerParams GetBucketerParamsForLatencyHistogram();
BucketerParams GetBucketerParamsForSizeHistogram();
BucketerParams GetBucketerParamsForBWHistogram(double nic_speed);
BucketerParams GetBucketerParamsForNicBWHistogram(double nic_speed);
BucketerParams GetBucketerParamsForNicBWVarianceHistogram(double nic_speed);
BucketerParams GetBucketerParamsForNicLatencyVarianceHistogram();

// Should we enable saving millisecond bandwidth estimates to a file
bool IsMillisecondBandwidthOutputEnabled();
std::string GetMillisecondBandwidthOutputFilenamePrefix();
// Millisecond bandwidth timing output file will switch to new file after
// this much time has ellapsed.
int64_t GetMillisecondBandwidthOutputRotationRate();
// Number of milliseconds in one batch.
int64_t GetMillisecondBandwidthOutputBatchSize();
// How often to dump the buffer to the output file
int64_t GetMillisecondBandwidthOutputIntervalInMilliSeconds();
// The maximum number of bytes used by accumulated non-aggregated data.
size_t GetNonAggregatedDataSizeCap();

}  // namespace params
}  // namespace gpuviz

#endif // PARAMS_H_

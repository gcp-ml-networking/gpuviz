// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "params.h"

#include <array>
#include <limits>
#include <string.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/status/statusor.h"
#include "nccl_stats.h"

#define GIGA_BITS_TO_BITS 1000000000

constexpr char kMillisecondBandwidthOutputFilePrefix[] = "millisecond_bandwidth";

// SendLatency, SendMsgSize and LatencyVariance histograms are uploaded by default for always-on telemetry.
constexpr bool kSendLatencyHistUploadEnable = true;
constexpr bool kSendMsgSizeHistUploadEnable = true;
constexpr bool kLatencyVarianceHistUploadEnable = true;

// Any other histogram is not uploaded by default so customers have to opt-in.
constexpr bool kRecvLatencyHistUploadEnable = false;
constexpr bool kRecvMsgSizeHistUploadEnable = false;
constexpr bool kSendBWHistUploadEnable = false;
constexpr bool kRecvBWHistUploadEnable = false;
constexpr bool kBWVarianceHistUploadEnable = false;

constexpr int64_t kAcsEndPointZonal = 0;
constexpr int64_t kAcsEndPointRegional = 1;
constexpr int64_t kAcsEndPointZonalDefault = 0;

constexpr int64_t kAcsAckTimeoutMilliseconds = 5000;
constexpr int64_t kMinAcsAckTimeoutMilliseconds = 0;
constexpr int64_t kMaxAcsAckTimeoutMilliseconds = 60000;  // 1 minute

constexpr int64_t kMinBwBucketTimeInMilliseconds = 1;
constexpr int64_t kMaxBwBucketTimeInMilliseconds = 10000;
constexpr int64_t kBwBucketTimeInMilliseconds = 1;

constexpr int64_t kMinExpectedMaxLatencyInMilliseconds = 0;
constexpr int64_t kMaxExpectedMaxLatencyInMilliseconds = 10000;
constexpr int64_t kExpectedMaxLatencyInMilliseconds = 3;

constexpr int64_t kEnableMillisecondBandwidthOutputEnable = 0;
constexpr int64_t kMinEnableMillisecondBandwidthOutputEnable = 0;
constexpr int64_t kMaxEnableMillisecondBandwidthOutputEnable = 1;

constexpr int32_t kMillisecondBandwidthOutputBatchSize = 100;
constexpr int32_t kMinMillisecondBandwidthOutputBatchSize = 1;
constexpr int32_t kMaxMillisecondBandwidthOutputBatchSize = 1000;

constexpr double kMinAcceptedMaxBWPerNICInBitsPerSec = 1; // No negative BW
constexpr double kMaxAcceptedMaxBWPerNICInBitsPerSec = __DBL_MAX__;

constexpr double kMinBaselineLatencyInNanoSecond = __DBL_MIN__;
constexpr double kMaxBaselineLatencyInNanoSecond = __DBL_MAX__;
constexpr double kBaselineLatencyInNanoSecond = 10000; // 10 microseconds

constexpr double kMaxBucketsForLatencyHistogramInNanoSec = 1000000000;        // 10^9 buckets
constexpr double kScaleForLatencyHistogramInNanoSec = 100;
constexpr double kMaxBucketsForLatencyVarianceHistogramInNanoSec = 1000000000;        // 10^9 buckets
constexpr double kScaleForLatencyVarianceHistogramInNanoSec = 100;
constexpr double kMaxBucketsForSizeHistogramInBytes = 10000000;        // 10^7 buckets
constexpr double kScaleForSizeHistogramInBytes = 1;
constexpr double kScaleForBWHistogramInBitsPerSec = 1;
constexpr double kScaleForNicBWVarianceHistogramInBitsPerSec = 1;

constexpr double kScaleForNicBWHistogramInBitsPerSec = 1;

constexpr double kMaxAcceptedMaxBucketsForLatencyHistogramInNanoSec = __DBL_MAX__;
constexpr double kMinAcceptedMaxBucketsForLatencyHistogramInNanoSec = 1;
constexpr double kMaxAcceptedScaleLatencyHistogramInNanoSec = __DBL_MAX__;
constexpr double kMinAcceptedScaleLatencyHistogramInNanoSec = 1; // prevent the
                                                                 // user from
                                                                 // forcing us
                                                                 // into an
                                                                 // infinite
                                                                 // loop with 0
                                                                 // here.

constexpr double kMaxAcceptedMaxBucketsForLatencyVarianceHistogramInNanoSec = __DBL_MAX__;
constexpr double kMinAcceptedMaxBucketsForLatencyVarianceHistogramInNanoSec = 1;
constexpr double kMaxAcceptedScaleLatencyVarianceHistogramInNanoSec = __DBL_MAX__;
constexpr double kMinAcceptedScaleLatencyVarianceHistogramInNanoSec = 1;

constexpr double kMaxAcceptedMaxBucketsForSizeHistogramInBytes = __DBL_MAX__;
constexpr double kMinAcceptedMaxBucketsForSizeHistogramInBytes = 1;
constexpr double kMaxAcceptedScaleSizeHistogramInBytes = __DBL_MAX__;
constexpr double kMinAcceptedScaleSizeHistogramInBytes = 1;

constexpr double kMinAcceptedMaxBucketsForBWHistogramInBitsPerSec = 1;
constexpr double kMaxAcceptedMaxBucketsForBWHistogramInBitsPerSec = __DBL_MAX__;
constexpr double kMinAcceptedScaleBWHistogramInBitsPerSec = 1;
constexpr double kMaxAcceptedScaleBWHistogramInBitsPerSec = __DBL_MAX__;

constexpr double kMinAcceptedMaxBucketsForNicBWHistogramInBitsPerSec = 1;
constexpr double kMaxAcceptedMaxBucketsForNicBWHistogramInBitsPerSec = __DBL_MAX__;
constexpr double kMinAcceptedScaleNicBWHistogramInBitsPerSec = 1;
constexpr double kMaxAcceptedScaleNicBWHistogramInBitsPerSec = __DBL_MAX__;

constexpr double kMinAcceptedMaxBucketsForNicBWVarianceHistogramInBitsPerSec = 0;
constexpr double kMaxAcceptedMaxBucketsForNicBWVarianceHistogramInBitsPerSec = __DBL_MAX__;
constexpr double kMinAcceptedScaleNicBWVarianceHistogramInBitsPerSec = 0;
constexpr double kMaxAcceptedScaleNicBWVarianceHistogramInBitsPerSec = __DBL_MAX__;

constexpr size_t kMinNonAggregatedDataSizeCap = 0;
constexpr size_t kMaxNonAggregatedDataSizeCap = std::numeric_limits<size_t>::max();
constexpr size_t kNonAggregatedDataSizeCap = 1 * 1024 * 1024; // 1 MB

constexpr double kMinBucketBaseForLatencyHistogram = 1.0;
constexpr double kMaxBucketBaseForLatencyHistogram = __DBL_MAX__;
constexpr double kBucketBaseForLatencyHistogram = 1.2;

constexpr double kMinBucketBaseForSizeHistogram = 1.0;
constexpr double kMaxBucketBaseForSizeHistogram = __DBL_MAX__;
constexpr double kBucketBaseForSizeHistogram = 1.2;

constexpr double kMinBucketBaseForBWHistogram = 1.0;
constexpr double kMaxBucketBaseForBWHistogram = __DBL_MAX__;
constexpr double kBucketBaseForBWHistogram = 1.2;

constexpr double kMinBucketBaseForNicBWHistogram = 1.0;
constexpr double kMaxBucketBaseForNicBWHistogram = __DBL_MAX__;
constexpr double kBucketBaseForNicBWHistogram = 1.2;

constexpr double kMinBucketBaseForNicBWVarianceHistogram = 1.0;
constexpr double kMaxBucketBaseForNicBWVarianceHistogram = __DBL_MAX__;
constexpr double kBucketBaseForNicBWVarianceHistogram = 1.2;

constexpr double kMinBucketBaseForLatencyVarianceHistogram = 1.0;
constexpr double kMaxBucketBaseForLatencyVarianceHistogram = __DBL_MAX__;
constexpr double kBucketBaseForLatencyVarianceHistogram = 1.2;

namespace gpuviz::params {
    // Removes all whitespaces from a string.
    void trimString(std::string &str)
    {
        if (str.empty())
        {
            return;
        }
        str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    }

    double readDoubleFromEnv(const char *env_name, double min_value, double max_value, double default_value, double wrong_input_value)
    {
        char *str = getenv(env_name);
        if (!str) {
            // value is not set
            return default_value;
        }
        double value = 0;
        if (strlen(str) > 0 && absl::SimpleAtod(str, &value) && value >= min_value && value <= max_value) {
            // valid value case
            return value;
        }
        // invalid value case
        return wrong_input_value;
    }

template <typename T>
bool SimpleAtoT(absl::string_view str, T* num) {
    if constexpr (std::is_integral_v<T>) return absl::SimpleAtoi<T>(str, num);
    if constexpr (std::is_same_v<T,double>) return absl::SimpleAtod(str,num);
    if constexpr (std::is_same_v<T,float>) return absl::SimpleAtof(str,num);
    if constexpr (std::is_same_v<T,bool>) return absl::SimpleAtob(str,num);
    return false;
}

template <typename T>
T ReadNumericFromEnv(const NumericParam<T>& param) {
    char *str = getenv(std::string(param.env_name).c_str());
    if (!str) {
        // value is not set
        if (param.fallback == nullptr) return param.default_value;
        return (*param.fallback)();
    }
    T value = 0;
    if (strlen(str) > 0 && SimpleAtoT<T>(str, &value) && value >= param.min && value <= param.max) {
        // valid value case
        return value;
    }
    // invalid value case
    return param.wrong_input_value;
}

std::string ReadStringFromEnv(const StringParam& param) {
    char *str = getenv(std::string(param.env_name).c_str());
    if (str && strlen(str) > 0) {
        return std::string{str};
    }
    return std::string{param.default_value};
}

template <typename T>
T NumericParam<T>::value() const {
    // For the moment, we just read it from the environment, but in the future,
    // this function will do other things.
    return ReadNumericFromEnv(*this);
}

std::string StringParam::value() const {
    std::string value = ReadStringFromEnv(*this);
    if (this->trim) {
        trimString(value);
    }
    return value;
}

    int64_t readIntFromEnv(const char *env_name, int64_t min_value, int64_t max_value, int64_t default_value, int64_t wrong_input_value)
    {
        char *str = getenv(env_name);
        if (!str) {
            // value is not set
            return default_value;
        }
        int64_t value = 0;
        if (strlen(str) > 0 && absl::SimpleAtoi(str, &value) && value >= min_value && value <= max_value) {
            // valid value case
            return value;
        }
        // invalid value case
        return wrong_input_value;
    }

    bool readBoolFromEnv(const char *env_name, bool default_value)
    {
        return readIntFromEnv(env_name, /*min_value=*/0, /*max_value=*/1, default_value, default_value);
    }

    // Takes in an environment variable name, a minimum expected value, and
    // a function to call to get the value if there is no value provided for
    // the environment variable or if the value is less than the minimum
    // allowed.
    int64_t readIntFromEnvWithFallbackFunction(const char *env_name, int64_t min_value, const std::function<int64_t()>& fallback) {
        const char *str = getenv(env_name);
        if (str && strlen(str) > 0) {
            // Value was set in the environment
            int64_t value = 0;
            if (absl::SimpleAtoi(str, &value) && value >= min_value) {
                return value;
            }
        }
        // If value was not set or was out of range, default to the same
        // read interval used for the histogram output
        return fallback();
    }

    std::string readStrFromEnv(const std::string& env_name)
    {
        char *str = getenv(env_name.c_str());
        if (str && strlen(str) > 0)
        {
            return std::string(str);
        }
        return "";
    }

    ncclTelemetryMode mapIntToTelemetryMode(const int64_t i)
    {
        switch (i)
        {
            case 0:
                return ncclTelemetryMode::kOff;
            case 1:
                return ncclTelemetryMode::kWriteOnLocalDiskOnly;
            case 2:
                return ncclTelemetryMode::kReserved;
            case 3:
                return ncclTelemetryMode::kUploadOnlyControlledByMds;
            case 4:
                return ncclTelemetryMode::kBothWriteAndUploadControlledByMds;
        }

        return ncclTelemetryMode::kUndefined;
    }

int GetAbslLogVerbosity() {
    return kAbslLogVerbosity.value();
}

// Controls telemetry mode.  Off:0, Local Disk:1, Upload:3, Local Disk + Upload:4
NumericParam<int32_t> nccl_telemetry_mode_param = {
    {
        .env_name = "NCCL_TELEMETRY_MODE",
    },
    /*min*/ 0,
    /*max*/ 4,
    /*default_value*/ 0,
    /*wrong_input_value*/ 0,
};

bool IsProfilerPluginEnabled() {
    // NCCL_TELEMETRY_MODE indicates active Profiler Plugin
    // NetPlugin should not activate GPUViz client (see code in net_ib.cc)
    // Profiler Plugin is not active if env var is not set or wrong
    return nccl_telemetry_mode_param.value()!=0;
}

ncclTelemetryMode GetPluginTelemetryMode() {
    // Profiler Plugin env var has higher priority than the env var for NetPlugin.
    if (IsProfilerPluginEnabled()) {
        return mapIntToTelemetryMode(nccl_telemetry_mode_param.value());
    } else {
        return mapIntToTelemetryMode(kNetPluginTelemetryMode.value());
    }
}

std::string AcsServerEndPointOverwritten() {
    return std::string(kAcsEndPointOverride.value());
}

        bool IsAcsEndPointRegional()
        {
            return readIntFromEnv("NCCL_GPUVIZ_ACS_ENDPOINT_REGIONAL", kAcsEndPointZonal, kAcsEndPointRegional, kAcsEndPointZonalDefault, kAcsEndPointZonalDefault) == kAcsEndPointRegional;
        }

        int64_t GetAcsServerAckTimeoutInMilliseconds()
        {
            return readIntFromEnv("NCCL_GPUVIZ_ACS_ACK_TIMEOUT_IN_MILLISECONDS", kMinAcsAckTimeoutMilliseconds, kMaxAcsAckTimeoutMilliseconds, kAcsAckTimeoutMilliseconds, kAcsAckTimeoutMilliseconds);
        }

std::string GetDirectoryPathForGPUVizFiles() {
    return kOutputFilesPath.value();
}

int64_t GetReadIntervalInMicroSeconds() {
    return kReadIntervalInMicroseconds.value();
}

int64_t GetFileRotationIntervalInSeconds() {
    return kFileRotationIntervalInSeconds.value();
}

int64_t GetBandwidthBurstBucketCount() {
    return kBwBurstBucketCount.value();
}

int64_t GetJitterBandwidthBucketCount() {
    return kJitterBwBucketCount.value();
}

bool IsBandwidthHistogramCollectionEnabled() {
    // default off
    return kBWHistogramCollectionEnable.value() != 0;
}

bool IsVarianceDetectionEnabled() {
    // default on
    return kVarianceHistCollectionEnable.value() != 0;
}

        int64_t GetBandwidthBucketTimeInMilliSeconds()
        {
            return readIntFromEnv("NCCL_GPUVIZ_BW_BUCKET_TIME_IN_MILLISECONDS", kMinBwBucketTimeInMilliseconds, kMaxBwBucketTimeInMilliseconds, kBwBucketTimeInMilliseconds, kBwBucketTimeInMilliseconds);
        }
        int64_t GetMaxExpectedLatencyPerTxInMilliSeconds()
        {
            return readIntFromEnv("NCCL_GPUVIZ_MAX_EXPECTED_LATENCY_PER_TRANSACTION_IN_MILLISECONDS", kMinExpectedMaxLatencyInMilliseconds, kMaxExpectedMaxLatencyInMilliseconds, kExpectedMaxLatencyInMilliseconds, kExpectedMaxLatencyInMilliseconds);
        }
        double GetBaselineLatencyInNanoSec()
        {
            return readDoubleFromEnv("NCCL_GPUVIZ_BASELINE_LATENCY_IN_NANOSECOND", kMinBaselineLatencyInNanoSecond, kMaxBaselineLatencyInNanoSecond, kBaselineLatencyInNanoSecond, kBaselineLatencyInNanoSecond);
        }
        std::pair<double, double> GetMaxAndTargetBWPerNICInBitsPerSec(double nic_speed, double empr_nic_speed)
        {
            double max_bw_per_nic_in_bits_per_sec = readDoubleFromEnv("NCCL_GPUVIZ_MAX_BW_PER_NIC_IN_BITS_PER_SEC", kMinAcceptedMaxBWPerNICInBitsPerSec, kMaxAcceptedMaxBWPerNICInBitsPerSec, nic_speed *GIGA_BITS_TO_BITS, nic_speed *GIGA_BITS_TO_BITS);
            // Target BW should be smaller than or equal to Max BW NIC offers
            double target_bw_per_nic_in_bits_per_sec = readDoubleFromEnv("NCCL_GPUVIZ_TARGET_BW_PER_NIC_IN_BITS_PER_SEC", kMinAcceptedMaxBWPerNICInBitsPerSec, max_bw_per_nic_in_bits_per_sec, empr_nic_speed * GIGA_BITS_TO_BITS, empr_nic_speed * GIGA_BITS_TO_BITS);
            return {max_bw_per_nic_in_bits_per_sec, target_bw_per_nic_in_bits_per_sec};
        }

        namespace {
        BucketerParams GetBucketParams(const char* max_buckets_env_var_name,
            double max_buckets_min_val, double max_buckets_max_val, double max_buckets_default_val,
            const char* scale_env_var_name,
            double scale_min_val, double scale_max_val, double scale_default_val,
            const char* base_env_var_name,
            double base_min_val, double base_max_val, double base_default_val)
            {
                double max_buckets = readDoubleFromEnv(max_buckets_env_var_name,
                    max_buckets_min_val,
                    max_buckets_max_val,
                    max_buckets_default_val,
                    max_buckets_default_val);
                double scale = readDoubleFromEnv(scale_env_var_name,
                    scale_min_val,
                    scale_max_val,
                    scale_default_val,
                    scale_default_val);
                if (max_buckets <= scale) {
                    // max_buckets should never be smaller than scale for the exponential histogram.
                    // If user accidentally chooses so, we should use default values for both max_bucket and scale back.
                    max_buckets = max_buckets_default_val;
                    scale = scale_default_val;
                }
                double base = readDoubleFromEnv(base_env_var_name,
                    base_min_val,
                    base_max_val,
                    base_default_val,
                    base_default_val);
                if (base <= 1) {
                    // Base must be greater than 1 for a exponential histogram.
                    // We should use the default value for base if the user misconfigured this.
                    base = base_default_val;
                }
                if (base >= max_buckets / scale) {
                    // Base should be less than max_buckets/scale to ensure there are more than 2 buckets
                    // and to avoid overflow in calculations.
                    // We should use the default value for base if the user misconfigured this.
                    base = base_default_val;
                }
                return {.max_buckets = max_buckets, .scale = scale, .base = base};
            }
        }  // namespace.

        BucketerParams GetBucketerParamsForLatencyHistogram()
        {
            return GetBucketParams("NCCL_GPUVIZ_GET_MAX_BUCKETS_LATENCY_HISTOGRAM_IN_NANOSECONDS",
                kMinAcceptedMaxBucketsForLatencyHistogramInNanoSec,
                kMaxAcceptedMaxBucketsForLatencyHistogramInNanoSec,
                kMaxBucketsForLatencyHistogramInNanoSec,
                "NCCL_GPUVIZ_GET_SCALE_LATENCY_HISTOGRAM_IN_NANOSECONDS",
                kMinAcceptedScaleLatencyHistogramInNanoSec,
                kMaxAcceptedScaleLatencyHistogramInNanoSec,
                kScaleForLatencyHistogramInNanoSec,
                "NCCL_GPUVIZ_BUCKET_BASE_LATENCY_HISTOGRAM",
                kMinBucketBaseForLatencyHistogram,
                kMaxBucketBaseForLatencyHistogram,
                kBucketBaseForLatencyHistogram
            );
        }

        BucketerParams GetBucketerParamsForSizeHistogram()
        {
            return GetBucketParams("NCCL_GPUVIZ_GET_MAX_BUCKETS_SIZE_HISTOGRAM_IN_BYTES",
                kMinAcceptedMaxBucketsForSizeHistogramInBytes,
                kMaxAcceptedMaxBucketsForSizeHistogramInBytes,
                kMaxBucketsForSizeHistogramInBytes,
                "NCCL_GPUVIZ_GET_SCALE_SIZE_HISTOGRAM_IN_BYTES",
                kMinAcceptedScaleSizeHistogramInBytes,
                kMaxAcceptedScaleSizeHistogramInBytes,
                kScaleForSizeHistogramInBytes,
                "NCCL_GPUVIZ_BUCKET_BASE_SIZE_HISTOGRAM",
                kMinBucketBaseForSizeHistogram,
                kMaxBucketBaseForSizeHistogram,
                kBucketBaseForSizeHistogram
            );
        }

        BucketerParams GetBucketerParamsForBWHistogram(double nic_speed)
        {
            return GetBucketParams("NCCL_GPUVIZ_GET_MAX_BUCKETS_BW_HISTOGRAM_IN_BITS_PER_SEC",
                kMinAcceptedMaxBucketsForBWHistogramInBitsPerSec,
                kMaxAcceptedMaxBucketsForBWHistogramInBitsPerSec,
                nic_speed * GIGA_BITS_TO_BITS,
                "NCCL_GPUVIZ_GET_SCALE_BW_HISTOGRAM_IN_BITS_PER_SEC",
                kMinAcceptedScaleBWHistogramInBitsPerSec,
                kMaxAcceptedScaleBWHistogramInBitsPerSec,
                kScaleForBWHistogramInBitsPerSec,
                "NCCL_GPUVIZ_BUCKET_BASE_BW_HISTOGRAM",
                kMinBucketBaseForBWHistogram,
                kMaxBucketBaseForBWHistogram,
                kBucketBaseForBWHistogram
            );
        }

        BucketerParams GetBucketerParamsForNicBWHistogram(double nic_speed)
        {
            return GetBucketParams("NCCL_GPUVIZ_GET_MAX_BUCKETS_NIC_BW_HISTOGRAM_IN_BITS_PER_SEC",
                kMinAcceptedMaxBucketsForNicBWHistogramInBitsPerSec,
                kMaxAcceptedMaxBucketsForNicBWHistogramInBitsPerSec,
                nic_speed * GIGA_BITS_TO_BITS,
                "NCCL_GPUVIZ_GET_SCALE_NIC_BW_HISTOGRAM_IN_BITS_PER_SEC",
                kMinAcceptedScaleNicBWHistogramInBitsPerSec,
                kMaxAcceptedScaleNicBWHistogramInBitsPerSec,
                kScaleForNicBWHistogramInBitsPerSec,
                "NCCL_GPUVIZ_BUCKET_BASE_NIC_BW_HISTOGRAM",
                kMinBucketBaseForNicBWHistogram,
                kMaxBucketBaseForNicBWHistogram,
                kBucketBaseForNicBWHistogram
            );
        }

        BucketerParams GetBucketerParamsForNicBWVarianceHistogram(double nic_speed)
        {
            return GetBucketParams("NCCL_GPUVIZ_GET_MAX_BUCKETS_NIC_BW_VARIANCE_HISTOGRAM_IN_BITS_PER_SEC",
                kMinAcceptedMaxBucketsForNicBWVarianceHistogramInBitsPerSec,
                kMaxAcceptedMaxBucketsForNicBWVarianceHistogramInBitsPerSec,
                nic_speed * GIGA_BITS_TO_BITS,
                "NCCL_GPUVIZ_GET_SCALE_NIC_BW_VARIANCE_HISTOGRAM_IN_BITS_PER_SEC",
                kMinAcceptedScaleNicBWVarianceHistogramInBitsPerSec,
                kMaxAcceptedScaleNicBWVarianceHistogramInBitsPerSec,
                kScaleForNicBWVarianceHistogramInBitsPerSec,
                "NCCL_GPUVIZ_BUCKET_BASE_NIC_BW_VARIANCE_HISTOGRAM",
                kMinBucketBaseForNicBWVarianceHistogram,
                kMaxBucketBaseForNicBWVarianceHistogram,
                kBucketBaseForNicBWVarianceHistogram
            );
        }

        BucketerParams GetBucketerParamsForNicLatencyVarianceHistogram()
        {
            return GetBucketParams("NCCL_GPUVIZ_GET_MAX_BUCKETS_NIC_LATENCY_VARIANCE_HISTOGRAM_IN_NANOSECONDS",
                kMinAcceptedMaxBucketsForLatencyVarianceHistogramInNanoSec,
                kMaxAcceptedMaxBucketsForLatencyVarianceHistogramInNanoSec,
                kMaxBucketsForLatencyVarianceHistogramInNanoSec,
                "NCCL_GPUVIZ_GET_SCALE_NIC_LATENCY_VARIANCE_HISTOGRAM_IN_NANOSECONDS",
                kMinAcceptedScaleLatencyVarianceHistogramInNanoSec,
                kMaxAcceptedScaleLatencyVarianceHistogramInNanoSec,
                kScaleForLatencyVarianceHistogramInNanoSec,
                "NCCL_GPUVIZ_BUCKET_BASE_NIC_LATENCY_VARIANCE_HISTOGRAM",
                kMinBucketBaseForLatencyVarianceHistogram,
                kMaxBucketBaseForLatencyVarianceHistogram,
                kBucketBaseForLatencyVarianceHistogram
            );
        }

        bool IsMillisecondBandwidthOutputEnabled() {
            // ms-BW feature  (relies on NetPlugin) cannot be enabled with Profiler Plugin.
            return readIntFromEnv("NCCL_GPUVIZ_ENABLE_MILLISECOND_BANDWIDTH_OUTPUT", kMinEnableMillisecondBandwidthOutputEnable,
                kMaxEnableMillisecondBandwidthOutputEnable, kEnableMillisecondBandwidthOutputEnable, kEnableMillisecondBandwidthOutputEnable) != 0;
        }

        std::string GetMillisecondBandwidthOutputFilenamePrefix() {
            std::string prefix = readStrFromEnv("NCCL_GPUVIZ_MILLSECOND_BANDWIDTH_FILE_PREFIX");
            return prefix.empty() ? kMillisecondBandwidthOutputFilePrefix : prefix;
        }

        int64_t GetMillisecondBandwidthOutputRotationRate() {
            return readIntFromEnvWithFallbackFunction("NCCL_GPUVIZ_MILLISECOND_BANDWIDTH_FILE_ROTATION_INTERVAL_IN_SECONDS",
                1 /* One second minumum. */,
                &GetFileRotationIntervalInSeconds);
        }

        int64_t GetMillisecondBandwidthOutputBatchSize() {
            return readIntFromEnv("NCCL_GPUVIZ_MILLISECOND_BANDWIDTH_BATCH_SIZE",
                kMinMillisecondBandwidthOutputBatchSize,
                kMaxMillisecondBandwidthOutputBatchSize,
                kMillisecondBandwidthOutputBatchSize, kMillisecondBandwidthOutputBatchSize);
        }

        int64_t GetMillisecondBandwidthOutputIntervalInMilliSeconds() {
            return readIntFromEnvWithFallbackFunction("NCCL_GPUVIZ_MILLISECOND_BANDWIDTH_OUTPUT_INTERVAL",
                100 /* One tenth of a second minimum. */,
                [](){return GetReadIntervalInMicroSeconds()/1000;});
        }

        bool isSendLatencyUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_SEND_LATENCY_HISTOGRAM_UPLOAD",
                kSendLatencyHistUploadEnable);
        }

        bool isRecvLatencyUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_RECV_LATENCY_HISTOGRAM_UPLOAD",
                kRecvLatencyHistUploadEnable);
        }

        bool isSendMessageSizeUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_SEND_MESSAGE_SIZE_HISTOGRAM_UPLOAD",
                kSendMsgSizeHistUploadEnable);
        }

        bool isRecvMessageSizeUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_RECEIVE_MESSAGE_SIZE_HISTOGRAM_UPLOAD",
                kRecvMsgSizeHistUploadEnable);
        }

        bool isSendBandwidthUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_SEND_BANDWIDTH_HISTOGRAM_UPLOAD",
                kSendBWHistUploadEnable);
        }

        bool isRecvBandwidthUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_RECV_BANDWIDTH_HISTOGRAM_UPLOAD",
                kRecvBWHistUploadEnable);
        }

        bool isBandwidthVarianceUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_BANDWIDTH_VARIANCE_HISTOGRAM_UPLOAD",
                kBWVarianceHistUploadEnable);
        }

        bool isLatencyVarianceUploadEnabled() {
            return readBoolFromEnv("NCCL_TELEMETRY_LATENCY_VARIANCE_HISTOGRAM_UPLOAD",
                kLatencyVarianceHistUploadEnable);
        }

        size_t GetNonAggregatedDataSizeCap() {
            return readIntFromEnv("NCCL_GPUVIZ_NON_AGGREGATED_DATA_SIZE_CAP",
                kMinNonAggregatedDataSizeCap, kMaxNonAggregatedDataSizeCap,
                kNonAggregatedDataSizeCap, kNonAggregatedDataSizeCap);
        }
}  // namespace gpuviz::params

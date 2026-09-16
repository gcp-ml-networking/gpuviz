// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_aggregator.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <queue>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "distribution_bucketer.h"
#include "google/protobuf/util/time_util.h"
#include "include/nccl_net.h"
#include "millisecond_bandwidth_output.h"
#include "nccl_stats.h"
#include "params.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "utils.h"

#define BYTES_TO_BITS_MULTIPLIER 8
#define DISTRIBUTION_TYPE(x) (x##Idx)
#define IS_SET(val, mask) (((val) & mask) == mask)

namespace gpuviz {

static_assert(
    static_cast<int>(protoDist::ncclStatsPluginType::NetPlugin) ==
        static_cast<int>(ncclStatsPluginType::NetPlugin),
    "protoDist::ncclStatsPluginType and ncclStatsPluginType do not match");
static_assert(
    static_cast<int>(protoDist::ncclStatsPluginType::CollNetPlugin) ==
        static_cast<int>(ncclStatsPluginType::CollNetPlugin),
    "protoDist::ncclStatsPluginType and ncclStatsPluginType do not match");

static_assert(
    static_cast<int>(protoDist::EntityNVLConnection) ==
        static_cast<int>(EntityNVLConnection),
    "protoDist::EntityNVLConnection and EntityNVLConnection do not match");
static_assert(
    static_cast<int>(protoDist::EntityPCIConnection) ==
        static_cast<int>(EntityPCIConnection),
    "protoDist::EntityPCIConnection and EntityPCIConnection do not match");
static_assert(
    static_cast<int>(protoDist::EntityTCPConnection) ==
        static_cast<int>(EntityTCPConnection),
    "protoDist::EntityTCPConnection and EntityTCPConnection do not match");
static_assert(
    static_cast<int>(protoDist::EntityRDMAConnection) ==
        static_cast<int>(EntityRDMAConnection),
    "protoDist::EntityRDMAConnection and EntityRDMAConnection do not match");

namespace utils {
namespace ncclaggregator {
static bool isBWCollectionEnabledinBitMap(
    uint64_t distribution_collector_bitmap, uint8_t latencyDistribution,
    uint8_t sizeDistribution) {
  bool LatencySWDistCollectionEnabled =
      IS_SET(distribution_collector_bitmap, latencyDistribution);
  bool MessageSizeDistCollectionEnabled =
      IS_SET(distribution_collector_bitmap, sizeDistribution);
  return LatencySWDistCollectionEnabled && MessageSizeDistCollectionEnabled;
}
// Skip adding unused and carry-over buckets to the histogram.
uint64_t getBwVecValidLimit(uint64_t carry_over_buckets,
                            uint64_t unused_buckets, uint64_t vec_size) {
  /*
  carry over buckets are last K buckets before the unused buckets.
  To examplain with example:

  Example 1: No Used buckets
  Bytes array

                                  |<-----Carry Over----->|
  idx    0      1      2      3      4      5      6     7
  values 8      9      5      6      3      2      2     2


  Example 2: 2 Unused buckets
  Bytes array
                                |<-Carry Over->|<-unused-->|
  idx    0      1      2      3      4      5      6     7
  values 8      9      5      6      5      6      0     0


  */
  uint64_t skip_buckets = unused_buckets + carry_over_buckets;
  return (skip_buckets >= vec_size) ? 0 : (vec_size - skip_buckets);
}
static double calculateBWPerBucketInBitsPerSec(double sizePerBucketInBytes,
                                               double latencyPerBucketInMs) {
  /*
  bw to be calculated in bits/sec. sizePerBucketInBytes is the number of bytes
  sent during a single bucket interval. Multiplication with 8 is to convert
  bytes to bits, and dividing it by the time a single bucket covers in
  milliseconds. To avoid latencyPerBucketInMs becoming 0 when converted to
  seconds, we multiply sizePerBucketInBytes with
  SECOND_TO_MILLISECOND_MULTIPLIER.
  */
  double bw =
      (((double)sizePerBucketInBytes * 8 * SECOND_TO_MILLISECOND_MULTIPLIER) /
       (double)latencyPerBucketInMs);
  return bw;
}
void populateNcclStatsConnectionIdentifier(
    ncclStatsPluginType ncclPluginType, std::string ncclPluginName,
    std::string gpuPciAddr, std::string gpuUuid, int32_t pluginMajor,
    int32_t pluginMinor, protoDist::ConnectionIdentifier& messageConnId) {
  messageConnId.set_nccl_plugin_type(
      protoDist::ncclStatsPluginType(ncclPluginType));
  messageConnId.set_nccl_plugin_name(ncclPluginName);
  messageConnId.set_gpu_pci_addr(gpuPciAddr);
  if (!gpuUuid.empty()) {
    messageConnId.set_gpu_uuid(gpuUuid);
  }
  if (pluginMajor != 0 || pluginMinor != 0) {
    messageConnId.mutable_plugin_version()->set_major(pluginMajor);
    messageConnId.mutable_plugin_version()->set_minor(pluginMinor);
  }
}
// return the index of the last histogram added to the connection stats.
// If no histogram is added, return -1.
int populateStatsDistributionWithDistributionBucketer(
    protoDist::ConnectionStats* connStats, DistributionBucketer* distBucketer,
    protoDist::DistType distributionType) {
  if (distBucketer->GetSampleSubmitted()) {
    protoDist::StatsDistribution* stats_dist = connStats->add_histogram();
    stats_dist->set_dist_type(distributionType);

    stats_dist->set_max(distBucketer->GetMax().value());
    stats_dist->set_min(distBucketer->GetMin().value());
    stats_dist->set_avg(distBucketer->GetAvg().value());

    stats_dist->set_scale_factor(distBucketer->GetScale());
    stats_dist->set_max_bucket(distBucketer->GetMaxBucket());
    stats_dist->set_base(distBucketer->GetBase());

    std::vector<int64_t> bucket_counts = distBucketer->GetBucketCounts();
    int64_t total_samples = 0;
    stats_dist->mutable_bucket_counts()->Reserve(bucket_counts.size());
    for (int64_t count : bucket_counts) {
      total_samples += count;
      stats_dist->add_bucket_counts(count);
    }
    stats_dist->set_num_samples(total_samples);
    return connStats->histogram_size() - 1;
  }
  return -1;
}
}  // namespace ncclaggregator
}  // namespace utils

using namespace utils::ncclaggregator;
using namespace gpuviz::utils;

NcclStatsAggregator::NcclStatsAggregator(
    ncclDebugLogger_t logFunction, uint64_t distributionCollectorBitmap,
    gpuviz::MillisecondBandwidthOutput* msec_bw)
    : bw_collection_start_time_milliseconds_(0),
      bw_bucket_time_in_milliseconds_(
          gpuviz::params::GetBandwidthBucketTimeInMilliSeconds()),
      non_aggregated_data_size_cap_(params::GetNonAggregatedDataSizeCap()),
      log_function_(logFunction),
      distribution_collector_bitmap_(distributionCollectorBitmap),
      bw_hist_collection_enabled_(
          gpuviz::params::IsBandwidthHistogramCollectionEnabled()),
      baseline_latency_in_ns_(gpuviz::params::GetBaselineLatencyInNanoSec()),
      variance_hist_collection_enabled_(
          gpuviz::params::IsVarianceDetectionEnabled()),
      msec_bw_output_enabled_(
          gpuviz::params::IsMillisecondBandwidthOutputEnabled()),
      msec_bw_(msec_bw) {
  int64_t bw_burst_bucket_count =
      gpuviz::params::GetBandwidthBurstBucketCount();
  // This accounts for the additional size needed to capture the BW stats if the
  // bucket flip doesn't happen at expected time.
  int64_t jitter_bandwidth_bucket_count =
      gpuviz::params::GetJitterBandwidthBucketCount();
  bw_burst_count_ = (bw_burst_bucket_count + jitter_bandwidth_bucket_count);
  max_expected_latency_per_tx_milliseconds_ =
      gpuviz::params::GetMaxExpectedLatencyPerTxInMilliSeconds();
  carry_over_buckets_ = max_expected_latency_per_tx_milliseconds_ /
                        bw_bucket_time_in_milliseconds_;
}

void NcclStatsAggregator::InitConstants(absl::string_view nccl_plugin_name) {
  if (!nic_speed_init_) {
    if (absl::StartsWith(nccl_plugin_name, "profiler") ||
        nccl_plugin_name == "net-ib") {
      // currently profiler plugin is only available on A3U and later
      nic_speed_init_ = true;
      nic_speed_ = 400;
      empr_nic_speed_ = 385;
    } else {
      if (nccl_plugin_name == "FasTrak") {
        nic_speed_init_ = true;
      }
      std::string log_message =
          "Unknown plugin name, using default values for NIC speed. High "
          "frequency bandwidth estimates may be incorrect.";
      addLogMsg(log_message, NCCL_LOG_WARN, __PRETTY_FUNCTION__, __LINE__);
      nic_speed_ = 200;
      empr_nic_speed_ = 200;
    }

    auto [max_bw_per_nic_in_bits_per_sec, target_bw_per_nic_in_bits_per_sec] =
        gpuviz::params::GetMaxAndTargetBWPerNICInBitsPerSec(nic_speed_,
                                                            empr_nic_speed_);
    target_bw_per_nic_in_bits_per_sec_ = target_bw_per_nic_in_bits_per_sec;
    // calculate max_bytes_per_bucket_ by multiplying
    // max_bw_per_nic_in_bits_per_sec * bw_bucket_time_in_milliseconds.
    // max_bw_per_nic_in_bits_per_sec is in bits per sec and
    // bw_bucket_time_in_milliseconds is in millisecond. Converting
    // bw_bucket_time_in_milliseconds to seconds can become 0 so we first
    // multiply (max_bw_per_nic_in_bits_per_sec *
    // bw_bucket_time_in_milliseconds) and then divide by
    // BYTES_TO_BITS_MULTIPLIER(8) to convert bits to bytes and divide by
    // SECOND_TO_MILLISECOND_MULTIPLIER(1000) to convert milliseconds to
    // seconds.
    max_bytes_per_bucket_ =
        (max_bw_per_nic_in_bits_per_sec * bw_bucket_time_in_milliseconds_) /
        (BYTES_TO_BITS_MULTIPLIER * SECOND_TO_MILLISECOND_MULTIPLIER);
  }
}

NcclStatsAggregator::~NcclStatsAggregator() {
  if (!clean().ok()) {
    std::string log_msg =
        "Failed to clean the aggregator while destructing NcclStatsAggregator.";
    addLogMsg(log_msg, NCCL_LOG_WARN, __PRETTY_FUNCTION__, __LINE__);
  }
}

void NcclStatsAggregator::addLogMsg(absl::string_view message,
                                    ncclDebugLogLevel level,
                                    const char* pretty_func, uint32_t line,
                                    ncclDebugLogSubSys subsys,
                                    std::string category_name, int32_t category,
                                    int32_t msg_id) {
  std::string msg = absl::StrCat(
      kLogPrefix, level == NCCL_LOG_WARN ? kNonFatalStrForNcclWarn : "",
      message);

  log_function_(level, subsys, pretty_func, line, msg.c_str());

  protoDist::LogEntry entry;

  google::protobuf::Timestamp* timestamp = entry.mutable_time_stamp();
  *timestamp = google::protobuf::util::TimeUtil::GetCurrentTime();

  entry.set_entry_category(category);
  entry.set_entry_message_id(msg_id);
  entry.set_category_name(std::move(category_name));
  entry.set_event_message(std::move(msg));
  entry.set_severity(level);
  const size_t entry_size = entry.SpaceUsedLong();

  absl::MutexLock lock(&log_messages_mtx_);
  if (log_messages_size_ + entry_size <= non_aggregated_data_size_cap_) {
    log_messages_.push(std::move(entry));
    log_messages_size_ += entry_size;
  }
}

absl::Status NcclStatsAggregator::addEvent(const uint8_t* event_data,
                                           size_t len) {
  absl::MutexLock lock(&events_mtx_);
  if (events_size_ + len <= non_aggregated_data_size_cap_) {
    events_.emplace(event_data, event_data + len);
    events_size_ += len;
    return absl::OkStatus();
  }
  return absl::DataLossError(
      absl::StrCat("too many events: ", events_size_, " bytes accumulated"));
}

absl::Status NcclStatsAggregator::init() {
  /* Placeholder: To be implemented in future */
  return absl::OkStatus();
}

absl::StatusOr<NcclStatsConnectionStatistics*>
NcclStatsAggregator::addConnection(
    const ncclStatsConnectionIdentifier* connectionIdentifier) {
  {
    absl::MutexLock lock(&connections_mtx_);
    InitConstants(connectionIdentifier->nccl_plugin_name);
    std::unique_ptr<NcclStatsConnectionStatistics> connection_handle =
        std::make_unique<NcclStatsConnectionStatistics>(
            *this, *connectionIdentifier, distribution_collector_bitmap_,
            bw_burst_count_, bw_bucket_time_in_milliseconds_,
            carry_over_buckets_, baseline_latency_in_ns_,
            target_bw_per_nic_in_bits_per_sec_, bw_hist_collection_enabled_,
            variance_hist_collection_enabled_);
    NcclStatsConnectionStatistics* key = connection_handle.get();
    std::string nic_ip = connection_handle->getLocalIp();
    if (nic_bw_map_.find(nic_ip) == nic_bw_map_.end()) {
      allocateBWStats(*connectionIdentifier, connection_handle->getLocalIp());
    }
    connection_handlers_map_[key] = std::move(connection_handle);
    return key;
  }
  return absl::InternalError("Failed to add connection");
}

absl::StatusOr<NcclStatsConnectionStatistics*>
NcclStatsAggregator::addConnectionV4(
    const ncclStatsConnectionIdentifier_v4* connectionIdentifier) {
  {
    absl::MutexLock lock(&connections_mtx_);
    InitConstants(connectionIdentifier->nccl_plugin_name);
    std::unique_ptr<NcclStatsConnectionStatistics> connection_handle =
        std::make_unique<NcclStatsConnectionStatistics>(
            *this, *connectionIdentifier, distribution_collector_bitmap_,
            bw_burst_count_, bw_bucket_time_in_milliseconds_,
            carry_over_buckets_, baseline_latency_in_ns_,
            target_bw_per_nic_in_bits_per_sec_, bw_hist_collection_enabled_,
            variance_hist_collection_enabled_);
    NcclStatsConnectionStatistics* key = connection_handle.get();
    std::string nic_ip = connection_handle->getLocalIp();
    if (nic_bw_map_.find(nic_ip) == nic_bw_map_.end()) {
      allocateBWStats(*connectionIdentifier, connection_handle->getLocalIp());
    }
    connection_handlers_map_[key] = std::move(connection_handle);
    return key;
  }
  return absl::InternalError("Failed to add connection");
}

void NcclStatsAggregator::allocateBWStats(
    const ncclStatsConnectionIdentifier_v4& connectionIdentifier,
    const std::string nicIp) {
  if (bw_hist_collection_enabled_) {
    nic_bw_map_[nicIp] = std::make_unique<NicBWAggregationStats>();
    const params::BucketerParams bucketer_params_bandwidth =
        params::GetBucketerParamsForNicBWHistogram(nic_speed_);
    const double max_buckets_bandwidth = bucketer_params_bandwidth.max_buckets;
    const double scale_bandwidth = bucketer_params_bandwidth.scale;
    const double base_bandwidth = bucketer_params_bandwidth.base;
    NicBWAggregationStats* nic_bw_stats = nic_bw_map_[nicIp].get();
    nic_bw_stats->nccl_plugin_type = connectionIdentifier.nccl_plugin_type;
    nic_bw_stats->nccl_plugin_name = connectionIdentifier.nccl_plugin_name
                                         ? connectionIdentifier.nccl_plugin_name
                                         : "";
    nic_bw_stats->gpu_pci_addr = connectionIdentifier.gpu_pci_addr
                                     ? connectionIdentifier.gpu_pci_addr
                                     : "";
    nic_bw_stats->nic_ip = nicIp;
    nic_bw_stats->conn_type = connectionIdentifier.conn_type;

    if (isBWCollectionEnabledinBitMap(distribution_collector_bitmap_,
                                      SendLatencySW, SendMessageSize)) {
      allocateNicBwAggregationStats(nicIp, max_buckets_bandwidth,
                                    scale_bandwidth, base_bandwidth,
                                    NcclNicSend);
      if (variance_hist_collection_enabled_) {
        const params::BucketerParams bucketer_params_bandwidth_variance =
            params::GetBucketerParamsForNicBWVarianceHistogram(nic_speed_);
        const double max_buckets_bandwidth_variance =
            bucketer_params_bandwidth_variance.max_buckets;
        const double scale_bandwidth_variance =
            bucketer_params_bandwidth_variance.scale;
        const double base_bandwidth_variance =
            bucketer_params_bandwidth_variance.base;
        nic_bw_stats->nic_bw_variance_stats =
            std::make_unique<DistributionBucketer>(
                max_buckets_bandwidth_variance, scale_bandwidth_variance,
                base_bandwidth_variance);
        const params::BucketerParams bucketer_params_latency_variance =
            params::GetBucketerParamsForNicLatencyVarianceHistogram();
        const double max_buckets_latency_variance =
            bucketer_params_latency_variance.max_buckets;
        const double scale_latency_variance =
            bucketer_params_latency_variance.scale;
        const double base_latency_variance =
            bucketer_params_latency_variance.base;
        nic_bw_stats->nic_latency_variance_stats =
            std::make_unique<DistributionBucketer>(max_buckets_latency_variance,
                                                   scale_latency_variance,
                                                   base_latency_variance);
        nic_bw_stats->latency.assign(bw_burst_count_, 0);
      }
    }
    if (isBWCollectionEnabledinBitMap(distribution_collector_bitmap_,
                                      RecvLatencySW, RecvMessageSize)) {
      allocateNicBwAggregationStats(nicIp, max_buckets_bandwidth,
                                    scale_bandwidth, base_bandwidth,
                                    NcclNicRecv);
    }
    if (msec_bw_output_enabled_) {
      msec_bw_->AllocateBufferForNic(nicIp);
    }
  }
}

void NcclStatsAggregator::allocateBWStats(
    const ncclStatsConnectionIdentifier& connectionIdentifier,
    const std::string nicIp) {
  if (bw_hist_collection_enabled_) {
    nic_bw_map_[nicIp] = std::make_unique<NicBWAggregationStats>();
    const params::BucketerParams bucketer_params_bandwidth =
        params::GetBucketerParamsForNicBWHistogram(nic_speed_);
    const double max_buckets_bandwidth = bucketer_params_bandwidth.max_buckets;
    const double scale_bandwidth = bucketer_params_bandwidth.scale;
    const double base_bandwidth = bucketer_params_bandwidth.base;
    NicBWAggregationStats* nic_bw_stats = nic_bw_map_[nicIp].get();
    nic_bw_stats->nccl_plugin_type = connectionIdentifier.nccl_plugin_type;
    nic_bw_stats->nccl_plugin_name = connectionIdentifier.nccl_plugin_name
                                         ? connectionIdentifier.nccl_plugin_name
                                         : "";
    nic_bw_stats->gpu_pci_addr = connectionIdentifier.gpu_pci_addr
                                     ? connectionIdentifier.gpu_pci_addr
                                     : "";
    nic_bw_stats->nic_ip = nicIp;
    nic_bw_stats->conn_type = connectionIdentifier.conn_type;

    if (isBWCollectionEnabledinBitMap(distribution_collector_bitmap_,
                                      SendLatencySW, SendMessageSize)) {
      allocateNicBwAggregationStats(nicIp, max_buckets_bandwidth,
                                    scale_bandwidth, base_bandwidth,
                                    NcclNicSend);
      if (variance_hist_collection_enabled_) {
        const params::BucketerParams bucketer_params_bandwidth_variance =
            params::GetBucketerParamsForNicBWVarianceHistogram(nic_speed_);
        const double max_buckets_bandwidth_variance =
            bucketer_params_bandwidth_variance.max_buckets;
        const double scale_bandwidth_variance =
            bucketer_params_bandwidth_variance.scale;
        const double base_bandwidth_variance =
            bucketer_params_bandwidth_variance.base;
        nic_bw_stats->nic_bw_variance_stats =
            std::make_unique<DistributionBucketer>(
                max_buckets_bandwidth_variance, scale_bandwidth_variance,
                base_bandwidth_variance);
        const params::BucketerParams bucketer_params_latency_variance =
            params::GetBucketerParamsForNicLatencyVarianceHistogram();
        const double max_buckets_latency_variance =
            bucketer_params_latency_variance.max_buckets;
        const double scale_latency_variance =
            bucketer_params_latency_variance.scale;
        const double base_latency_variance =
            bucketer_params_latency_variance.base;
        nic_bw_stats->nic_latency_variance_stats =
            std::make_unique<DistributionBucketer>(max_buckets_latency_variance,
                                                   scale_latency_variance,
                                                   base_latency_variance);
        nic_bw_stats->latency.assign(bw_burst_count_, 0);
      }
    }
    if (isBWCollectionEnabledinBitMap(distribution_collector_bitmap_,
                                      RecvLatencySW, RecvMessageSize)) {
      allocateNicBwAggregationStats(nicIp, max_buckets_bandwidth,
                                    scale_bandwidth, base_bandwidth,
                                    NcclNicRecv);
    }
    if (msec_bw_output_enabled_) {
      msec_bw_->AllocateBufferForNic(nicIp);
    }
  }
}

void NcclStatsAggregator::allocateNicBwAggregationStats(const std::string nicIp,
                                                        double maxBucketBw,
                                                        double scaleBw,
                                                        double bucket_baseBw,
                                                        uint8_t direction) {
  NicBWAggregationStats* nic_bw_stats = nic_bw_map_[nicIp].get();
  nic_bw_stats->nic_bw[direction].assign(bw_burst_count_, 0);
  nic_bw_stats->offered_load[direction].assign(bw_burst_count_, 0);
  nic_bw_stats->nic_bw_stats[direction] =
      std::make_unique<DistributionBucketer>(maxBucketBw, scaleBw,
                                             bucket_baseBw);
}

void NcclStatsAggregator::aggregateNicStatsinHistogram(
    uint64_t unused_buckets) {
  for (const auto& nic_bw_stats : nic_bw_map_) {
    std::string nicIp = nic_bw_stats.first;
    auto& nic_distribution_bucketer = nic_bw_map_[nicIp]->nic_bw_stats;
    if (nic_distribution_bucketer[NcclNicSend] != nullptr) {
      calculateAndUpdateHistogram(NcclNicSend, nicIp, unused_buckets);
    }
    if (nic_distribution_bucketer[NcclNicRecv] != nullptr) {
      calculateAndUpdateHistogram(NcclNicRecv, nicIp, unused_buckets);
    }
  }
}

void NcclStatsAggregator::detectAndUpdateVarianceHistograms(
    std::string nicIp, double nic_bw_bits_per_sec,
    uint64_t offered_load_bytes_per_bucket, double actual_latency_in_ns) {
  auto& distribution_bw_variance_bucketer =
      nic_bw_map_[nicIp]->nic_bw_variance_stats;
  uint64_t expected_load_in_bits =
      (target_bw_per_nic_in_bits_per_sec_ * bw_bucket_time_in_milliseconds_) /
      SECOND_TO_MILLISECOND_MULTIPLIER;
  uint64_t offered_load_in_bits =
      (offered_load_bytes_per_bucket * BYTES_TO_BITS_MULTIPLIER);
  double bw_variance =
      ((offered_load_in_bits >= expected_load_in_bits) &&
       (target_bw_per_nic_in_bits_per_sec_ > nic_bw_bits_per_sec))
          ? (target_bw_per_nic_in_bits_per_sec_ - nic_bw_bits_per_sec)
          : 0;
  if (bw_variance > 0) {
    distribution_bw_variance_bucketer->Submit(bw_variance);
  }

  auto& distribution_latency_variance_bucketer =
      nic_bw_map_[nicIp]->nic_latency_variance_stats;
  /*
  To avoid max_bw_per_nic_in_bits_per_sec_ becoming 0 when converted to
  bits_per_ns, we multiply offered_load_in_bits with
  SECOND_TO_NANOSECOND_MULTIPLIER.
  */
  double queueing_delay_in_ns =
      (offered_load_in_bits * SECOND_TO_NANOSECOND_MULTIPLIER) /
      target_bw_per_nic_in_bits_per_sec_;
  double expected_latency_in_ns =
      queueing_delay_in_ns + baseline_latency_in_ns_;
  double latency_variance =
      (actual_latency_in_ns > expected_latency_in_ns)
          ? (actual_latency_in_ns - expected_latency_in_ns)
          : 0;
  if (latency_variance > 0) {
    distribution_latency_variance_bucketer->Submit(latency_variance);
  }
}

void NcclStatsAggregator::calculateAndUpdateHistogram(uint8_t direction,
                                                      std::string nicIp,
                                                      uint64_t unused_buckets) {
  NicBWAggregationStats* nic_bw_aggregation_stats = nic_bw_map_[nicIp].get();
  std::vector<double>& nic_bw_vec = nic_bw_aggregation_stats->nic_bw[direction];
  auto& distribution_bucketer =
      nic_bw_aggregation_stats->nic_bw_stats[direction];
  std::vector<uint64_t>& nic_offered_load_vec =
      nic_bw_aggregation_stats->offered_load[direction];
  std::vector<double>& latency_vec = nic_bw_aggregation_stats->latency;
  // Limiting bw_vec_size to skip processing index where carry over buckets
  // starts
  uint64_t bw_vec_size = getBwVecValidLimit(carry_over_buckets_, unused_buckets,
                                            nic_bw_vec.size());
  double carry_over_bytes = carry_over_bytes_from_last_generation_;
  for (uint64_t i = 0; i < bw_vec_size; i++) {
    int64_t bw_bucket_time_ms = bw_bucket_time_in_milliseconds_;
    double bytes_in_bucket = nic_bw_vec[i] + carry_over_bytes;
    if (bytes_in_bucket > max_bytes_per_bucket_) {
      carry_over_bytes = bytes_in_bucket - max_bytes_per_bucket_;
      nic_bw_vec[i] = max_bytes_per_bucket_;
    } else {
      carry_over_bytes = 0;
      nic_bw_vec[i] = bytes_in_bucket;
    }
    double bw =
        calculateBWPerBucketInBitsPerSec(nic_bw_vec[i], bw_bucket_time_ms);
    if (variance_hist_collection_enabled_) {
      if (direction == NcclNicSend) {
        detectAndUpdateVarianceHistograms(nicIp, bw, nic_offered_load_vec[i],
                                          latency_vec[i]);
      }
    }
    distribution_bucketer->Submit(bw);
  }
  if (msec_bw_output_enabled_) {
    // We copy over the stats in the range [0,bw_vec_size) because that's the
    // portion of the buffer which was actually updated this run.  It excludes
    // buckets which were unused (due to how much time had actually ellapsed)
    // and the portion which overlap the next time frame and will be copied
    // over.  The overlapping buckets will be copied over on the following call
    // when their calculation is fully complete.
    msec_bw_->CopyOverBandwidthStats(
        direction, nicIp,
        absl::Span<const double>(nic_bw_vec.data(), bw_vec_size));
  }
  carry_over_bytes_from_last_generation_ = carry_over_bytes;
  nic_bw_vec.assign(nic_bw_vec.size(), 0);
  nic_offered_load_vec.assign(nic_offered_load_vec.size(), 0);
}

absl::Status NcclStatsAggregator::deleteConnection(
    NcclStatsConnectionStatistics* statsConnectionHandle,
    ncclStatsConnectionCloseType closeType, const char* verboseReason) {
  log_function_(NCCL_LOG_INFO, NCCL_NET, __PRETTY_FUNCTION__, __LINE__,
                absl::StrFormat("Deleting connections. connectionHandler: %p "
                                "closeType: %d verboseReason: %s",
                                statsConnectionHandle, closeType, verboseReason)
                    .c_str());
  {
    absl::MutexLock lock(&connections_mtx_);
    DCHECK(connection_handlers_map_.find(statsConnectionHandle) !=
           connection_handlers_map_.end());
    to_be_deleted_list_.emplace_back(
        std::move(connection_handlers_map_[statsConnectionHandle]));
    connection_handlers_map_.erase(statsConnectionHandle);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<protoDist::AllStats>>
NcclStatsAggregator::readTelemetry() {
  std::unique_ptr<protoDist::AllStats> stats =
      std::make_unique<protoDist::AllStats>();
  {
    absl::MutexLock lock(&connections_mtx_);
    readNicStats(stats.get());
    for (const auto& connection_handler : connection_handlers_map_) {
      NcclStatsConnectionStatistics* connectionHandle =
          connection_handler.first;
      protoDist::ConnectionStats* conn_stats = stats->add_conn_stats();
      auto read_conn_stats_status =
          connectionHandle->readStatsFromLastInterval(conn_stats);
      if (!read_conn_stats_status.ok()) {
        std::string log_msg_info = absl::StrFormat(
            "Failed to read connection stats for connectionHandler: %lu",
            (uintptr_t)connectionHandle);
        addLogMsg(log_msg_info, NCCL_LOG_WARN, __PRETTY_FUNCTION__, __LINE__);
      }
    }
    /* Read the stats from Zombie connections, and delete all the connections
     * that were waiting to be released */
    while (!to_be_deleted_list_.empty()) {
      NcclStatsConnectionStatistics* connectionHandle =
          to_be_deleted_list_.front().get();
      protoDist::ConnectionStats* conn_stats = stats->add_conn_stats();
      auto read_conn_stats_status =
          connectionHandle->readStatsFromLastInterval(conn_stats);
      if (!read_conn_stats_status.ok()) {
        std::string log_msg_info = absl::StrFormat(
            "Failed to read connection stats for connectionHandler: %lu",
            (uintptr_t)connectionHandle);
        addLogMsg(log_msg_info, NCCL_LOG_WARN, __PRETTY_FUNCTION__, __LINE__);
      }
      to_be_deleted_list_.pop_front();
    }
  }
  {
    absl::MutexLock lock(&log_messages_mtx_);
    while (!log_messages_.empty()) {
      *stats->add_events_log() = std::move(log_messages_.front());
      log_messages_.pop();
    }
    log_messages_size_ = 0;
  }
  {
    // Create a local queue to hold the events for processing & limit critical
    // locked section.
    std::queue<std::vector<uint8_t>> local_events;
    {  // Minimized critical section
      absl::MutexLock lock(&events_mtx_);
      local_events.swap(events_);
      events_size_ = 0;
    }
    // process the data (unfortunately we need to deser it here)
    // possible refactoring: pass ser-ed bytes to the very end to avoid deser
    while (!local_events.empty()) {
      std::vector<uint8_t>& serialized_event = local_events.front();
      protoDist::Event* new_event = stats->add_events();
      if (!new_event->ParseFromArray(serialized_event.data(),
                                     serialized_event.size())) {
        addLogMsg(
            absl::StrFormat("[Telemetry warning] failed to deser heartbeat"),
            NCCL_LOG_WARN, __PRETTY_FUNCTION__, __LINE__);
        stats->mutable_events()->RemoveLast();
      }
      local_events.pop();
    }
  }
  return stats;
}

void NcclStatsAggregator::readNicStats(protoDist::AllStats* stats) {
  for (const auto& nic_distribution : nic_bw_map_) {
    std::string nicIp = nic_distribution.first;
    auto& nic_distribution_bucketer = (nic_distribution.second)->nic_bw_stats;
    protoDist::ConnectionStats* conn_stats = stats->add_conn_stats();
    readNicConnectionDetails(conn_stats, nicIp);
    if (nic_distribution_bucketer[NcclNicSend] != nullptr) {
      readNicDistributions(conn_stats, NcclNicSend, nicIp);
    }
    if (nic_distribution_bucketer[NcclNicRecv] != nullptr) {
      readNicDistributions(conn_stats, NcclNicRecv, nicIp);
    }
  }
}

void NcclStatsAggregator::assignNicNcclStatsConnection(
    protoDist::ncclStatsConnection& messageConnection, std::string nicIp) {
  NicBWAggregationStats* nic_bw_aggregation_stats = nic_bw_map_[nicIp].get();
  messageConnection.set_conn_type(
      protoDist::ncclStatsConnectionType(nic_bw_aggregation_stats->conn_type));
  switch (nic_bw_aggregation_stats->conn_type) {
    case ncclStatsConnectionType::EntityNVLConnection:
      break;
    case ncclStatsConnectionType::EntityPCIConnection:
      break;
    case ncclStatsConnectionType::EntityTCPConnection:
      messageConnection.mutable_tcp_conn()->set_local_endpoint(nicIp);
      break;
    case ncclStatsConnectionType::EntityRDMAConnection:
      messageConnection.mutable_rdma_conn()->set_local_endpoint(nicIp);
      break;
    case ncclStatsConnectionType::EntityProfilerPluginConnection:
      messageConnection.mutable_profiler_conn()->set_local_network_endpoint(
          nicIp);
      break;
  }
}

void NcclStatsAggregator::readNicConnectionDetails(
    protoDist::ConnectionStats* connStats, std::string nicIp) {
  NicBWAggregationStats* nic_bw_aggregation_stats = nic_bw_map_[nicIp].get();
  // Add connection identifier information to connStats.
  protoDist::ConnectionIdentifier* messageConnectionId =
      connStats->mutable_connection_id();
  populateNcclStatsConnectionIdentifier(
      nic_bw_aggregation_stats->nccl_plugin_type,
      nic_bw_aggregation_stats->nccl_plugin_name,
      nic_bw_aggregation_stats->gpu_pci_addr, /*gpuUuid=*/"",
      getPluginMajor(), getPluginMinor(), *messageConnectionId);

  // Add connection information to connStats.
  protoDist::ncclStatsConnection* messageConnection =
      messageConnectionId->mutable_connection();
  assignNicNcclStatsConnection(*messageConnection, nicIp);
  google::protobuf::Timestamp* timestamp = connStats->mutable_time_stamp();
  *timestamp = google::protobuf::util::TimeUtil::GetCurrentTime();
}
void NcclStatsAggregator::readNicDistributions(
    protoDist::ConnectionStats* connStats, uint8_t direction,
    std::string nicIp) {
  NicBWAggregationStats* nic_bw_aggregation_stats = nic_bw_map_[nicIp].get();
  auto& nic_distribution_bucketer = nic_bw_aggregation_stats->nic_bw_stats;
  std::vector<std::pair<DistributionBucketer*, protoDist::DistType>>
      potentialDistList;
  if (direction == NcclNicSend) {
    potentialDistList.push_back(
        std::make_pair(nic_distribution_bucketer[direction].get(),
                       protoDist::protoSendBandwidth));
    if (variance_hist_collection_enabled_) {
      potentialDistList.push_back(
          std::make_pair(nic_bw_aggregation_stats->nic_bw_variance_stats.get(),
                         protoDist::protoBandwidthVariance));
      potentialDistList.push_back(std::make_pair(
          nic_bw_aggregation_stats->nic_latency_variance_stats.get(),
          protoDist::protoLatencyVariance));
    }
  } else {
    potentialDistList.push_back(
        std::make_pair(nic_distribution_bucketer[direction].get(),
                       protoDist::protoRecvBandwidth));
  }
  for (auto dist : potentialDistList) {
    int _ = populateStatsDistributionWithDistributionBucketer(
        connStats, dist.first, dist.second);
    dist.first->Reset();
  }
}

absl::Status NcclStatsAggregator::processHighFrequencyTelemetry() {
  if (bw_hist_collection_enabled_) {
    absl::MutexLock lock(&connections_mtx_);
    // Calculate the buckets which were not touched in this generation
    uint64_t current_time_milliseconds =
        get_current_monotonic_time_milliseconds();
    if (msec_bw_output_enabled_) {
      msec_bw_->RegisterCurrentTimeInMilliseconds(current_time_milliseconds);
    }
    uint64_t bw_collection_duration_in_milliseconds =
        current_time_milliseconds - bw_collection_start_time_milliseconds_;
    uint64_t num_of_buckets_used = bw_collection_duration_in_milliseconds /
                                       bw_bucket_time_in_milliseconds_ +
                                   1;
    uint64_t unused_buckets = (num_of_buckets_used < bw_burst_count_)
                                  ? (bw_burst_count_ - num_of_buckets_used)
                                  : 0;
    // if bw_collection_start_time_ was not initialized by this function and has
    // default initialization, we should skip that generation.
    if (bw_collection_start_time_milliseconds_ == 0) {
      unused_buckets = bw_burst_count_;
    }
    // updating start time here ensures all connections are initialized same
    // start time, making aggregation easier.
    bw_collection_start_time_milliseconds_ =
        current_time_milliseconds - max_expected_latency_per_tx_milliseconds_;
    for (const auto& connection_handler : connection_handlers_map_) {
      NcclStatsConnectionStatistics* connectionHandle =
          connection_handler.first;
      connectionHandle->swapStateForBWstatsandInitializeTime(
          bw_collection_start_time_milliseconds_, unused_buckets);
    }
    for (const auto& connection_handler : connection_handlers_map_) {
      NcclStatsConnectionStatistics* connectionHandle =
          connection_handler.first;
      std::string nicIp = connectionHandle->getLocalIp();
      auto& nic_bw_aggregation_stats = nic_bw_map_[nicIp];
      for (uint8_t i = 0; i < TotalNicBWDirection; i++) {
        std::vector<double>& nicBwVec = nic_bw_aggregation_stats->nic_bw[i];
        std::vector<uint64_t>& nicOfferedLoadvec =
            nic_bw_aggregation_stats->offered_load[i];
        std::vector<double>& nicLatencyVec = nic_bw_aggregation_stats->latency;
        auto read_conn_stats_status =
            connectionHandle->aggregateBWStatsFromLastInterval(
                i, nicBwVec, nicOfferedLoadvec, nicLatencyVec, unused_buckets);
      }
    }
    if (variance_hist_collection_enabled_) {
      for (const auto& connection_handler : connection_handlers_map_) {
        NcclStatsConnectionStatistics* connectionHandle =
            connection_handler.first;
        std::string nicIp = connectionHandle->getLocalIp();
        auto& nic_bw_aggregation_stats = nic_bw_map_[nicIp];
        std::vector<uint64_t>& nicOfferedLoadvec =
            nic_bw_aggregation_stats->offered_load[NcclNicSend];
        connectionHandle->aggregateVarianceStats(nicOfferedLoadvec,
                                                 unused_buckets);
      }
    }
    aggregateNicStatsinHistogram(unused_buckets);
    zeroOutNicStats();
  }
  return absl::OkStatus();
}

void NcclStatsAggregator::zeroOutNicStats() {
  for (const auto& nic_bw_stats : nic_bw_map_) {
    for (uint8_t direction = 0; direction < NUM_BW_DIRECTION; direction++) {
      std::vector<double>& nic_bw_array =
          nic_bw_stats.second->nic_bw[direction];
      nic_bw_array.assign(nic_bw_array.size(), 0);
      std::vector<uint64_t>& offered_load_array =
          nic_bw_stats.second->offered_load[direction];
      offered_load_array.assign(offered_load_array.size(), 0);
    }
    std::vector<double>& latency_vec = nic_bw_stats.second->latency;
    latency_vec.assign(latency_vec.size(), 0);
  }
}

size_t NcclStatsAggregator::getNumberOfConnections() {
  absl::MutexLock lock(&connections_mtx_);
  return connection_handlers_map_.size();
}

absl::Status NcclStatsAggregator::clean() {
  if (connections_mtx_.TryLock()) {
    if (!connection_handlers_map_.empty()) {
      addLogMsg(
          "clean() called while connections are still open.  GPUViz may be "
          "being shut down prematurely.",
          NCCL_LOG_WARN, __PRETTY_FUNCTION__, __LINE__);
      // This should only happen inside a destructor, so we skip the usual part
      // where we make sure that the last bit of the statistics are collected.
      connection_handlers_map_.clear();
    }
    connections_mtx_.Unlock();
    return absl::OkStatus();
  } else {
    return absl::InternalError(
        "Failed to clean connections - mutex is locked at destruction time.");
  }
}

NcclStatsConnectionStatistics::NcclStatsConnectionStatistics(
    NcclStatsAggregator& aggregator,
    const ncclStatsConnectionIdentifier& connectionId,
    uint64_t distributionCollectorBitmap, const int64_t bwBurstCount,
    const int64_t bwBucketTime, const uint64_t carryOverBuckets,
    const double baselineLatencyInNs, const double targetBwPerNicInBitsPerSec,
    const bool bwHistCollectionEnabled,
    const bool varianceHistCollectionEnabled)
    : nccl_stats_aggregator_(aggregator),
      connection_id_(ConnectionIdentifier(connectionId)),
      generation_(0),
      generation_bw_(0),
      generation_for_reporter_read_(0),
      generation_bw_for_reporter_read_(0),
      bw_bucket_time_in_milliseconds_(bwBucketTime),
      carry_over_buckets_(carryOverBuckets),
      distribution_collector_bitmap_(distributionCollectorBitmap),
      baseline_latency_in_ns_(baselineLatencyInNs),
      target_bw_per_nic_in_bits_per_sec_(targetBwPerNicInBitsPerSec),
      bw_hist_collection_enabled_(bwHistCollectionEnabled),
      variance_hist_collection_enabled_(varianceHistCollectionEnabled) {
  /* Init 2 distribution map: 1 for reader and other for writer */
  for (int i = 0; i < NUM_STATES; i++) {
    initDistributionMap(distributionCollectorBitmap, i);
    initBWCollectionBuckets(bwBurstCount, i);
  }
}

NcclStatsConnectionStatistics::NcclStatsConnectionStatistics(
    NcclStatsAggregator& aggregator,
    const ncclStatsConnectionIdentifier_v4& connectionId,
    uint64_t distributionCollectorBitmap, const int64_t bwBurstCount,
    const int64_t bwBucketTime, const uint64_t carryOverBuckets,
    const double baselineLatencyInNs, const double targetBwPerNicInBitsPerSec,
    const bool bwHistCollectionEnabled,
    const bool varianceHistCollectionEnabled)
    : nccl_stats_aggregator_(aggregator),
      connection_id_(ConnectionIdentifier(connectionId)),
      generation_(0),
      generation_bw_(0),
      generation_for_reporter_read_(0),
      generation_bw_for_reporter_read_(0),
      bw_bucket_time_in_milliseconds_(bwBucketTime),
      carry_over_buckets_(carryOverBuckets),
      distribution_collector_bitmap_(distributionCollectorBitmap),
      baseline_latency_in_ns_(baselineLatencyInNs),
      target_bw_per_nic_in_bits_per_sec_(targetBwPerNicInBitsPerSec),
      bw_hist_collection_enabled_(bwHistCollectionEnabled),
      variance_hist_collection_enabled_(varianceHistCollectionEnabled) {
  /* Init 2 distribution map: 1 for reader and other for writer */
  for (int i = 0; i < NUM_STATES; i++) {
    initDistributionMap(distributionCollectorBitmap, i);
    initBWCollectionBuckets(bwBurstCount, i);
  }
}

NcclStatsConnectionStatistics::ConnectionIdentifier::ConnectionIdentifier(
    const ncclStatsConnectionIdentifier& connectionId)
    : conn_type(connectionId.conn_type),
      nccl_plugin_type(connectionId.nccl_plugin_type),
      nccl_plugin_name(
          connectionId.nccl_plugin_name ? connectionId.nccl_plugin_name : ""),
      gpu_pci_addr(connectionId.gpu_pci_addr ? connectionId.gpu_pci_addr : ""),
      connection(connectionId.connection) {
  std::pair<std::string, std::string> ip_port;
  switch (conn_type) {
    case ncclStatsConnectionType::EntityNVLConnection:
      break;
    case ncclStatsConnectionType::EntityPCIConnection:
      break;
    case ncclStatsConnectionType::EntityTCPConnection:
      ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
          connectionId.connection.tcp_conn.local_endpoint);
      local_nic_ip = ip_port.first;
      local_endpoint = gpuviz::utils::getEndpointString(ip_port);
      ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
          connectionId.connection.tcp_conn.remote_endpoint);
      remote_endpoint = gpuviz::utils::getEndpointString(ip_port);
      break;
    case ncclStatsConnectionType::EntityRDMAConnection:
      ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
          connectionId.connection.rdma_conn.local_endpoint);
      local_nic_ip = ip_port.first;
      local_endpoint = gpuviz::utils::getEndpointString(ip_port);
      ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
          connectionId.connection.rdma_conn.remote_endpoint);
      remote_endpoint = gpuviz::utils::getEndpointString(ip_port);
      break;
    case ncclStatsConnectionType::EntityProfilerPluginConnection:
      ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
          connectionId.connection.profiler_conn.local_endpoint);
      local_nic_ip = ip_port.first;
      local_endpoint =
          local_nic_ip + "|" +
          std::to_string(connectionId.connection.profiler_conn.local_rank);
      remote_endpoint = std::to_string(
          connectionId.connection.profiler_conn.remote_or_root_rank);
      collective_type =
          connectionId.connection.profiler_conn.collective_type
              ? std::string(
                    connectionId.connection.profiler_conn.collective_type)
              : std::string();
      collective_algorithm =
          connectionId.connection.profiler_conn.collective_algorithm
              ? std::string(
                    connectionId.connection.profiler_conn.collective_algorithm)
              : std::string();
      description =
          connectionId.connection.profiler_conn.description
              ? std::string(connectionId.connection.profiler_conn.description)
              : std::string();
      break;
  }
}

NcclStatsConnectionStatistics::ConnectionIdentifier::ConnectionIdentifier(
    const ncclStatsConnectionIdentifier_v4& connectionId)
    : conn_type(connectionId.conn_type),
      nccl_plugin_type(connectionId.nccl_plugin_type),
      nccl_plugin_name(
          connectionId.nccl_plugin_name ? connectionId.nccl_plugin_name : ""),
      gpu_pci_addr(connectionId.gpu_pci_addr ? connectionId.gpu_pci_addr : ""),
      gpu_uuid(connectionId.gpu_uuid ? connectionId.gpu_uuid : "") {
  std::pair<std::string, std::string> ip_port;
  if (conn_type == ncclStatsConnectionType::EntityProfilerPluginConnection) {
    this->connection.profiler_conn = connectionId.connection.profiler_conn.profiler_conn;
    this->is_nvl_telemetry = connectionId.connection.profiler_conn.is_nvl_telemetry;
    this->channel_id = connectionId.connection.profiler_conn.channel_id;

    ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
        connectionId.connection.profiler_conn.profiler_conn.local_endpoint);
    local_nic_ip = ip_port.first;
    local_endpoint =
        local_nic_ip + "|" +
        std::to_string(connectionId.connection.profiler_conn.profiler_conn.local_rank);
    remote_endpoint = std::to_string(
        connectionId.connection.profiler_conn.profiler_conn.remote_or_root_rank);
    collective_type =
        connectionId.connection.profiler_conn.profiler_conn.collective_type
            ? std::string(
                  connectionId.connection.profiler_conn.profiler_conn.collective_type)
            : std::string();
    collective_algorithm =
        connectionId.connection.profiler_conn.profiler_conn.collective_algorithm
            ? std::string(
                  connectionId.connection.profiler_conn.profiler_conn.collective_algorithm)
            : std::string();
    description =
        connectionId.connection.profiler_conn.profiler_conn.description
            ? std::string(connectionId.connection.profiler_conn.profiler_conn.description)
            : std::string();
  } else {
    // For other connection types, the union layouts are identical.
    std::memcpy(&this->connection, &connectionId.connection, sizeof(ncclStatsConnection));
    switch (conn_type) {
      case ncclStatsConnectionType::EntityNVLConnection:
        break;
      case ncclStatsConnectionType::EntityPCIConnection:
        break;
      case ncclStatsConnectionType::EntityTCPConnection:
        ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
            connectionId.connection.tcp_conn.local_endpoint);
        local_nic_ip = ip_port.first;
        local_endpoint = gpuviz::utils::getEndpointString(ip_port);
        ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
            connectionId.connection.tcp_conn.remote_endpoint);
        remote_endpoint = gpuviz::utils::getEndpointString(ip_port);
        break;
      case ncclStatsConnectionType::EntityRDMAConnection:
        ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
            connectionId.connection.rdma_conn.local_endpoint);
        local_nic_ip = ip_port.first;
        local_endpoint = gpuviz::utils::getEndpointString(ip_port);
        ip_port = gpuviz::utils::ConvertSockaddrStorageToString(
            connectionId.connection.rdma_conn.remote_endpoint);
        remote_endpoint = gpuviz::utils::getEndpointString(ip_port);
        break;
      case ncclStatsConnectionType::EntityProfilerPluginConnection:
        break;
    }
  }
}

void NcclStatsConnectionStatistics::initBWCollectionBuckets(
    const int64_t bwBurstCount, int state) {
  auto& bw_stats = bw_stats_[state];
  if (bw_hist_collection_enabled_) {
    for (int j = 0; j < NUM_BW_DIRECTION; j++) {
      std::vector<double>& bw_vec = bw_stats.bw[j];
      std::vector<uint64_t>& offered_load = bw_stats.offered_load[j];
      bw_vec.assign(bwBurstCount, 0);
      offered_load.assign(bwBurstCount, 0);
    }
    if (variance_hist_collection_enabled_) {
      bw_stats.latency.assign(bwBurstCount, 0);
    }
    bw_stats.start_time_milliseconds =
        0;  // Avoid uninitialized value if the updateBW call happens before the
            // first swap
  }
}
void NcclStatsConnectionStatistics::initDistributionMap(
    uint64_t distributionCollectorBitmap, int state) {
  const params::BucketerParams bucketer_params_latency =
      params::GetBucketerParamsForLatencyHistogram();
  const double max_buckets_latency = bucketer_params_latency.max_buckets;
  const double scale_latency = bucketer_params_latency.scale;
  const double base_latency = bucketer_params_latency.base;

  const params::BucketerParams bucketer_params_size =
      params::GetBucketerParamsForSizeHistogram();
  const double max_buckets_size = bucketer_params_size.max_buckets;
  const double scale_size = bucketer_params_size.scale;
  const double base_size = bucketer_params_size.base;

  const params::BucketerParams bucketer_params_bandwidth =
      params::GetBucketerParamsForBWHistogram(
          getAggregator()->GetNicSpeeds().first);
  const double max_buckets_bandwidth = bucketer_params_bandwidth.max_buckets;
  const double scale_bandwidth = bucketer_params_bandwidth.scale;
  const double base_bandwidth = bucketer_params_bandwidth.base;

  const params::BucketerParams bucketer_params_latency_variance =
      params::GetBucketerParamsForNicLatencyVarianceHistogram();
  const double max_buckets_latency_variance =
      bucketer_params_latency_variance.max_buckets;
  const double scale_latency_variance = bucketer_params_latency_variance.scale;
  const double base_latency_variance = bucketer_params_latency_variance.base;

  for (int j = 0; j < TotalDistributionType; j++) {
    if ((distributionCollectorBitmap & (1 << j))) {
      switch (1 << j) {
        case SendLatencySW:
        case RecvLatencySW:
        case SendLatencyNetHW:
        case RecvLatencyNetHW:
        case RecvReadyLatency:
          distribution_map_[state][j] = std::make_unique<DistributionBucketer>(
              max_buckets_latency, scale_latency, base_latency);
          break;
        case SendMessageSize:
        case RecvMessageSize:
          distribution_map_[state][j] = std::make_unique<DistributionBucketer>(
              max_buckets_size, scale_size, base_size);
          break;
        default:
          break;
      }
    }
  }
  if (bw_hist_collection_enabled_) {
    bool SendLatencySWDistCollectionEnabled =
        IS_SET(distributionCollectorBitmap, SendMessageSize);
    bool SendMessageSizeDistCollectionEnabled =
        IS_SET(distributionCollectorBitmap, SendMessageSize);
    if (SendLatencySWDistCollectionEnabled != 0 &&
        SendMessageSizeDistCollectionEnabled != 0) {
      distribution_map_[state][protoDist::protoSendBandwidth] =
          std::make_unique<DistributionBucketer>(
              max_buckets_bandwidth, scale_bandwidth, base_bandwidth);
      if (variance_hist_collection_enabled_) {
        distribution_map_[state][protoDist::protoLatencyVariance] =
            std::make_unique<DistributionBucketer>(max_buckets_latency_variance,
                                                   scale_latency_variance,
                                                   base_latency_variance);
      }
    }
    bool RecvLatencySWDistCollectionEnabled =
        IS_SET(distributionCollectorBitmap, RecvLatencySW);
    bool RecvMessageSizeDistCollectionEnabled =
        IS_SET(distributionCollectorBitmap, RecvMessageSize);
    if (RecvLatencySWDistCollectionEnabled != 0 &&
        RecvMessageSizeDistCollectionEnabled != 0) {
      distribution_map_[state][protoDist::protoRecvBandwidth] =
          std::make_unique<DistributionBucketer>(
              max_buckets_bandwidth, scale_bandwidth, base_bandwidth);
    }
  }
}

absl::Status NcclStatsConnectionStatistics::notifyOperationMeasurement(
    const ncclStatsOperationMetric* measurement) {
  {
    uint64_t reported_time_milliseconds =
        bw_hist_collection_enabled_
            ? gpuviz::utils::get_current_monotonic_time_milliseconds()
            : 0;
    absl::MutexLock lock(&mtx_);
    auto sizeDistributionType = getSizeDistributionType(measurement->type);
    if (!sizeDistributionType.ok()) {
      return sizeDistributionType.status();
    }
    uint64_t idx = getCurrentWriteDistributionMapIndex();
    uint64_t idx_bw = getCurrentBWWriteIndex();
    auto& distribution_map = distribution_map_[idx];
    distribution_map[sizeDistributionType.value()]->Submit(measurement->op_sz);
    for (uint32_t i = 0; i < measurement->num_measurements; i++) {
      auto latencyDistributionType = getLatencyDistributionType(
          measurement->type, measurement->measurements[i].latency_type);
      if (!latencyDistributionType.ok()) {
        return latencyDistributionType.status();
      }
      uint64_t distribution_type = latencyDistributionType.value();
      uint64_t latency_in_ns =
          measurement->measurements[i].latency_in_nanoseconds;
      distribution_map[distribution_type]->Submit(latency_in_ns);

      if (bw_hist_collection_enabled_) {
        uint64_t direction;
        switch (distribution_type) {
          case DISTRIBUTION_TYPE(SendLatencySW):
            direction = NcclSend;
            break;
          case DISTRIBUTION_TYPE(RecvLatencySW):
            direction = NcclRecv;
            break;
          default:
            direction = UINT64_MAX;
        }
        if (direction != UINT64_MAX) {
          updateBwAndLatencyStats(measurement->op_sz, latency_in_ns, direction,
                                  idx_bw, reported_time_milliseconds);
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status NcclStatsConnectionStatistics::notifyProfilerEvent(
    const uint8_t* event_data, size_t len) {
  return getAggregator()->addEvent(event_data, len);
}

/* This function can't run in parallel with itself at any time because there is
 * only one reader thread which is responsible for calling this function. */
absl::Status NcclStatsConnectionStatistics::readStatsFromLastInterval(
    protoDist::ConnectionStats* connStats) {
  uint64_t readBucketIdx = getReadBucketIdxAndSwapGeneration();
  auto connStatsStatus =
      getConnectionStatsProtoFromReadBucket(readBucketIdx, connStats);
  resetReadBucket(readBucketIdx);
  return connStatsStatus;
}

absl::Status NcclStatsConnectionStatistics::aggregateBWStatsFromLastInterval(
    uint8_t direction, std::vector<double>& nicBwVec,
    std::vector<uint64_t>& nicOfferedLoadVec,
    std::vector<double>& nicLatencyVec, uint64_t unused_buckets) {
  uint64_t state = getCurrentBWReadIndexUnlocked();
  aggregateBWStatsinHistogram(state, (BWDirection)direction, nicBwVec,
                              nicOfferedLoadVec, nicLatencyVec, unused_buckets);
  zeroOutBWReadStats(state);
  return absl::OkStatus();
}

NcclStatsAggregator* NcclStatsConnectionStatistics::getAggregator() const {
  return &nccl_stats_aggregator_;
}

std::string NcclStatsConnectionStatistics::getLocalIp() const {
  return connection_id_.local_nic_ip;
}

uint64_t NcclStatsConnectionStatistics::getCurrentWriteDistributionMapIndex() {
  return getCurrentWriteDistributionMapIndexUnlocked();
}

uint64_t
NcclStatsConnectionStatistics::getCurrentWriteDistributionMapIndexUnlocked() {
  return (generation_for_reporter_read_) % 2;
}

uint64_t NcclStatsConnectionStatistics::getReadBucketIdxAndSwapGeneration() {
  absl::MutexLock lock(&mtx_);
  uint64_t lastWriteIdx = getCurrentWriteDistributionMapIndex();
  /* Update generation so that writer now writes to the next bucket*/
  generation_++;
  generation_for_reporter_read_ = generation_;
  return lastWriteIdx;
}

void NcclStatsConnectionStatistics::resetReadBucket(uint64_t idx) {
  auto& distribution_map_read_bucket = distribution_map_[idx];
  for (int j = 0; j < protoDist::protoTotalDistributionType; j++) {
    if (distribution_map_read_bucket[j] != nullptr) {
      distribution_map_read_bucket[j]->Reset();
    }
  }
}

uint64_t NcclStatsConnectionStatistics::getCurrentBWWriteIndex() {
  return (generation_bw_) % NUM_STATES;
}

uint64_t NcclStatsConnectionStatistics::getCurrentBWWriteIndexUnlocked() {
  return (generation_bw_for_reporter_read_) % NUM_STATES;
}

uint64_t NcclStatsConnectionStatistics::getCurrentBWReadIndexUnlocked() {
  static_assert(NUM_STATES == 2,
                "NUM_STATES is changed and getCurrentBWReadIndex will not "
                "return correct index.");
  return (generation_bw_for_reporter_read_ + 1) % NUM_STATES;
}

void NcclStatsConnectionStatistics::swapStateForBWstatsandInitializeTime(
    uint64_t start_time_milliseconds, uint64_t unused_buckets) {
  absl::MutexLock lock(&mtx_);
  uint64_t old_idx = getCurrentBWWriteIndexUnlocked();
  generation_bw_++;
  generation_bw_for_reporter_read_ = generation_bw_;
  uint64_t new_idx = getCurrentBWWriteIndexUnlocked();
  bw_stats_[new_idx].start_time_milliseconds = start_time_milliseconds;
  // Copy the carry over buckets from old BW array to the new BW array
  for (uint64_t direction = 0; direction < TotalBWDirection; direction++) {
    auto& bw_vec_old = bw_stats_[old_idx].bw[direction];
    auto& bw_vec_new = bw_stats_[new_idx].bw[direction];
    uint64_t start_idx_old_vec = getBwVecValidLimit(
        carry_over_buckets_, unused_buckets, bw_vec_old.size());
    // only copy till carry_over_buckets_
    for (uint64_t i = 0; i < carry_over_buckets_; i++) {
      bw_vec_new[i] = bw_vec_old[i + start_idx_old_vec];
    }
  }
}
void NcclStatsConnectionStatistics::aggregateVarianceStats(
    const std::vector<uint64_t>& nicOfferedLoadVec, uint64_t unused_buckets) {
  uint64_t state = getCurrentBWReadIndexUnlocked();
  uint64_t hist_write_idx = getCurrentWriteDistributionMapIndexUnlocked();
  uint64_t end_idx = getBwVecValidLimit(carry_over_buckets_, unused_buckets,
                                        nicOfferedLoadVec.size());
  for (uint64_t i = 0; i < end_idx; i++) {
    uint64_t offered_load_in_bits =
        nicOfferedLoadVec[i] * BYTES_TO_BITS_MULTIPLIER;
    double actual_latency_in_ns = bw_stats_[state].latency[i];
    double queueing_delay_in_ns =
        (offered_load_in_bits * SECOND_TO_NANOSECOND_MULTIPLIER) /
        target_bw_per_nic_in_bits_per_sec_;
    double expected_latency_in_ns =
        queueing_delay_in_ns + baseline_latency_in_ns_;
    double latency_variance =
        (actual_latency_in_ns > expected_latency_in_ns)
            ? (actual_latency_in_ns - expected_latency_in_ns)
            : 0;
    if (latency_variance > 0) {
      distribution_map_[hist_write_idx][protoDist::protoLatencyVariance]
          ->Submit(latency_variance);
    }
  }
  zeroOutLatencyStats(state);
}
// In addition to collecting BW stats, this function also collects latency at
// NIC level by comparing latency of transactions from all connections that
// begin transmitting data simultaneously.
void NcclStatsConnectionStatistics::aggregateBWStatsinHistogram(
    uint64_t state, BWDirection direction, std::vector<double>& nicBwVec,
    std::vector<uint64_t>& nicOfferedLoadVec,
    std::vector<double>& nicLatencyVec, uint64_t unused_buckets) {
  absl::StatusOr<uint64_t> bwDistributionType =
      getBWDistributionType(direction);
  if (bwDistributionType.ok()) {
    uint64_t write_idx = getCurrentWriteDistributionMapIndexUnlocked();
    auto& distribution_map = distribution_map_[write_idx];
    auto& bw_vec = bw_stats_[state].bw[direction];
    auto& offered_load_vec = bw_stats_[state].offered_load[direction];
    auto& latency_vec = bw_stats_[state].latency;
    // Limiting bw_vec_size to skip processing index where carry over buckets
    // starts
    uint64_t end_idx =
        getBwVecValidLimit(carry_over_buckets_, unused_buckets, bw_vec.size());
    for (uint64_t i = 0; i < end_idx; i++) {
      nicBwVec[i] += bw_vec[i];
      nicOfferedLoadVec[i] += offered_load_vec[i];
      if (variance_hist_collection_enabled_) {
        if (direction == NcclSend) {
          nicLatencyVec[i] = std::max(nicLatencyVec[i], latency_vec[i]);
        }
      }
      double bw = calculateBWPerBucketInBitsPerSec(
          bw_vec[i], bw_bucket_time_in_milliseconds_);
      distribution_map[bwDistributionType.value()]->Submit(bw);
    }
  }
}

void NcclStatsConnectionStatistics::zeroOutBWReadStats(uint64_t state) {
  for (int j = 0; j < NUM_BW_DIRECTION; j++) {
    std::vector<uint64_t>& offered_load = bw_stats_[state].offered_load[j];
    std::vector<double>& bw_vec = bw_stats_[state].bw[j];
    offered_load.assign(offered_load.size(), 0);
    bw_vec.assign(bw_vec.size(), 0);
  }
}

void NcclStatsConnectionStatistics::zeroOutLatencyStats(uint64_t state) {
  std::vector<double>& latency_vec = bw_stats_[state].latency;
  latency_vec.assign(latency_vec.size(), 0);
}

absl::StatusOr<uint64_t> NcclStatsConnectionStatistics::getSizeDistributionType(
    ncclStatsOpType op_type) {
  switch (op_type) {
    case OperationTypeChunkSend:
      return DISTRIBUTION_TYPE(SendMessageSize);
    case OperationTypeChunkRecv:
      return DISTRIBUTION_TYPE(RecvMessageSize);
    default:
      return absl::InvalidArgumentError("Provided op_type is invalid");
  }
}

absl::StatusOr<uint64_t> NcclStatsConnectionStatistics::getBWDistributionType(
    BWDirection direction) {
  switch (direction) {
    case NcclSend:
      return protoDist::protoSendBandwidth;
    case NcclRecv:
      return protoDist::protoRecvBandwidth;
    default:
      return absl::InvalidArgumentError("Provided op_type is invalid");
  }
}

absl::StatusOr<uint64_t>
NcclStatsConnectionStatistics::getLatencyDistributionType(
    ncclStatsOpType op_type, ncclStatsLatencyType latency_type) {
  switch (op_type) {
    case OperationTypeChunkSend:
      switch (latency_type) {
        case LatencySoftware:
          return DISTRIBUTION_TYPE(SendLatencySW);
        case LatencyNetHW:
          return DISTRIBUTION_TYPE(SendLatencyNetHW);
        default:
          return absl::InvalidArgumentError(
              "Provided op_type and latency_type is invalid");
      }
    case OperationTypeChunkRecv:
      switch (latency_type) {
        case LatencySoftware:
          return DISTRIBUTION_TYPE(RecvLatencySW);
        case LatencyNetHW:
          return DISTRIBUTION_TYPE(RecvLatencyNetHW);
        case LatencyRecvReady:
          return DISTRIBUTION_TYPE(RecvReadyLatency);
        default:
          return absl::InvalidArgumentError(
              "Provided op_type and latency_type is invalid");
      }
    default:
      return absl::InvalidArgumentError(
          "Provided op_type and latency_type is invalid");
  }
}

void NcclStatsConnectionStatistics::updateBwAndLatencyStats(
    uint64_t sz, uint64_t latency_in_ns, uint64_t direction, uint64_t state,
    uint64_t reported_time_milliseconds) {
  BwStats& bw_stats = bw_stats_[state];
  if (bw_stats.start_time_milliseconds == 0) {
    // Avoid collecting BW and latency numbers for the first high-frequency
    // reporting period, as the workload is likely to be still starting up.
    return;
  }
  std::vector<double>& bw_vec = bw_stats.bw[direction];
  std::vector<uint64_t>& offered_load = bw_stats.offered_load[direction];
  // latency is collected in double because the latency histogram and latency
  // variance histogram are both in double, and requires conversion from
  // uint64_t, the type in which we receive latency from NCCL plugin.
  std::vector<double>& latency = bw_stats.latency;
  int64_t time_elapsed =
      (reported_time_milliseconds - bw_stats.start_time_milliseconds);
  int64_t latency_in_ms =
      (int64_t)(latency_in_ns / MILLISECOND_TO_NANOSECOND_MULTIPLIER);
  // Using int64_t instead of uint64_t because (time_elapsed - latency_in_ms)
  // might be negative, and uint64_t will make it positive, leading to incorrect
  // result from max() below.
  uint64_t start_idx =
      std::max((time_elapsed - latency_in_ms) / bw_bucket_time_in_milliseconds_,
               (int64_t)0);
  uint64_t end_idx =
      std::min((uint64_t)((time_elapsed / bw_bucket_time_in_milliseconds_) + 1),
               (uint64_t)bw_vec.size());
  uint64_t num_buckets = end_idx - start_idx;
  if (start_idx >= end_idx) {
    // Possible if reported time is greater than expected end of interval
    bw_vec[bw_vec.size() - 1] += double(sz);
    return;
  }
  double bytes_per_bucket = double(sz) / double(num_buckets);
  for (uint64_t i = start_idx; i < end_idx; i++) {
    bw_vec[i] += bytes_per_bucket;
    offered_load[i] += sz;
  }
  if (variance_hist_collection_enabled_) {
    if (direction == NcclSend) {
      latency[start_idx] = std::max(latency[start_idx], (double)latency_in_ns);
    }
  }
}

void NcclStatsConnectionStatistics::assignNcclStatsRDMAConnection(
    protoDist::ncclStatsRDMAConnection& messageRDMAconn) {
  messageRDMAconn.set_local_endpoint(connection_id_.local_endpoint);
  messageRDMAconn.set_remote_endpoint(connection_id_.remote_endpoint);
  messageRDMAconn.set_local_qpn(connection_id_.connection.rdma_conn.local_qpn);
  messageRDMAconn.set_remote_qpn(
      connection_id_.connection.rdma_conn.remote_qpn);
}

void NcclStatsConnectionStatistics::assignNcclStatsTCPConnection(
    protoDist::ncclStatsTCPConnection& messageTCPconn) {
  messageTCPconn.set_local_endpoint(connection_id_.local_endpoint);
  messageTCPconn.set_remote_endpoint(connection_id_.remote_endpoint);
}

void NcclStatsConnectionStatistics::assignNcclStatsProfilerConnection(
    protoDist::ncclStatsProfilerPluginConnection& messageProfilerConn) {
  // Format: local IP-address|rank
  messageProfilerConn.set_local_network_endpoint(connection_id_.local_endpoint);
  messageProfilerConn.set_local_comm_hash(
      connection_id_.connection.profiler_conn.local_comm_hash);
  messageProfilerConn.set_local_rank(
      connection_id_.connection.profiler_conn.local_rank);
  // Remote rank number for P2P telemetry (proxy op level or p2p operation)
  // If statistics are for a collective operation, identifies the root rank for
  // the operation.
  messageProfilerConn.set_remote_or_root_rank(
      connection_id_.connection.profiler_conn.remote_or_root_rank);
  messageProfilerConn.set_collective_type(connection_id_.collective_type);
  messageProfilerConn.set_collective_algorithm(
      connection_id_.collective_algorithm);
  messageProfilerConn.set_description(connection_id_.description);
  messageProfilerConn.set_is_nvl_telemetry(connection_id_.is_nvl_telemetry);
  messageProfilerConn.set_channel_id(connection_id_.channel_id);
}

void NcclStatsConnectionStatistics::assignNcclStatsConnection(
    protoDist::ncclStatsConnection& messageConnection) {
  messageConnection.set_conn_type(
      protoDist::ncclStatsConnectionType(connection_id_.conn_type));
  switch (connection_id_.conn_type) {
    case ncclStatsConnectionType::EntityNVLConnection:
      break;
    case ncclStatsConnectionType::EntityPCIConnection:
      break;
    case ncclStatsConnectionType::EntityTCPConnection:
      assignNcclStatsTCPConnection(*messageConnection.mutable_tcp_conn());
      break;
    case ncclStatsConnectionType::EntityRDMAConnection:
      assignNcclStatsRDMAConnection(*messageConnection.mutable_rdma_conn());
      break;
    case ncclStatsConnectionType::EntityProfilerPluginConnection:
      assignNcclStatsProfilerConnection(
          *messageConnection.mutable_profiler_conn());
      break;
  }
}

absl::Status
NcclStatsConnectionStatistics::getConnectionStatsProtoFromReadBucket(
    uint64_t idx, protoDist::ConnectionStats* connStats) {
  auto& distribution_map_read_bucket = distribution_map_[idx];

  // Add connection identifier information to connStats.
  protoDist::ConnectionIdentifier* messageConnectionId =
      connStats->mutable_connection_id();
  populateNcclStatsConnectionIdentifier(
      connection_id_.nccl_plugin_type, connection_id_.nccl_plugin_name,
      connection_id_.gpu_pci_addr, connection_id_.gpu_uuid,
      nccl_stats_aggregator_.getPluginMajor(),
      nccl_stats_aggregator_.getPluginMinor(), *messageConnectionId);

  // Add connection information to connStats.
  protoDist::ncclStatsConnection* messageConnection =
      messageConnectionId->mutable_connection();
  assignNcclStatsConnection(*messageConnection);

  // Add timestamp to connStats.
  google::protobuf::Timestamp* timestamp = connStats->mutable_time_stamp();
  *timestamp = google::protobuf::util::TimeUtil::GetCurrentTime();

  // Convert distribution_map_ to StatsDistribution and add to connection.
  int msg_size_hist_index = -1, lat_histogram_index = -1;
  for (int i = 0; i < protoDist::protoTotalDistributionType; i++) {
    if (distribution_map_read_bucket[i] != nullptr) {
      DistributionBucketer* dist_bucketer =
          distribution_map_read_bucket[i].get();
      protoDist::DistType dist_type = (protoDist::DistType)i;
      const int hist_index = populateStatsDistributionWithDistributionBucketer(
          connStats, dist_bucketer, dist_type);
      if (dist_type == protoDist::protoSendMessageSize) {
        msg_size_hist_index = hist_index;
      }
      if (dist_type == protoDist::protoSendLatencySW) {
        lat_histogram_index = hist_index;
      }
    }
  }
  // post-processing:
  // take pair of min max msg sizes from msg size histogram and ingest into
  // latency histogram f
  if (msg_size_hist_index != -1 && lat_histogram_index != -1) {
    protoDist::StatsDistribution* lat_histogram =
        connStats->mutable_histogram(lat_histogram_index);
    lat_histogram->set_min_msg_size_bucket(
        connStats->histogram(msg_size_hist_index).min());
    lat_histogram->set_max_msg_size_bucket(
        connStats->histogram(msg_size_hist_index).max());
  }

  return absl::OkStatus();
}

}  // namespace gpuviz

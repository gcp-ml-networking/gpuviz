/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_TELEMETRY_AGGREGATOR_H_
#define NCCL_TELEMETRY_AGGREGATOR_H_

#include <ctime>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "distribution_bucketer.h"
#include "millisecond_bandwidth_output.h"
#include "nccl_stats.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "telemetry_provider.h"

#define NUM_STATES 2        // 1 for reader 1 for writer
#define NUM_BW_DIRECTION 2  // Send and recv latency at Software Layer

namespace gpuviz {

class NcclStatsAggregator;
/* NcclStatsConnectionStatistics is responsible for aggregating statistics in
 * respective histograms for each NCCL connection being monitored by GPUViz. */
/*
   Locking Mechanism to avoid data race between writer thread updating the
   distribution(notifyOperationMeasurement) and reader thread reading the
   distribution(readStatsFromLastInterval).

   1. There are 2 buckets(NUM_STATES). One for reader and other for writer.
   2. Current index for writer thread is determined by the current value of
   generation_(getCurrentWriteDistributionMapIndex).
   3. generation_ is increment everytime by reader thread before it starts
   reading from the current index where writer was
   updating(getReadBucketIdxAndSwapGeneration).
   4. generation_ is incremented by reader thread within a lock to avoid writer
   fetching stale generation_(getReadBucketIdxAndSwapGeneration).
   5. writer always take a lock before fetching the current index(which is based
   on generation_) and does entire update to that index within the
   lock(notifyOperationMeasurement).
*/
class NcclStatsConnectionStatistics {
 public:
  NcclStatsConnectionStatistics(
      NcclStatsAggregator& aggregator,
      const ncclStatsConnectionIdentifier& connectionId,
      uint64_t distributionCollectorBitmap, const int64_t bwBurstCount,
      const int64_t bwBucketTime, const uint64_t carryOverBuckets,
      const double baselineLatencyInNs, const double targetBwPerNicInBitsPerSec,
      const bool bwHistCollectionEnabled,
      const bool varianceHistCollectionEnabled);
  NcclStatsConnectionStatistics(
      NcclStatsAggregator& aggregator,
      const ncclStatsConnectionIdentifier_v4& connectionId,
      uint64_t distributionCollectorBitmap, const int64_t bwBurstCount,
      const int64_t bwBucketTime, const uint64_t carryOverBuckets,
      const double baselineLatencyInNs, const double targetBwPerNicInBitsPerSec,
      const bool bwHistCollectionEnabled,
      const bool varianceHistCollectionEnabled);

  absl::Status notifyOperationMeasurement(
      const ncclStatsOperationMetric* measurement);
  absl::Status notifyProfilerEvent(const uint8_t* event_data, size_t len);

  absl::Status readStatsFromLastInterval(protoDist::ConnectionStats* connStats);
  void aggregateVarianceStats(const std::vector<uint64_t>& nicOfferedLoadVec,
                              uint64_t unused_buckets);
  absl::Status aggregateBWStatsFromLastInterval(
      uint8_t direction, std::vector<double>& nicBwVec,
      std::vector<uint64_t>& nicOfferedLoadVec,
      std::vector<double>& nicLatencyVec, uint64_t unused_buckets);
  void swapStateForBWstatsandInitializeTime(uint64_t start_time_milliseconds,
                                            uint64_t unused_buckets);

  NcclStatsAggregator* getAggregator() const;
  std::string getLocalIp() const;

 private:
  typedef enum {
    NcclSend = 0,
    NcclRecv = 1,
    TotalBWDirection = 2,
  } BWDirection;
  void initBWCollectionBuckets(const int64_t bwBurstCount, int state);
  void initDistributionMap(uint64_t distributionCollectorBitmap, int state);
  uint64_t getCurrentWriteDistributionMapIndex()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mtx_);
  // This function is supposed to be called only from single reader thread. This
  // function is same as getCurrentWriteDistributionMapIndex but doesn't require
  // mutex because it reads generation_ which is updated by the same thread.
  uint64_t getCurrentWriteDistributionMapIndexUnlocked();
  uint64_t getReadBucketIdxAndSwapGeneration();
  absl::Status getConnectionStatsProtoFromReadBucket(
      uint64_t idx, protoDist::ConnectionStats* connStats);
  void resetReadBucket(uint64_t idx);
  uint64_t getCurrentBWWriteIndex() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mtx_);
  uint64_t getCurrentBWWriteIndexUnlocked();
  uint64_t getCurrentBWReadIndexUnlocked();
  void aggregateBWStatsinHistogram(uint64_t state, BWDirection direction,
                                   std::vector<double>& nicBwVec,
                                   std::vector<uint64_t>& nicOfferedLoadVec,
                                   std::vector<double>& nicLatencyVec,
                                   uint64_t unused_buckets);
  void zeroOutBWReadStats(uint64_t state);
  void zeroOutLatencyStats(uint64_t state);
  absl::StatusOr<uint64_t> getSizeDistributionType(ncclStatsOpType op_type);
  absl::StatusOr<uint64_t> getBWDistributionType(BWDirection op_type);
  absl::StatusOr<uint64_t> getLatencyDistributionType(
      ncclStatsOpType op_type, ncclStatsLatencyType latency_type);
  void updateBwAndLatencyStats(uint64_t sz, uint64_t latency_in_ns,
                               uint64_t direction, uint64_t state,
                               uint64_t reported_time_milliseconds);
  // aggregator lifetime is guaranteed to exceed all of the connections created
  // for this aggregator
  NcclStatsAggregator& nccl_stats_aggregator_;

  // This private struct would store only the fields we need from
  // ncclStatsConnectionIdentifier, and it convert the C-style char* into
  // std::string to hold a deep copy of the chars.

  struct ConnectionIdentifier {
    ConnectionIdentifier(const ncclStatsConnectionIdentifier& connectionId);
    ConnectionIdentifier(const ncclStatsConnectionIdentifier_v4& connectionId);
    ncclStatsConnectionType conn_type;
    ncclStatsPluginType nccl_plugin_type;
    std::string nccl_plugin_name;
    std::string gpu_pci_addr;
    std::string gpu_uuid;
    ncclStatsConnection connection;
    bool is_nvl_telemetry = false;
    uint32_t channel_id = 0;
    std::string local_nic_ip;
    std::string remote_endpoint;
    std::string local_endpoint;
    std::string collective_type;
    std::string collective_algorithm;
    std::string description;
  };
  const ConnectionIdentifier connection_id_;
  uint64_t generation_ ABSL_GUARDED_BY(mtx_);
  uint64_t generation_bw_ ABSL_GUARDED_BY(mtx_);
  // The following variables are exact mirrors of the above variables, but are
  // used only by the reporter thread.
  //
  // This is to avoid the reporter thread having to take a lock to fetch the
  // current index without breaking ABSL annotation.
  uint64_t generation_for_reporter_read_;
  uint64_t generation_bw_for_reporter_read_;
  const int64_t bw_bucket_time_in_milliseconds_;
  const uint64_t carry_over_buckets_;
  const uint64_t distribution_collector_bitmap_;
  const double baseline_latency_in_ns_;
  const double target_bw_per_nic_in_bits_per_sec_;
  absl::Mutex mtx_;
  const bool bw_hist_collection_enabled_;
  const bool variance_hist_collection_enabled_;
  // 1 bucket for writer thread to write to and other for reader thread to read
  // from.
  //
  // Note that for all of the "NUM_STATES" arrays, they are accessed by
  // potentially multiple threads, but each array element is guaranteed to be
  // accessed only by one of the threads (writer or reader). The writer thread
  // will hold the "mtx_" lock for the duration of updating the relevant entries
  // to prevent switching while updating, but we cannot add annotation for that
  // as ABSL is not versitile enough to indicate "I have lock for writers, with
  // a switching data copy for the readers to look at."
  //
  // This "non-traditional" locking is done to prevent lock contention between
  // the slow reporting operations (which are "read" from the telemetry
  // in-memory storage perspective) and the hot-path operation notification
  // operations (which are "write" from the in-memory storage perspective).
  std::unique_ptr<DistributionBucketer>
      distribution_map_[NUM_STATES][protoDist::protoTotalDistributionType];

  struct BwStats {
    uint64_t start_time_milliseconds;
    std::vector<double> bw[NUM_BW_DIRECTION];
    std::vector<uint64_t> offered_load[NUM_BW_DIRECTION];
    std::vector<double> latency;
  };
  BwStats bw_stats_[NUM_STATES];

  void assignNcclStatsRDMAConnection(
      protoDist::ncclStatsRDMAConnection& messageRDMAconn);
  void assignNcclStatsTCPConnection(
      protoDist::ncclStatsTCPConnection& messageTCPconn);
  void assignNcclStatsConnection(
      protoDist::ncclStatsConnection& messageConnection);
  void assignNcclStatsProfilerConnection(
      protoDist::ncclStatsProfilerPluginConnection& messageProfilerConn);
};

/* NcclStatsAggregator is responsible for aggregating statistics in respective
 * histograms for all NCCL connections being monitored by GPUViz. Additionally,
 * it is responsible for reading and reporting the telemetry to exporter.*/
class NcclStatsAggregator : public TelemetryProvider {
 public:
  NcclStatsAggregator(ncclDebugLogger_t logFunction,
                      uint64_t distributionCollectorBitmap,
                      gpuviz::MillisecondBandwidthOutput* msec_bw = nullptr);
  ~NcclStatsAggregator();
  absl::Status init();
  absl::StatusOr<NcclStatsConnectionStatistics*> addConnection(
      const ncclStatsConnectionIdentifier* connectionIdentifier);
  absl::StatusOr<NcclStatsConnectionStatistics*> addConnectionV4(
      const ncclStatsConnectionIdentifier_v4* connectionIdentifier);
  absl::Status deleteConnection(
      NcclStatsConnectionStatistics* statsConnectionHandle,
      ncclStatsConnectionCloseType closeType, const char* verboseReason);
  absl::StatusOr<std::unique_ptr<protoDist::AllStats>> readTelemetry() override;
  absl::Status processHighFrequencyTelemetry() override;
  size_t getNumberOfConnections();

  void setPluginVersion(int32_t major, int32_t minor) {
    plugin_major_ = major;
    plugin_minor_ = minor;
  }

  int32_t getPluginMajor() const { return plugin_major_; }
  int32_t getPluginMinor() const { return plugin_minor_; }
  void addLogMsg(absl::string_view message, ncclDebugLogLevel level,
                 const char* pretty_func, uint32_t line,
                 ncclDebugLogSubSys subsys, std::string category_name,
                 int32_t category, int32_t msg_id);

  void addLogMsg(absl::string_view message, ncclDebugLogLevel level,
                 const char* pretty_func, uint32_t line) {
    addLogMsg(message, level, pretty_func, line, NCCL_NET, "", 0, 0);
  }

  // nic_speed_ and empr_nic_speed_ are in Gbps
  std::pair<double, double> GetNicSpeeds() {
    return std::pair(nic_speed_, empr_nic_speed_);
  }
  absl::Status addEvent(const uint8_t* event_data, size_t len);

 private:
  // Common header for any absl log coming from GPUViz.
  inline static constexpr absl::string_view kLogPrefix = "TELEMETRY/GPUViz: ";

  // All of the warning/error messages coming from GPUViz should not be fatal to
  // workload. To print as much info as we want to allow debugging, we still
  // want to keep some GPUViz related logging severity as NCCL_WARN, even though
  // the error messages do not harm workload. NCCL_WARN, however, means errors
  // happen in NCCL operation, which could cause confusion for customers. We
  // added this heading to logging message to avoid confusion for customer.
  inline static constexpr absl::string_view kNonFatalStrForNcclWarn =
      "[non-fatal] ";
  absl::Mutex connections_mtx_;
  std::unordered_map<NcclStatsConnectionStatistics*,
                     std::unique_ptr<NcclStatsConnectionStatistics>>
      connection_handlers_map_ ABSL_GUARDED_BY(connections_mtx_);
  std::list<std::unique_ptr<NcclStatsConnectionStatistics>> to_be_deleted_list_
      ABSL_GUARDED_BY(connections_mtx_);
  uint64_t bw_collection_start_time_milliseconds_
      ABSL_GUARDED_BY(connections_mtx_);
  // Includes the size of actual bw burst  + jitter
  uint64_t bw_burst_count_;
  const int64_t bw_bucket_time_in_milliseconds_;
  uint64_t max_expected_latency_per_tx_milliseconds_;
  double target_bw_per_nic_in_bits_per_sec_;
  double max_bytes_per_bucket_;
  double carry_over_bytes_from_last_generation_ = 0;
  uint64_t carry_over_buckets_;
  typedef enum {
    NcclNicSend = 0,
    NcclNicRecv = 1,
    TotalNicBWDirection = 2,
  } NicBWDirection;
  typedef struct NicBWAggregationStats {
    ncclStatsPluginType nccl_plugin_type;
    std::string nccl_plugin_name;
    std::string gpu_pci_addr;
    std::string nic_ip;
    ncclStatsConnectionType conn_type;
    // No need to maintain 2 states for these data structures because they are
    // accessed only by the reader thread today.
    std::vector<double> nic_bw[NUM_BW_DIRECTION];
    std::vector<uint64_t> offered_load[NUM_BW_DIRECTION];
    std::vector<double> latency;
    std::unique_ptr<DistributionBucketer> nic_bw_stats[NUM_BW_DIRECTION];
    std::unique_ptr<DistributionBucketer> nic_bw_variance_stats;
    std::unique_ptr<DistributionBucketer> nic_latency_variance_stats;
  } NicBWAggregationStats;
  std::unordered_map<std::string, std::unique_ptr<NicBWAggregationStats>>
      nic_bw_map_ ABSL_GUARDED_BY(connections_mtx_);
  absl::Mutex log_messages_mtx_;
  size_t log_messages_size_ = 0;
  std::queue<protoDist::LogEntry> log_messages_
      ABSL_GUARDED_BY(log_messages_mtx_);
  // Profiler Plugin marshaled events
  absl::Mutex events_mtx_;
  size_t events_size_ = 0;
  std::queue<std::vector<uint8_t>> events_ ABSL_GUARDED_BY(events_mtx_);

  const size_t non_aggregated_data_size_cap_;

  ncclDebugLogger_t log_function_;
  uint64_t distribution_collector_bitmap_;
  const bool bw_hist_collection_enabled_;
  double baseline_latency_in_ns_;
  const bool variance_hist_collection_enabled_;
  const bool msec_bw_output_enabled_;
  gpuviz::MillisecondBandwidthOutput* msec_bw_;
  int32_t plugin_major_ = 0;
  int32_t plugin_minor_ = 0;
  void allocateBWStats(
      const ncclStatsConnectionIdentifier& connectionIdentifier,
      const std::string nicIp) ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void allocateBWStats(
      const ncclStatsConnectionIdentifier_v4& connectionIdentifier,
      const std::string nicIp) ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void allocateNicBwAggregationStats(const std::string nicIp,
                                     double maxBucketBw, double scaleBw,
                                     double bucket_base, uint8_t direction)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void aggregateNicStatsinHistogram(uint64_t unused_buckets)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void detectAndUpdateVarianceHistograms(std::string nicIp,
                                         double nic_bw_bits_per_sec,
                                         uint64_t offered_load_bytes_per_bucket,
                                         double actual_latency_in_ns)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void calculateAndUpdateHistogram(uint8_t direction, std::string nicIp,
                                   uint64_t unused_buckets)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void readNicStats(protoDist::AllStats* stats)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void readNicConnectionDetails(protoDist::ConnectionStats* connStats,
                                std::string nicIp)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void readNicDistributions(protoDist::ConnectionStats* connStats,
                            uint8_t direction, std::string nicIp)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void assignNicNcclStatsConnection(
      protoDist::ncclStatsConnection& messageConnection, std::string nicIp)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  void zeroOutNicStats() ABSL_EXCLUSIVE_LOCKS_REQUIRED(connections_mtx_);
  absl::Status clean();

  // nic_speed_ and empr_nic_speed_ are in Gbps
  double nic_speed_ = 0;
  double empr_nic_speed_ = 0;
  bool nic_speed_init_ = 0;

  void InitConstants(absl::string_view nccl_plugin_name);
};

}  // namespace gpuviz

#endif  // NCCL_TELEMETRY_AGGREGATOR_H_
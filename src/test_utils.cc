// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "absl/random/distributions.h"
#include "absl/random/random.h"

using namespace ::protoDist;

namespace {
// Below are the local helper functions for generating AllStats.

// Fills StatsDistribution with random generated count of bucket.
void GenerateStatsDist(StatsDistribution* stats_dist, DistType dist_type,
                       int num_buckets = 0) {
  absl::BitGen gen_;
  stats_dist->set_dist_type(dist_type);
  stats_dist->set_min(100);
  stats_dist->set_max(1000);
  stats_dist->set_avg(500);
  stats_dist->set_num_samples(absl::Uniform<int>(gen_, 1, 10000));
  stats_dist->set_scale_factor(1);
  stats_dist->set_base(2);
  stats_dist->set_max_bucket(1000);

  if (num_buckets == 0) {
    num_buckets =
        ceil(log(stats_dist->max_bucket() / stats_dist->scale_factor()) /
             log(stats_dist->base())) +
        1;
  }
  for (int i = 0; i < num_buckets; ++i) {
    stats_dist->add_bucket_counts(absl::Uniform<int>(gen_, 0, 1000));
  }
}

void GenerateConnectionId(ConnectionIdentifier* conn_id, std::string pci_addr) {
  conn_id->set_nccl_plugin_type(ncclStatsPluginType::NetPlugin);
  conn_id->set_nccl_plugin_name("nccl-plugin-net");
  conn_id->set_gpu_pci_addr(std::move(pci_addr));
  conn_id->mutable_connection()->set_conn_type(
      ncclStatsConnectionType::EntityTCPConnection);
  conn_id->mutable_connection()->mutable_tcp_conn()->set_local_endpoint(
      "10.138.0.10:8888");
  conn_id->mutable_connection()->mutable_tcp_conn()->set_remote_endpoint(
      "10.138.0.11:8888");
}

// Fills ConnectionStats with connection id and StatsDistribution of types
void GenerateConnectionStats(ConnectionStats* conn_stats, std::string pci_addr,
                             const std::vector<DistType>& types,
                             int num_buckets = 0) {
  ConnectionIdentifier* conn_id = conn_stats->mutable_connection_id();
  GenerateConnectionId(conn_id, std::move(pci_addr));

  google::protobuf::Timestamp* timestamp = conn_stats->mutable_time_stamp();
  timestamp->set_seconds(absl::ToUnixSeconds(absl::Now()));

  for (const auto& type : types) {
    StatsDistribution* stats_dist = conn_stats->add_histogram();
    GenerateStatsDist(stats_dist, type, num_buckets);
  }
}

void GenerateLogEntry(LogEntry* log_entry, std::string pci_addr) {
  GenerateConnectionId(log_entry->mutable_connection_id(), std::move(pci_addr));
  google::protobuf::Timestamp* timestamp = log_entry->mutable_time_stamp();
  timestamp->set_seconds(absl::ToUnixSeconds(absl::Now()));
  log_entry->set_entry_category(100);
  log_entry->set_entry_message_id(1000);
  log_entry->set_category_name("test");
  log_entry->set_event_message("test message");
}

}  // namespace

namespace gpuviz {
namespace utils {
AllStats GenerateAllStats(const std::vector<DistType>& types,
                          const int num_stats, const int num_logs,
                          int num_buckets) {
  AllStats all_stats;
  absl::BitGen gen_;
  for (int i = 0; i < num_stats; ++i) {
    ConnectionStats* conn_stats = all_stats.add_conn_stats();
    std::string pci_addr =
        absl::StrFormat("0000:%d:00.0", absl::Uniform<int>(gen_, 1, 99));
    GenerateConnectionStats(conn_stats, std::move(pci_addr), types,
                            num_buckets);
  }

  for (int i = 0; i < num_logs; ++i) {
    LogEntry* log_entry = all_stats.add_events_log();
    std::string pci_addr =
        absl::StrFormat("0000:%d:00.0", absl::Uniform<int>(gen_, 1, 99));
    GenerateLogEntry(log_entry, std::move(pci_addr));
  }
  return all_stats;
}

void DeleteFilesMatchingRegex(std::string path_directory,
                              std::string file_pattern) {
  std::filesystem::path directory(std::move(path_directory));
  std::regex pattern(std::move(file_pattern));
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() &&
        std::regex_match(entry.path().filename().string(), pattern)) {
      std::filesystem::remove(entry.path());
    }
  }
}

bool NonEmptyFilesMatchingRegexExists(std::string path_directory,
                                      std::string file_pattern) {
  std::filesystem::path directory(std::move(path_directory));
  std::regex pattern(std::move(file_pattern));
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.file_size() > 0 &&
        std::regex_match(entry.path().filename().string(), pattern)) {
      return true;
    }
  }
  return false;
}

}  // namespace utils
}  // namespace gpuviz
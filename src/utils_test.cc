// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "utils.h"

#include <cmath>

#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "test_utils.h"

using namespace ::gpuviz::utils;
using namespace ::testing;
using namespace ::protoDist;

namespace {

// Tests all histograms within the AllStats proto has the right dist_type.
void HistogramMatcher(std::unique_ptr<AllStats>& msg, const int num_conn_stats,
                      const std::unordered_set<DistType>& filter) {
  EXPECT_EQ(msg->conn_stats_size(), num_conn_stats);
  for (const ConnectionStats& conn_stats : msg->conn_stats()) {
    EXPECT_EQ(conn_stats.histogram_size(), filter.size());
    for (const StatsDistribution& histogram : conn_stats.histogram()) {
      EXPECT_TRUE(filter.find(histogram.dist_type()) != filter.end());
    }
  }
}

TEST(UtilsTest, CanFilterNcclStatsHistogram) {
  auto msg = std::make_unique<AllStats>(GenerateAllStats(
      {DistType::protoBandwidthVariance, DistType::protoSendLatencySW,
       DistType::protoSendMessageSize, DistType::protoRecvMessageSize},
      10, 0));
  std::unordered_set<DistType> filter = {DistType::protoBandwidthVariance,
                                         DistType::protoRecvMessageSize};
  filterHistogramsByDistType(msg, filter);
  HistogramMatcher(msg, 10, filter);
}

TEST(UtilsTest, CanFilterOutEverything) {
  auto msg = std::make_unique<AllStats>(GenerateAllStats(
      {DistType::protoBandwidthVariance, DistType::protoSendLatencySW,
       DistType::protoSendMessageSize, DistType::protoRecvMessageSize},
      10, 0));
  std::unordered_set<DistType> filter = {};
  filterHistogramsByDistType(msg, filter);
  HistogramMatcher(msg, 10, filter);
}

TEST(UtilsTest, CanIncludeEverything) {
  auto msg = std::make_unique<AllStats>(GenerateAllStats(
      {DistType::protoBandwidthVariance, DistType::protoSendLatencySW,
       DistType::protoSendMessageSize, DistType::protoRecvMessageSize},
      10, 0));
  std::unordered_set<DistType> filter = {
      DistType::protoBandwidthVariance, DistType::protoSendLatencySW,
      DistType::protoSendMessageSize, DistType::protoRecvMessageSize};
  filterHistogramsByDistType(msg, filter);
  HistogramMatcher(msg, 10, filter);
}

TEST(UtilsTest, GetProcessIdentifier) {
  auto collectedPid = getProcessIdentifier();
  EXPECT_THAT(collectedPid, StartsWith(std::to_string(getpid())));
  std::cout << "Got PID:" << collectedPid << "\n";
}

}  // namespace

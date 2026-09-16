// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "distribution_bucketer.h"

#include <string.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace gpuviz {
namespace {

TEST(DistributionBucketer, DistributionBucketerTest) {
  const double kScale = 10.0;
  const double kMaxBucket = 50.0;
  const double kBase = 1.2;
  DistributionBucketer bucketer(kMaxBucket, kScale, kBase);

  // In order to have a bucketer that can have a top bucket upper limit
  // of >= 5, the following buckets will be generated:
  // 1.2^0: 1 * kScale
  // 1.2^1: 1.2 * kScale
  // 1.2^2: 1.44 * kScale
  // 1.2^3: 1.728 * kScale
  // 1.2^4: 2.0736 * kScale
  // 1.2^5: 2.48832 * kScale
  // 1.2^6: 2.985984 * kScale
  // 1.2^7: 3.5831808 * kScale
  // 1.2^8: 4.29981696 * kScale
  // 1.2^9: 5.159780352 * kScale

  bucketer.Submit(0.5 * kScale);         // Bucket 0
  bucketer.Submit(1.1 * kScale);         // Bucket 1
  bucketer.Submit(1.1 * kScale);         // Bucket 1
  bucketer.Submit(3.384 * kScale);       // Bucket 7
  bucketer.Submit(4.28 * kScale);        // Bucket 8
  bucketer.Submit(4.281 * kScale);       // Bucket 8
  bucketer.Submit(9999999999 * kScale);  // Bucket last

  EXPECT_THAT(bucketer.GetBucketCounts(),
              testing::ElementsAre(1, 2, 0, 0, 0, 0, 0, 1, 2, 1));
  EXPECT_EQ(bucketer.GetBucketCounts().size(), 10);

  std::vector<double> bucket_bounds = bucketer.GetBucketBounds();
  EXPECT_EQ(bucket_bounds.back(), pow(1.2, bucket_bounds.size() - 1) * kScale);
}
TEST(DistributionBucketer, DistributionBucketerTestReset) {
  const double kScale = 10.0;
  const double kMaxBucket = 50.0;
  const double kBase = 1.2;
  DistributionBucketer bucketer(kMaxBucket, kScale, kBase);
  EXPECT_FALSE(bucketer.GetMin().has_value());
  bucketer.Submit(100);
  bucketer.Submit(200);
  EXPECT_TRUE(bucketer.GetMin().has_value());
  EXPECT_TRUE(bucketer.GetMax().has_value());
  EXPECT_EQ(bucketer.GetMin().value(), 100);
  EXPECT_EQ(bucketer.GetMax().value(), 200);
  EXPECT_EQ(bucketer.GetAvg().value(), 150);
  EXPECT_THAT(bucketer.GetBucketCounts(),
              testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 2));

  bucketer.Reset();

  EXPECT_FALSE(bucketer.GetMin().has_value());
  bucketer.Submit(5);
  bucketer.Submit(10);
  EXPECT_TRUE(bucketer.GetMin().has_value());
  EXPECT_EQ(bucketer.GetMin().value(), 5);
  EXPECT_TRUE(bucketer.GetMax().has_value());
  EXPECT_EQ(bucketer.GetMax().value(), 10);
  EXPECT_EQ(bucketer.GetAvg().value(), 7.5);
  EXPECT_THAT(bucketer.GetBucketCounts(),
              testing::ElementsAre(1, 1, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(DistributionBucketer, DistributionBucketerMinMaxTest) {
  const double kScale = 10.0;
  const double kMaxBucket = 500.0;
  const double kBase = 1.2;
  DistributionBucketer bucketer(kMaxBucket, kScale, kBase);

  EXPECT_FALSE(bucketer.GetMin().has_value());
  EXPECT_FALSE(bucketer.GetMax().has_value());
  bucketer.Submit(100);

  EXPECT_TRUE(bucketer.GetMin().has_value());
  EXPECT_TRUE(bucketer.GetMax().has_value());

  EXPECT_EQ(bucketer.GetMin().value(), 100);
  EXPECT_EQ(bucketer.GetMax().value(), 100);

  bucketer.Submit(-3);
  bucketer.Submit(105);

  EXPECT_EQ(bucketer.GetMin().value(), -3);
  EXPECT_EQ(bucketer.GetMax().value(), 105);
}

TEST(DistributionBucketer, DistributionBucketerAvgTest) {
  const double kScale = 10.0;
  const double kMaxBucket = 50.0;
  const double kBase = 1.2;
  DistributionBucketer bucketer(kMaxBucket, kScale, kBase);
  EXPECT_FALSE(bucketer.GetAvg().has_value());
  bucketer.Submit(100);
  EXPECT_EQ(bucketer.GetAvg().value(), 100);
  bucketer.Submit(-1);
  EXPECT_EQ(bucketer.GetAvg().value(), 49.5);
  bucketer.Submit(105);
  EXPECT_EQ(bucketer.GetAvg().value(), 68);
}

TEST(CeilingButIntegersGainOne, RoundUp) {
  EXPECT_EQ(CeilingButIntegersGainOne(0.5), 1.0);
  EXPECT_EQ(CeilingButIntegersGainOne(-0.5), 0.0);
  EXPECT_EQ(CeilingButIntegersGainOne(12.99999999), 13.0);
  EXPECT_EQ(CeilingButIntegersGainOne(372.000000001), 373.0);
}

TEST(CeilingButIntegersGainOne, IntegersGainOne) {
  EXPECT_EQ(CeilingButIntegersGainOne(0.0), 1.0);
  EXPECT_EQ(CeilingButIntegersGainOne(-1.0), 0.0);
  EXPECT_EQ(CeilingButIntegersGainOne(13), 14.0);
  EXPECT_EQ(CeilingButIntegersGainOne(372), 373.0);
}

}  // namespace
}  // namespace gpuviz
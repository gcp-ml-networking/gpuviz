// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "distribution_bucketer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"

namespace gpuviz {

using namespace std;
DistributionBucketer::DistributionBucketer(double max_bucket, double scale,
                                           double base)
    : max_bucket_(max_bucket),
      scale_(scale),
      base_(base),
      one_over_log_of_base_(1 / log(base)),
      buckets_(ceil(log(max_bucket / scale) * one_over_log_of_base_) + 1) {
  for (size_t i = 0; i < buckets_.size(); ++i) {
    buckets_[i].upper_bound = std::pow(base, i) * scale;
  }
}

std::vector<int64_t> DistributionBucketer::GetBucketCounts() {
  std::vector<int64_t> tmp;
  tmp.reserve(buckets_.size());
  for (auto& bucket : buckets_) {
    tmp.push_back(bucket.count);
  }
  return tmp;
}

std::vector<double> DistributionBucketer::GetBucketBounds() {
  std::vector<double> tmp;
  tmp.reserve(buckets_.size());
  for (auto& bucket : buckets_) {
    tmp.push_back(bucket.upper_bound);
  }
  return tmp;
}

void DistributionBucketer::Submit(double sample) {
  // To get the bucket, we reverse the process which defines the bucket upper
  // bound (see the DistributionBucketer constructor) by dividing the sample
  // by the scale and then taking the log base base_ of the result.
  // We then round to the appropriate bucket using CeilingButIntegersGainOne
  // and put a minimum of 0, since that's the smallest bucket and covers the
  // region of [0..base_).
  size_t index = max(0.0, CeilingButIntegersGainOne(log(sample / scale_) *
                                                    one_over_log_of_base_));
  if (index >= buckets_.size()) index = buckets_.size() - 1;
  buckets_[index].count++;
  sum_ += sample;
  ++count_;
  if (sample < min_) min_ = sample;
  if (sample > max_) max_ = sample;
}

std::optional<double> DistributionBucketer::GetMin() {
  if (count_ > 0) return min_;
  return std::nullopt;
}

std::optional<double> DistributionBucketer::GetMax() {
  if (count_ > 0) return max_;
  return std::nullopt;
}

std::optional<double> DistributionBucketer::GetAvg() {
  if (count_ > 0) return sum_ / count_;
  return std::nullopt;
}

double DistributionBucketer::GetMaxBucket() { return max_bucket_; }

double DistributionBucketer::GetScale() { return scale_; }

double DistributionBucketer::GetBase() { return base_; }

void DistributionBucketer::Reset() {
  sum_ = 0;
  min_ = std::numeric_limits<double>::max();
  max_ = std::numeric_limits<double>::lowest();
  count_ = 0;
  for (size_t i = 0; i < buckets_.size(); ++i) {
    buckets_[i].count = 0;
  }
}

bool DistributionBucketer::GetSampleSubmitted() { return count_ > 0; }

std::string DistributionBucketer::Dump() {
  std::string ret;

  double coalesced_lower_bound = 0.0;
  double coalesced_upper_bound = 0.0;
  double prev_upper_bound = 0.0;
  bool coalesced = false;

  auto min = GetMin();
  if (min.has_value()) {
    absl::StrAppend(&ret, "Min: ", min.value(), "\n");
  }

  auto max = GetMax();
  if (max.has_value()) {
    absl::StrAppend(&ret, "Max: ", max.value(), "\n");
  }

  auto avg = GetAvg();
  if (avg.has_value()) {
    absl::StrAppend(&ret, "Avg: ", avg.value(), "\n");
  }

  for (size_t i = 0; i < buckets_.size(); ++i) {
    const auto count = buckets_[i].count;

    if (!count) {
      if (!coalesced) {
        coalesced_lower_bound = prev_upper_bound;
      }
      coalesced_upper_bound = buckets_[i].upper_bound;
      coalesced = true;
      continue;
    }

    if (coalesced) {
      absl::StrAppend(&ret, "Bucket ", coalesced_lower_bound, "-",
                      coalesced_upper_bound, " Count: 0\n");
      prev_upper_bound = coalesced_upper_bound;
    }

    coalesced = false;

    absl::StrAppend(&ret, "Bucket ", prev_upper_bound, "-",
                    buckets_[i].upper_bound, " Count: ", count, "\n");
    prev_upper_bound = buckets_[i].upper_bound;
  }

  if (coalesced) {
    absl::StrAppend(&ret, "Bucket ", coalesced_lower_bound, "-",
                    coalesced_upper_bound, " Count: 0\n");
  }

  return ret;
}

}  // namespace gpuviz

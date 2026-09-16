/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef DISTRIBUTION_BUCKETER_H_
#define DISTRIBUTION_BUCKETER_H_

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace gpuviz {

/*DistributionBucketer functions are not reentrant. The caller is responsible
for calling the functions in thread safe manner.*/
class DistributionBucketer {
 public:
  explicit DistributionBucketer(double max_bucket, double scale, double base);
  std::vector<int64_t> GetBucketCounts();
  std::vector<double> GetBucketBounds();
  void Submit(double sample);
  std::string Dump();
  std::optional<double> GetMin();
  std::optional<double> GetMax();
  // Returns a coarse-grained average of all values submitted previously
  std::optional<double> GetAvg();
  bool GetSampleSubmitted();
  void Reset();
  double GetMaxBucket();
  double GetScale();
  double GetBase();

 private:
  struct bucket {
    double upper_bound;
    int64_t count = 0;
  };
  int64_t count_ = 0;
  double sum_ = 0;
  double min_ = std::numeric_limits<double>::max();
  double max_ = std::numeric_limits<double>::lowest();
  double max_bucket_;
  const double scale_;
  const double base_;

  // 1/log(base_) is stored for the sake of being able to do efficient
  // calculations
  const double one_over_log_of_base_;
  std::vector<bucket> buckets_;
};

// This function should take a floating point number, round to
// the nearest whole number above it, but, if that round doesn't
// change the value, it should add 1.  This is to deal with
// the fact that we're calculating bucket bounds.  Anything
// one step under a given bucket bound should go into that
// bucket, but the bound is meant to be exclusive, so an
// exact match for a bucket should go into the bucket above.
inline double CeilingButIntegersGainOne(double num) { return floor(num + 1.0); }

}  // namespace gpuviz

#endif  // DISTRIBUTION_BUCKETER_H_

/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef UTILS_H_
#define UTILS_H_
#include <arpa/inet.h>

#include <memory>
#include <queue>
#include <string>

#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "src/proto/nccl_telemetry_proto.pb.h"

#define SECOND_TO_MILLISECOND_MULTIPLIER 1000         // 1,000 = 1e+3
#define MILLISECOND_TO_NANOSECOND_MULTIPLIER 1000000  // 1,000,000 = 1e+6
#define SECOND_TO_MICROSECOND_MULTIPLIER 1000000      // 1,000,000 = 1e+6
#define SECOND_TO_NANOSECOND_MULTIPLIER 1000000000    // 1,000,000,000 = 1e+9
#define MICROSECOND_TO_NANOSECOND_MULTIPLIER 1000     // 1,000 = 1e+3

namespace gpuviz {
namespace utils {
// CLOCK_MONOTONIC represents the absolute elapsed wall-clock time since some
// arbitrary, fixed point in the past and is usually recommended to compute the
// elapsed time between two events observed on the one machine. All our usecase
// which involves calculating time interval should call this function.
absl::Time get_current_monotonic_time();
// Same as the above clock fetching routine, but avoids the expensive
// timestamp->absl::Time conversion.
uint64_t get_current_monotonic_time_milliseconds();
std::pair<std::string, std::string> ConvertSockaddrStorageToString(
    const sockaddr_storage& sockaddr_storage);
std::string getEndpointString(std::pair<std::string, std::string> ip_port);
void filterHistogramsByDistType(
    std::unique_ptr<protoDist::AllStats>& msg,
    const std::unordered_set<protoDist::DistType>& dist_types);
absl::StatusOr<std::string> getMdsAttributeByKey(absl::string_view key);

// Fetches a unique identifier for this process in this container
// (PID:process_start_time) and returns it as a string
std::string getProcessIdentifier();
}  // namespace utils
}  // namespace gpuviz

#endif  // UTILS_H_

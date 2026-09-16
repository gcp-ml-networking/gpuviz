/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include "src/proto/nccl_telemetry_proto.pb.h"

namespace gpuviz {
namespace utils {
// Creates AllStats proto.
protoDist::AllStats GenerateAllStats(
    const std::vector<protoDist::DistType>& types, const int num_stats,
    const int num_logs, int num_buckets = 0);

// Removes all files under directory with regex pattern.
void DeleteFilesMatchingRegex(std::string path_directory,
                              std::string file_pattern);

// Returns if any non empty files matching a certain pattern exists.
bool NonEmptyFilesMatchingRegexExists(std::string path_directory,
                                      std::string file_pattern);
}  // namespace utils
}  // namespace gpuviz
#endif  // TEST_UTILS_H_
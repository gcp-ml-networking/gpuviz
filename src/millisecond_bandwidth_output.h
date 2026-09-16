/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef MILLISECOND_BANDWIDTH_OUTPUT
#define MILLISECOND_BANDWIDTH_OUTPUT

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "params.h"
#include "src/proto/millisecond_bw_file_format.pb.h"

namespace gpuviz {

// Class to handle outputting the millisecond bandwidth stats.
// Class is not threadsafe.
class MillisecondBandwidthOutput {
 public:
  MillisecondBandwidthOutput();
  ~MillisecondBandwidthOutput();
  void AllocateBufferForNic(std::string nic_ip);
  // Method which should be called before copying over the
  // bandwidth stats to establish the current time.
  void RegisterCurrentTimeInMilliseconds(uint64_t current_time) {
    if (time_diff_ == 0) {
      time_diff_ = absl::ToUnixMillis(absl::Now()) - current_time;
    }
    current_time_ = time_diff_ + current_time;
    batch_times_set_ = false;
    current_batch_ = next_batch_;
  };
  // Copy given stats into buffer.  It is required that
  // `RegisterCurrentTimeInMilliseconds` be called before
  // calling this method.
  void CopyOverBandwidthStats(uint8_t direction, std::string nic_ip,
                              absl::Span<const double> data);
  // Save buffer out to output file.
  // `include_incomplete` should be set if the call is
  // the final call to SaveBandwidthStats due to things
  // closing down.  Normally, the final incomplete batch
  // will not be output until later when it has been
  // completed.  Setting include_incomplete to true will
  // cause the incomplete batch (with trailing zeros) to
  // be output along with the complete ones.
  void SaveBandwidthStats(bool include_incomplete = false);
  // Save final data and shut the file down.
  // Also disables the class just in case there
  // are any calls which happen later.
  void CloseOutBandwidthStats();

 private:
  void RotateLogs();
  void ZeroOutNicData(Batch& batch);
  bool enabled_;
  uint64_t current_time_;
  uint64_t num_bw_msec_batches_in_buffer_;
  uint64_t batch_size_in_msec_;
  uint64_t batch_size_in_samples_;
  uint64_t bucket_size_in_msec_;
  std::vector<Batch> batches_;
  // Convenience data structure to provide O(1) access to the NicData
  // object given knowledge of the NIC IP address and batch number.
  // The map is indexed by IP address and the vector by batch number.
  std::unordered_map<std::string, std::vector<NicData*>>
      ip_address_to_nic_data_map_;
  int32_t current_batch_ = 0;
  int32_t next_batch_ = 0;
  bool batch_times_set_ = false;
  std::string pid_ = std::to_string(getpid());
  std::string file_name_prefix_ = "";
  int32_t num_rotations_ = 0;
  std::ofstream out_;
  absl::Time last_rotation_time_ = absl::Now();
  absl::Duration file_rot_interval_ = absl::Seconds(
      gpuviz::params::GetMillisecondBandwidthOutputRotationRate());
  int64_t time_diff_ = 0;
};

}  // namespace gpuviz

#endif  // MILLISECOND_BANDWIDTH_OUTPUT
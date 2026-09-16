// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "millisecond_bandwidth_output.h"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/util/time_util.h"
#include "params.h"
#include "src/proto/millisecond_bw_file_format.pb.h"

namespace gpuviz {

MillisecondBandwidthOutput::MillisecondBandwidthOutput() {
  enabled_ = params::IsMillisecondBandwidthOutputEnabled();
  // To calculate how large the buffer for the millisecond bandwith output
  // should be, we need to know how long the interval between writes will be
  // and how many milliseconds are in each batch and then we add some extra
  // space to be on the safe side.
  int64_t msec_between_file_output =
      params::GetMillisecondBandwidthOutputIntervalInMilliSeconds();
  num_bw_msec_batches_in_buffer_ =
      msec_between_file_output /
      params::GetMillisecondBandwidthOutputBatchSize() * 1.5;
  batch_size_in_msec_ = params::GetMillisecondBandwidthOutputBatchSize();
  bucket_size_in_msec_ = params::GetBandwidthBucketTimeInMilliSeconds();
  batch_size_in_samples_ = batch_size_in_msec_ / bucket_size_in_msec_;
  if (enabled_) {
    batches_ = std::vector<Batch>(num_bw_msec_batches_in_buffer_);
  } else {
    // To minimize memory usage if we're disabled.
    batches_ = std::vector<Batch>();
  }

  // Instantiate file name with time when the run starts
  std::string time_str =
      absl::FormatTime("%Y-%m-%d_%H:%M:%S", absl::Now(), absl::UTCTimeZone());
  file_name_prefix_ = params::GetDirectoryPathForGPUVizFiles() + "/" +
                      params::GetMillisecondBandwidthOutputFilenamePrefix() +
                      "_" + pid_ + "_" + time_str + "_";

  if (enabled_) {
    // Does the initial file open
    RotateLogs();
  }
}

MillisecondBandwidthOutput::~MillisecondBandwidthOutput() {
  // This should generally not be needed since the workaround
  // for not having a destructor on A3+ (which is found in
  // NcclStatsContainer's deleteConnection method), already
  // calls this.
  CloseOutBandwidthStats();
}

void MillisecondBandwidthOutput::AllocateBufferForNic(std::string nic_ip) {
  ip_address_to_nic_data_map_[nic_ip] = std::vector<NicData*>();
  ip_address_to_nic_data_map_[nic_ip].reserve(batches_.size());
  for (Batch& batch : batches_) {
    NicData* nic_data = batch.add_nic_data();
    nic_data->set_ip_address(nic_ip);
    nic_data->mutable_tx_bandwidth_data()->Resize(batch_size_in_samples_, 0);
    nic_data->mutable_rx_bandwidth_data()->Resize(batch_size_in_samples_, 0);
    ip_address_to_nic_data_map_[nic_ip].push_back(nic_data);
    *batch.mutable_process_id() = pid_;
  }
}

void MillisecondBandwidthOutput::CopyOverBandwidthStats(
    uint8_t direction, std::string nic_ip, absl::Span<const double> data) {
  uint64_t milliseconds = current_time_ % 1000;
  uint64_t offset_within_batch =
      (milliseconds / bucket_size_in_msec_) % batch_size_in_samples_;
  int32_t starting_batch = current_batch_;
  int32_t batch_index = current_batch_;
  for (double datum : data) {
    if (direction == 0) {
      ip_address_to_nic_data_map_[nic_ip][batch_index]
          ->mutable_tx_bandwidth_data()
          ->Set(offset_within_batch, datum);
    } else {
      ip_address_to_nic_data_map_[nic_ip][batch_index]
          ->mutable_rx_bandwidth_data()
          ->Set(offset_within_batch, datum);
    }
    ++offset_within_batch;
    // Increment current batch if offset has gotten larger
    // than the batch size.
    batch_index += offset_within_batch / batch_size_in_samples_;
    offset_within_batch %= batch_size_in_samples_;
  }

  next_batch_ = batch_index;

  if (!batch_times_set_) {
    // Still need to set the timestamps for the batches.
    uint64_t first_batch_time_stamp =
        current_time_ - milliseconds % batch_size_in_msec_;
    for (int32_t i = starting_batch; i <= batch_index; i++) {
      uint64_t batch_time =
          first_batch_time_stamp + (i - starting_batch) * batch_size_in_msec_;
      *batches_[i].mutable_time_stamp() =
          google::protobuf::util::TimeUtil::MillisecondsToTimestamp(batch_time);
    }
    batch_times_set_ = true;
  }
}

void MillisecondBandwidthOutput::RotateLogs() {
  std::ofstream replacement_file_stream;
  std::string file_name =
      file_name_prefix_ + std::to_string(num_rotations_) + ".log";

  replacement_file_stream.open(file_name, std::ios_base::app);
  if (!replacement_file_stream) {
    std::string log_msg = "Failed to open or rotate file with error " +
                          std::string(std::strerror(errno));
  }
  if (out_.is_open()) {
    ++num_rotations_;
    out_.close();
  }
  out_.swap(replacement_file_stream);
}

void MillisecondBandwidthOutput::SaveBandwidthStats(bool include_incomplete) {
  if (file_rot_interval_ > absl::ZeroDuration() &&
      (absl::Now() - last_rotation_time_) > file_rot_interval_) {
    RotateLogs();
    last_rotation_time_ = absl::Now();
  }

  MillisecondBandwidthInfo bandwidth_message;
  google::protobuf::RepeatedPtrField<Batch>* pb_batch_list =
      bandwidth_message.mutable_batches();

  // Add the existing batches to the protobuf for export, but do not
  // include the current batch which should be incomplete.
  int upper_limit =
      std::min(static_cast<int>(batches_.size()),
               include_incomplete ? current_batch_ + 1 : current_batch_);
  for (int i = 0; i < upper_limit; ++i) {
    pb_batch_list->AddAllocated(&(batches_[i]));
  }
  bandwidth_message.SerializeToOstream(&out_);
  out_ << "==END==";
  out_.flush();

  // Then remove them again.
  std::vector<Batch*> extracted_batch_list(upper_limit);
  pb_batch_list->ExtractSubrange(0, upper_limit, extracted_batch_list.data());

  // If we're including the incomplete data, then we're finished
  // and can skip the reset procedure.
  if (include_incomplete) return;

  // And then reset the buffer to its initial state except for current batch
  // and anything after
  for (int i = 0; i < upper_limit; ++i) {
    ZeroOutNicData(batches_[i]);
  }

  // Copy data from the current batch to the first batch in the buffer
  for (const NicData& src_nic_data : batches_[upper_limit].nic_data()) {
    NicData* dest_nic_data =
        ip_address_to_nic_data_map_[src_nic_data.ip_address()][0];
    for (uint64_t i = 0; i < batch_size_in_samples_; ++i) {
      dest_nic_data->mutable_tx_bandwidth_data()->Set(
          i, src_nic_data.tx_bandwidth_data()[i]);
      dest_nic_data->mutable_rx_bandwidth_data()->Set(
          i, src_nic_data.rx_bandwidth_data()[i]);
    }
  }

  ZeroOutNicData(batches_[upper_limit]);

  current_batch_ = 0;
  next_batch_ = 0;
};

void MillisecondBandwidthOutput::ZeroOutNicData(Batch& batch) {
  for (NicData& nic_data : *batch.mutable_nic_data()) {
    nic_data.mutable_tx_bandwidth_data()->Clear();
    nic_data.mutable_tx_bandwidth_data()->Resize(batch_size_in_samples_, 0);
    nic_data.mutable_rx_bandwidth_data()->Clear();
    nic_data.mutable_rx_bandwidth_data()->Resize(batch_size_in_samples_, 0);
  }
}

void MillisecondBandwidthOutput::CloseOutBandwidthStats() {
  if (enabled_) {
    SaveBandwidthStats(true);
    out_.close();
    enabled_ = false;
  }
}

}  // namespace gpuviz
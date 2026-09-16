// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_stats_uploader_executor.h"

#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/time/time.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/util/time_util.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "utils.h"

using google::protobuf::RepeatedPtrField;

namespace gpuviz {
namespace {
// Move source[from] to the end of destination.
// Note that if from is at the beginning of the array, this function
// has an O(n) complexity due to the call of
// source->UnsafeArenaExtractSubrange(from, 1, nullptr);
template <typename T>
void MoveSingleElementToEnd(RepeatedPtrField<T>* source, int from,
                            RepeatedPtrField<T>* destination) {
  DCHECK_LE(0, from);
  DCHECK_LE(from, source->size());
  destination->Reserve(destination->size() + 1);
  T* moved_element = source->Mutable(from);
  source->UnsafeArenaExtractSubrange(from, 1, nullptr);
  if (source->GetArena() != nullptr) {
    if (source->GetArena() == destination->GetArena()) {
      // Even if source->Mutable(i) doesn't have an arena, it's still
      // owned by (and possibly allocated by) source->arena(). We can't
      // call AddAllocated in this case since it would delete
      // source->Mutable(i). UnsafeArenaAddAllocated avoids this.
      destination->UnsafeArenaAddAllocated(moved_element);
    } else {
      // Different arenas. Do defensive copy.
      *destination->Add() = *moved_element;
    }
  } else {
    // No arenas involved.
    destination->AddAllocated(moved_element);
  }
}

// Wrapper to invoke a function pointer during destruction.
class EndOfShutdownOperationWrapper {
 public:
  explicit EndOfShutdownOperationWrapper(absl::AnyInvocable<void()> func)
      : func_(std::move(func)) {}
  ~EndOfShutdownOperationWrapper() { func_(); }

 private:
  absl::AnyInvocable<void()> func_;
};
}  // namespace

NcclStatsUploaderExecutor::NcclStatsUploaderExecutor(
    std::unique_ptr<UploaderInterface> uploader, ncclDebugLogger_t log_function)
    : uploader_(std::move(uploader)),
      log_function_(log_function),
      source_process_identifier_(gpuviz::utils::getProcessIdentifier()) {
  uploader_thread_ =
      std::thread(&NcclStatsUploaderExecutor::RunUploadExecutor, this);
}

std::queue<std::unique_ptr<protoDist::AllStats>>
NcclStatsUploaderExecutor::SplitAllStatsMessageIntoQ(
    std::unique_ptr<protoDist::AllStats> msg, const size_t bytes_per_message) {
  // Create a new q and put the splitted messages into the it and replace
  // the q_ with this new q later. Doing it this way would avoid holding lock
  // in this function.
  std::queue<std::unique_ptr<protoDist::AllStats>> q;
  // local variable to single chunk of message.
  auto sub_m = std::make_unique<protoDist::AllStats>();

  // The wire format size of the sub_m, calculated by adding up all
  // the individual sub-fields' (LogEntry, ConnectionStats, Events) wire
  // format size. This is not 100% accurate, as there should be some
  // other overhead like descriptor and padding that would take up
  // some space, but a reasonable approximation.
  size_t current_cumulative_size = 0;

  // Helper lambda to add sub_m into q, and reset the sub_m to start
  // storing new chunk.
  auto addSubMessageToQ = [&q, &sub_m, &current_cumulative_size, this]() {
    sub_m->set_source_process_identifier(this->source_process_identifier_);
    // index_ is uint64, overflow behavior is well-defined:
    // https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Unsigned-Overflow.html
    sub_m->set_upload_index(this->index_++);
    q.push(std::move(sub_m));
    sub_m = std::make_unique<protoDist::AllStats>();
    current_cumulative_size = 0;
  };

  // The way we copy the address of the underlying data of RepeatedPtrField is
  // borrowed from the Google internal protobuf util function:
  // MoveRangeToEnd() in repeated_field_util.h.
  // We do it the reverse way because MoveSingleElementToEnd() involves
  // a call to move the element from RepeatedPtrField array. If we do it
  // from the beginning of the array, it would have a O(n^2) complexity.
  auto splitField = [&bytes_per_message, &current_cumulative_size,
                     &addSubMessageToQ]<typename T>(
                        RepeatedPtrField<T>* source,
                        RepeatedPtrField<T>* destination) {
    for (int i = source->size() - 1; i >= 0; --i) {
      size_t current_size = source->Mutable(i)->ByteSizeLong();

      // If current_size > bytes_per_message and current sub_m is empty,
      // i.e., the first message of the sub_m is already larger than quota,
      // we don't enter the if block below, will push this message into sub_m,
      // and then go into next iteration, where we will add sub_m to q.
      if (current_cumulative_size != 0 &&
          current_cumulative_size + current_size > bytes_per_message) {
        addSubMessageToQ();
      }
      // Pass the ownership of the underlying entry from msg to sub_m.
      MoveSingleElementToEnd(source, i, destination);
      current_cumulative_size += current_size;
    }
  };

  // Prioritize on processing all the LogEntry as they may contain
  // more fatal and obvious errors that we are interested.
  splitField(msg->mutable_events_log(), sub_m->mutable_events_log());
  splitField(msg->mutable_conn_stats(), sub_m->mutable_conn_stats());
  splitField(msg->mutable_events(), sub_m->mutable_events());

  // Push the last remaining sub_m into q.
  // If sub_m is empty, it will serve as a heart beat message.
  addSubMessageToQ();
  return q;
}

bool NcclStatsUploaderExecutor::UploadFinished() {
  // If uploader becomes dead, then the q_  will
  // never be empty, so we should consider the Upload is finished
  // to unblock the write on disk operation.
  if (uploader_ == nullptr || uploader_->IsDead()) {
    return true;
  }
  absl::MutexLock lock(&mtx_);
  return shutdown_ || q_.empty();
}

void NcclStatsUploaderExecutor::Shutdown() {
  if (uploader_thread_.joinable()) {
    {
      absl::MutexLock lock(&mtx_);
      shutdown_ = true;
    }
    uploader_thread_.join();
    uploader_ = nullptr;
  }
}

void NcclStatsUploaderExecutor::RunUploadExecutor() {
  // The purpose to create this wrapper is to invoke
  // UploadAndCleanupRemainingTelemetryOnShutdown before this function exits.
  EndOfShutdownOperationWrapper wrapper(absl::bind_front(
      &NcclStatsUploaderExecutor::UploadAndCleanupRemainingTelemetryOnShutdown,
      this));

  if (uploader_ == nullptr || uploader_->IsDead()) {
    return;
  }
  absl::Time next_execute_time = uploader_->NextUploadTime();
  // Local buffer to store single message for uploading.
  Msg m = nullptr;
  while (true) {
    // Block indefinitely unless q_ becomes non-empty or we want to shutdown.
    if (WaitForWorkOrShutdown()) {
      return;
    }

    // Now that queue becomes non-empty, upload them one by one.
    // The following WaitForTimeoutOrShutdown equals to "sleep until next
    // execute time, but allow to be interrupted by shutdown."
    if (WaitForTimeoutOrShutdown(next_execute_time)) {
      return;
    }

    {
      absl::MutexLock lock(&mtx_);
      // Note that we only get the value from the front of the q_, but we do not
      // want to pop the q_ until we know the Upload() is successful. We do it
      // this way because if the Upload() fails, we want the q_ to remain
      // non-empty, so that the WaitForWorkOrShutdown() in next iteration will
      // not block, and we can try to upload again quickly.
      m = std::move(q_.front());
    }
    absl::Status upload_status = uploader_->Upload(*m);
    if (upload_status.ok()) {
      // If upload succeeds, reset the buffer, and pop the queue.
      ABSL_VLOG(2) << "message with upload index: " << m->upload_index()
                   << " is successfully uploaded from source identifier: "
                   << m->source_process_identifier();
      m = nullptr;
      absl::MutexLock lock(&mtx_);
      q_.pop();
      last_upload_success_ = true;
    } else {
      ABSL_VLOG(1) << "message from source: " << m->source_process_identifier()
                   << " with upload index: " << m->upload_index()
                   << " failed with status:" << upload_status.ToString();
      {
        absl::MutexLock lock(&mtx_);
        // If upload failed, move the pointer stored in local buffer, m, back to
        // q_.
        q_.front() = std::move(m);
        last_upload_success_ = false;
      }
      if (uploader_->IsDead()) {
        absl::MutexLock lock(&mtx_);
        shutdown_ = true;
        return;
      }
    }
    next_execute_time = uploader_->NextUploadTime();
  }
}

absl::Status NcclStatsUploaderExecutor::UpdateQueue(Msg m) {
  if (uploader_ == nullptr) {
    ABSL_LOG(WARNING) << "attempting to upload telemetry but uploader is null. "
                         "This should never happen.";
    return absl::FailedPreconditionError("No uploader exists.");
  }
  *(m->add_events_log()) = CreateUploadStartMessage();
  MsgQ q = SplitAllStatsMessageIntoQ(std::move(m),
                                     uploader_->GetMaxMessageByteSize());
  absl::MutexLock lock(&mtx_);
  if (q_.empty()) {
    q_ = std::move(q);
    return absl::OkStatus();
  } else {
    ABSL_LOG(WARNING)
        << "NcclStatsUploaderExecutor only supported calling UpdateQueue "
        << "when UploadFinished() returns true.";
    return absl::FailedPreconditionError(
        "Updating queue is only supported when queue is empty.");
  }
}

NcclStatsUploaderExecutor::~NcclStatsUploaderExecutor() { Shutdown(); }

bool NcclStatsUploaderExecutor::ShouldWakeUp() {
  return shutdown_ || !q_.empty();
}

bool NcclStatsUploaderExecutor::WaitForWorkOrShutdown() {
  mtx_.LockWhen(
      absl::Condition(this, &NcclStatsUploaderExecutor::ShouldWakeUp));
  bool local_shutdown_sample = shutdown_;
  mtx_.Unlock();
  return local_shutdown_sample;
}

bool NcclStatsUploaderExecutor::WaitForTimeoutOrShutdown(
    const absl::Time next_execute_time) {
  mtx_.LockWhenWithTimeout(
      absl::Condition(
          +[](const bool* shutdown) { return *shutdown; }, &shutdown_),
      next_execute_time - gpuviz::utils::get_current_monotonic_time());
  bool local_shutdown_sample = shutdown_;
  mtx_.Unlock();
  return local_shutdown_sample;
}

protoDist::LogEntry NcclStatsUploaderExecutor::CreateShutdownMessage() {
  protoDist::LogEntry shutdown_message;
  shutdown_message.set_category_name("Uploader_Shutdown");
  *(shutdown_message.mutable_time_stamp()) =
      google::protobuf::util::TimeUtil::GetCurrentTime();
  return shutdown_message;
}

protoDist::LogEntry NcclStatsUploaderExecutor::CreateUploadStartMessage() {
  protoDist::LogEntry upload_start_message;
  upload_start_message.set_category_name("Uploader_Start");
  *(upload_start_message.mutable_time_stamp()) =
      google::protobuf::util::TimeUtil::GetCurrentTime();
  return upload_start_message;
}

void NcclStatsUploaderExecutor::UploadAndCleanupRemainingTelemetryOnShutdown() {
  {
    absl::MutexLock lock(&mtx_);
    if (!last_upload_success_ || uploader_ == nullptr || uploader_->IsDead()) {
      // Immediately return if the uploader was shutdown due to failure of
      // upload. Before return, we want to clean up Q to free memory taken by
      // data that will never be uploaded.
      CleanUpQ();
      return;
    }
  }
  Msg m = nullptr;
  {
    // If uploader is healthy, ideally we want to upload all the remaining
    // messages before shutdown. However, this could also flood the ACS server.
    // Therefore, we choose to only upload the front of the queue for now.
    absl::MutexLock lock(&mtx_);
    if (!q_.empty()) {
      m = std::move(q_.front());
    }
  }

  // Add the shutdown message into the last uploaded message.
  protoDist::LogEntry shutdown_log_entry = CreateShutdownMessage();
  if (m == nullptr) {
    m = std::make_unique<protoDist::AllStats>();
    m->set_source_process_identifier(source_process_identifier_);
  }
  *(m->add_events_log()) = std::move(shutdown_log_entry);
  absl::Status upload_status = uploader_->Upload(*m, /*ignore_next_execute_time=*/true);
  if (!upload_status.ok()) {
    ABSL_VLOG(1)
        << "Last message from source: " << m->source_process_identifier()
        << " with upload-index: " << m->upload_index()
        << " before shutdown upload failed, not retrying, with status: "
        << upload_status.ToString();
  }
  absl::MutexLock lock(&mtx_);
  CleanUpQ();
}

void NcclStatsUploaderExecutor::CleanUpQ() {
  while (!q_.empty()) {
    q_.pop();
  }
}

}  // namespace gpuviz

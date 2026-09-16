// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_uploader.h"

#include <cmath>
#include <cstdlib>

#include "utils.h"

namespace gpuviz {

using ::gpuviz::utils::get_current_monotonic_time;

absl::Status NcclUploader::Upload(const google::protobuf::Message& data,
                                bool ignore_next_execute_time) {
  if (is_dead_) {
    return absl::InternalError("Uploader is already dead, don't retry.");
  }
  if (!ignore_next_execute_time &&
      get_current_monotonic_time() < next_execute_time_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Execute the Upload function too early, please wait till: ",
        next_execute_time_));
  }

  // If the client is not dead, this function will send the message through
  // the client. If the send is not successful, it will set the
  // next_execute_time with a backoff and return failure status.
  // It is entirely possible that after UploadMessage(), client becomes dead.
  // We will defer the re-creation of the client to the next Upload() call to
  // allow a backoff to wait before next attempt to create the client.
  if (client_ != nullptr && !client_->IsDead()) {
    return UploadMessage(data);
  }

  // If the client is dead, this function will first try to create a
  // client. If the creation is not successful, it will set the
  // next_execute_time with a backoff and return failure status. If the creation
  // is successful, it will send the message through the client.
  return CreateClientAndUploadMessage(data);
}

absl::Time NcclUploader::NextUploadTime() { return next_execute_time_; }

size_t NcclUploader::GetMaxMessageByteSize() {
  if (client_ != nullptr && !client_->IsDead()) {
    return client_->GetMaxMessageByteSize();
  }
  return kMaxMessageSize;
}

absl::Status NcclUploader::UploadMessage(
    const google::protobuf::Message& data) {
  absl::Status send_status = client_->SendMessage(data);
  absl::Time current_time = get_current_monotonic_time();
  if (send_status.ok()) {
    attempts_send_message_ = 0;
    next_execute_time_ = current_time + min_wait_;
    return send_status;
  }

  // If this write failed and client is dead, we should not retry to write.
  // Instead we shutdown the client_, so that when Upload() is executed next
  // time, we will create a new client. Meanwhile, the next execute
  // time will be calculated given an exponential backoff.
  if (client_->IsDead()) {
    client_ = nullptr;
  }
  ++attempts_send_message_;
  if (attempts_send_message_ >= send_messages_backoff_.max_attempts) {
    GiveUpRetry("Give up on retry upload message, shutting down the uploader.");
  } else {
    next_execute_time_ =
        current_time +
        std::max(min_wait_, CalculateBackoff(send_messages_backoff_,
                                             attempts_send_message_));
  }
  return send_status;
}

void NcclUploader::DelayFirstClientCreate() {
  if (first_client_creation_done_) {
    return;
  }
  first_client_creation_done_ = true;
  // Overwritten only in unit test.
  if (delay_create_client_ms_override_.has_value()) {
    absl::SleepFor(
        absl::Milliseconds(delay_create_client_ms_override_.value()));
    return;
  }
  absl::SleepFor(absl::Milliseconds(
      absl::Uniform<uint64_t>(gen_, 0, kMaxDelayOfCreateClientMs)));
}

absl::Status NcclUploader::CreateClientAndUploadMessage(
    const google::protobuf::Message& data) {
  DelayFirstClientCreate();
  absl::StatusOr<std::unique_ptr<Client>> client = client_creator_();
  if (client.ok()) {
    client_ = *(std::move(client));
    min_wait_ = client_->GetMinWait();
    attempts_create_client_ = 0;
    return UploadMessage(data);
  }

  // Client creation failed, retry iff the number of attempts has not reached
  // max.
  ++attempts_create_client_;
  if (attempts_create_client_ >= create_client_backoff_.max_attempts) {
    GiveUpRetry("Give up on retry create client, shutting down the uploader.");
  } else {
    next_execute_time_ =
        get_current_monotonic_time() +
        CalculateBackoff(create_client_backoff_, attempts_create_client_);
  }
  return client.status();
}

void NcclUploader::GiveUpRetry(std::string log_message) {
  next_execute_time_ = absl::InfiniteFuture();
  client_ = nullptr;
  is_dead_ = true;
  ABSL_LOG(WARNING) << log_message;
}

absl::Duration NcclUploader::CalculateBackoff(
    const BackOffParameters& backoff_parameters, int64_t num_retries) {
  if (backoff_calculator_ != nullptr) {
    return backoff_calculator_(backoff_parameters, num_retries);
  }
  if (num_retries >= backoff_parameters.max_attempts) {
    ABSL_LOG(WARNING) << "num of attempts exceed max allowable.";
    return absl::InfiniteDuration();
  }

  // Calculation algorithm reference:
  // https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
  // We are using the "Equal jitter" here.
  double exponential_delay =
      std::min(backoff_parameters.cap,
               backoff_parameters.base_delay *
                   std::pow(backoff_parameters.base, num_retries - 1));
  double randomized_delay =
      exponential_delay / 2 +
      absl::Uniform<double>(gen_, 0, exponential_delay / 2);
  return absl::Milliseconds(randomized_delay);
}

}  // namespace gpuviz

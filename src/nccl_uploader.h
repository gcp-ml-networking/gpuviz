/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_UPLOADER_H_
#define NCCL_UPLOADER_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "client.h"
#include "google/protobuf/message.h"
#include "include/nccl_net.h"
#include "src/proto/nccl_telemetry_proto.pb.h"
#include "uploader_interface.h"
#include "utils.h"

namespace gpuviz {

// Facilitates the create of Client and its upload operations.
// This class is expected to be used in single-threaded environment, hence
// not thread-safe.
class NcclUploader : public UploaderInterface {
 public:
  struct BackOffParameters {
    // Example of using this parameters.
    // If failed_attempts < max_attempts, delay =
    // std::min(cap, base_delay * base ** (failed_attempt - 1));
    double base_delay;     // in milliseconds.
    double base;           // unitless
    double cap;            // in milliseconds.
    int64_t max_attempts;  // max allowable # of failed attempts.
  };
  using client_creator_func =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Client>>()>;
  using backoff_calculate_func =
      absl::AnyInvocable<absl::Duration(const BackOffParameters&, int64_t)>;

  // Constructor of the NcclUploader class.
  // client_creator: potentially factory method for any class that implements
  // the Client interface.
  // backoff_calculator: function pointer to calculate backoff when failure to
  // create client or send message happens.
  // send_message_backoff and create_client_backoff: parameter used to
  // calculate the backoff.
  explicit NcclUploader(
      ncclDebugLogger_t log_function, client_creator_func client_creator,
      backoff_calculate_func backoff_calculator,
      BackOffParameters send_message_backoff,
      BackOffParameters create_client_backoff,
      std::optional<uint64_t> delay_create_client_ms_override = std::nullopt)
      : log_function_(log_function),
        client_creator_(std::move(client_creator)),
        delay_create_client_ms_override_(delay_create_client_ms_override),
        first_client_creation_done_(false),
        backoff_calculator_(std::move(backoff_calculator)),
        send_messages_backoff_(std::move(send_message_backoff)),
        create_client_backoff_(std::move(create_client_backoff)) {
    next_execute_time_ = gpuviz::utils::get_current_monotonic_time();
  }

  // Uploads a data protobuf message through client.
  // Creates a gRPC client if existing client does not exist or is dead.
  // This call will block until we know the status of the gRPC write operation.
  // Regardless of whether the upload is successful or not, the caller needs to
  // call NextUploadTime() to know the next earliest allowable time to Upload
  // again. If # of failures exceed max allowable stored
  // in backoff parameter, uploader will be dead and return the corresponding
  // status.
  absl::Status Upload(const google::protobuf::Message& data,
                      bool ignore_next_execute_time = false) override;

  // Returns the next earliest allowable upload time.
  // It should be called after Upload() returns. The next Upload() call should
  // not happen before NextUploadTime().
  // If the upload was successful, the next upload time will be Now() +
  // min_wait_. If the upload was not successful, the next upload time will be
  // Now() + calculated backoff.
  absl::Time NextUploadTime() override;

  // Returns the max # of wire format bytes each uploaded message can have.
  size_t GetMaxMessageByteSize() override;

  // Returns whether the uploader is considered dead.
  // This is not to be confused with whether the grpc client is dead. When grpc
  // client is dead, uploader would try to recreate the client with some
  // backoff. Or if the Send message failed, uploader would also try to re-send
  // with some backoff. However, if the # of failed attempts reached the max
  // allowable, we would give up on retry and make the uploader dead. The caller
  // of this class is not supposed to call Upload() anymore after the uploader
  // is dead. Otherwise, the Upload() call will return a failure status to tell
  // caller to stop calling.
  bool IsDead() override { return is_dead_; }

  ~NcclUploader() override = default;

 private:
  // Uploads the message and return status.
  // This function is called within Upload() when the gRPC client is alive.
  absl::Status UploadMessage(const google::protobuf::Message& data);

  // Delay the first client creation.
  void DelayFirstClientCreate();

  // Creates a new grpc client and immediately call UploadMessage().
  absl::Status CreateClientAndUploadMessage(
      const google::protobuf::Message& data);

  // Max number of retries on UploadMessage() or CreateClientAndUploadMessage()
  // reached.
  void GiveUpRetry(std::string log_message);

  ncclDebugLogger_t log_function_;

  // Called in CreateClientAndUploadMessage() to generate new client_.
  client_creator_func client_creator_;

  std::unique_ptr<Client> client_ = nullptr;

  // Cap of delay when creating the client to avoid all agents connecting to
  // ACS at the same time.
  inline static constexpr uint64_t kMaxDelayOfCreateClientMs = 5000;

  // Used in unit test only.
  std::optional<uint64_t> delay_create_client_ms_override_ = std::nullopt;

  // flag to indicate whether we finished invoking the first client creation.
  bool first_client_creation_done_ = false;

  // Number of failed attempts to create the client. Reset to 0 after
  // any successful creation of client.
  int64_t attempts_create_client_ = 0;

  // Number of failed attempts to write. Reset after any successful write.
  int64_t attempts_send_message_ = 0;

  bool is_dead_ = false;

  // default max message size.
  inline static constexpr size_t kMaxMessageSize = 1048576;

  // Earliest Time for the next Upload.
  absl::Time next_execute_time_;

  backoff_calculate_func backoff_calculator_;

  // Parameters for the backoff calculation.
  BackOffParameters send_messages_backoff_;
  BackOffParameters create_client_backoff_;

  absl::BitGen gen_;

  // Calculates the next wait duration given the parameters and the current
  // number of retries. eg., if you have 1 attempt to upload or write, and
  // this attempt failed, then to calculate the backoff, you should input 0
  // instead of 1.
  absl::Duration CalculateBackoff(const BackOffParameters& backoff_parameters,
                                  int64_t num_retries);

  // Duration to be waited before the next client's write operation is
  // invoked. This variable is to store the result of Client::get_min_wait().
  absl::Duration min_wait_;
};

}  // namespace gpuviz

#endif  // NCCL_UPLOADER_H_

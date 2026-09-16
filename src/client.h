/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef CLIENT_H_
#define CLIENT_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cpp/acs_agent_client.h"
#include "google/protobuf/message.h"
#include "src/proto/acs_configuration.pb.h"

namespace gpuviz {

// Interface for gRPC client used in Uploader.
class Client {
 public:
  // Sends message through RPC stream.
  virtual absl::Status SendMessage(
      const google::protobuf::Message& message) = 0;

  // Queries if client is dead.
  // The caller of client is expected to shutdown the client by invoking
  // destructor, and then re-create the client with a backoff.
  virtual bool IsDead() = 0;

  // Returns the # of milliseconds to wait between two sends.
  // This is calculated by 1000 milliseconds / QPS.
  virtual absl::Duration GetMinWait() = 0;

  // Returns the max # of wire format bytes each uploaded message can have.
  virtual size_t GetMaxMessageByteSize() = 0;

  virtual ~Client() {}
};

class AcsAgentClientWrapper : public Client {
 public:
  static absl::StatusOr<std::unique_ptr<AcsAgentClientWrapper>> Create();
  absl::Status SendMessage(const google::protobuf::Message& message) override;
  bool IsDead() override;
  absl::Duration GetMinWait() override;
  size_t GetMaxMessageByteSize() override;
  ~AcsAgentClientWrapper() override = default;

 private:
  class AcsConfig {
   public:
    std::optional<std::string> GetVersion() const ABSL_LOCKS_EXCLUDED(mtx_);
    uint64_t GetMessagesPerMinute() const ABSL_LOCKS_EXCLUDED(mtx_);
    void Update(const AcsConfiguration& new_acs_config)
        ABSL_LOCKS_EXCLUDED(mtx_);

   private:
    static constexpr uint64_t kDefaultMessagesPerMinute =
        6;  // Send a message every 10 seconds.
    mutable absl::Mutex mtx_;
    std::optional<std::string> version_ ABSL_GUARDED_BY(mtx_);
    uint64_t messages_per_minute_ ABSL_GUARDED_BY(mtx_) =
        kDefaultMessagesPerMinute;
  };
  AcsAgentClientWrapper(
      std::unique_ptr<agent_communication::AcsAgentClient> client,
      std::unique_ptr<AcsConfig> config)
      : client_(std::move(client)), config_(std::move(config)) {}
  uint64_t GetMessagesPerMinute();
  std::unique_ptr<agent_communication::AcsAgentClient> client_;
  std::unique_ptr<AcsConfig> config_;
  inline static constexpr absl::string_view kChannelId =
      "compute.googleapis.com/nccl-telemetry-client";
};

}  // namespace gpuviz

#endif  // CLIENT_H_
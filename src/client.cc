// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "client.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/globals.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "cpp/acs_agent_client.h"
#include "cpp/acs_agent_client_reactor.h"
#include "cpp/acs_agent_helper.h"
#include "params.h"
#include "proto/agent_communication.grpc.pb.h"
#include "src/proto/acs_configuration.pb.h"

namespace gpuviz {

using namespace ::agent_communication;
using namespace ::gpuviz::params;
using ::google::cloud::agentcommunication::v1::MessageBody;
using ::google::cloud::agentcommunication::v1::StreamAgentMessagesResponse;

constexpr size_t kDefaultMaxMessageByteSize = 256 * 1024;  // 256 KB
constexpr absl::string_view kConfigVersionLabel = "config_version";

absl::StatusOr<std::unique_ptr<AcsAgentClientWrapper>>
AcsAgentClientWrapper::Create() {
  // Create AcsAgentClient.
  std::string end_point = AcsServerEndPointOverwritten();
  // Retrieve this boolean from env var if set.
  bool regional = IsAcsEndPointRegional();
  std::string channel_id = std::string(kChannelId);
  const std::chrono::milliseconds ack_timeout(
      GetAcsServerAckTimeoutInMilliseconds());
  absl::StatusOr<std::unique_ptr<agent_communication::AcsAgentClient>> client;
  auto config = std::make_unique<AcsAgentClientWrapper::AcsConfig>();
  auto read_callback = [config = config.get()](
                           StreamAgentMessagesResponse response) {
    if (response.has_message_body()) {
      if (response.message_body().body().Is<AcsConfiguration>()) {
        AcsConfiguration new_acs_config;
        if (response.message_body().body().UnpackTo(&new_acs_config)) {
          ABSL_VLOG(2) << "Successfully unpacked AcsConfiguration: "
                       << new_acs_config;
          config->Update(new_acs_config);
        } else {
          // This should not happen if Is<AcsConfiguration>() is true
          ABSL_LOG(ERROR) << "Failed to unpack AcsConfiguration even though "
                             "type check passed.";
        }
      } else {
        ABSL_VLOG(2)
            << "Received message body does not contain an AcsConfiguration: "
            << response.message_body().body().type_url();
      }
    }
  };

  if (!end_point.empty()) {
    // Endpoint is overwritten by env var.
    auto stub = AcsAgentClientReactor::CreateStub(end_point);
    absl::StatusOr<AgentConnectionId> agent_connection_id =
        agent_communication::GenerateAgentConnectionId(channel_id, regional);
    if (!agent_connection_id.ok()) {
      ABSL_LOG(ERROR) << "Failed to generate agent connection id: "
                      << agent_connection_id.status();
      return agent_connection_id.status();
    }
    client = AcsAgentClient::Create(
        // ACS stub.
        std::move(stub),
        // AgentConnectionId.
        *std::move(agent_connection_id),
        // Read callback. Invoked when a response is read from server.
        read_callback,
        // Acs stub generator. Invoked when the connection is cut off by server
        // and Client wants to restart itself.
        // Captured by value here because end_point is going to be outlived by
        // this function pointer.
        [end_point]() { return AcsAgentClientReactor::CreateStub(end_point); },
        // AgentConnectionId generator. Captured by value as well because
        // this function outlives the variable.
        [channel_id, regional]() {
          return agent_communication::GenerateAgentConnectionId(channel_id,
                                                                regional);
        },
        ack_timeout);
  } else {
    client = AcsAgentClient::Create(regional, channel_id, read_callback);
  }

  if (!client.ok()) {
    ABSL_LOG(INFO) << "failed to create AcsAgentClient: "
                   << client.status().ToString();
    return client.status();
  }
  return absl::WrapUnique(
      new AcsAgentClientWrapper(*std::move(client), std::move(config)));
}

std::optional<std::string> AcsAgentClientWrapper::AcsConfig::GetVersion()
    const {
  absl::MutexLock lock(&mtx_);
  return version_;
}

uint64_t AcsAgentClientWrapper::AcsConfig::GetMessagesPerMinute() const {
  absl::MutexLock lock(&mtx_);
  return messages_per_minute_;
}

void AcsAgentClientWrapper::AcsConfig::Update(
    const AcsConfiguration& new_acs_config) {
  absl::MutexLock lock(&mtx_);
  version_ = new_acs_config.version();
  if (new_acs_config.has_agent_config()) {
    messages_per_minute_ = new_acs_config.agent_config().messages_per_minute();
    ABSL_VLOG(2) << "Updated with the new AgentConfiguration: "
                 << new_acs_config.agent_config();
  } else {
    ABSL_VLOG(2) << "No agent_config present";
  }
}

absl::Status AcsAgentClientWrapper::SendMessage(
    const google::protobuf::Message& message) {
  MessageBody msg_body;
  std::optional<std::string> version = config_->GetVersion();
  if (version.has_value()) {
    (*msg_body.mutable_labels())[kConfigVersionLabel] = *version;
  }
  msg_body.mutable_body()->PackFrom(message);
  return client_->SendMessage(std::move(msg_body));
}

bool AcsAgentClientWrapper::IsDead() {
  return client_ == nullptr || client_->IsDead();
}

uint64_t AcsAgentClientWrapper::GetMessagesPerMinute() {
  const uint64_t messages_per_minute = config_->GetMessagesPerMinute();
  absl::StatusOr<uint64_t> messages_per_minute_quota =
      client_->GetMessagePerMinuteQuota();
  if (!messages_per_minute_quota.ok()) {
    ABSL_LOG(INFO) << "Messages per minute quota is not available: "
                   << messages_per_minute_quota.status();
    return messages_per_minute;
  }
  if (messages_per_minute_quota.value() == 0) {
    ABSL_LOG(INFO) << "Messages per minute quota cannot be 0";
    return messages_per_minute;
  }
  return std::min(messages_per_minute_quota.value(), messages_per_minute);
}

absl::Duration AcsAgentClientWrapper::GetMinWait() {
  return absl::Minutes(1) / GetMessagesPerMinute();
}

size_t AcsAgentClientWrapper::GetMaxMessageByteSize() {
  absl::StatusOr<uint64_t> bytes_per_minute_quota =
      client_->GetBytesPerMinuteQuota();
  if (!bytes_per_minute_quota.ok()) {
    ABSL_LOG(INFO) << "Bytes per minute quota is not available: "
                   << bytes_per_minute_quota.status();
    return kDefaultMaxMessageByteSize;
  }
  return bytes_per_minute_quota.value() / GetMessagesPerMinute();
}

}  // namespace gpuviz

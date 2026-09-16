/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef TELEMETRY_PROVIDER_H_
#define TELEMETRY_PROVIDER_H_

#include "../include/nccl_net.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"
#include "src/proto/nccl_telemetry_proto.pb.h"

namespace gpuviz {

class TelemetryProvider {
 public:
  virtual void addLogMsg(absl::string_view message, ncclDebugLogLevel level,
                         const char* pretty_func, uint32_t line,
                         ncclDebugLogSubSys subsys, std::string category_name,
                         int32_t category, int32_t msg_id) = 0;

  // Convenience overload
  virtual void addLogMsg(absl::string_view message, ncclDebugLogLevel level,
                         const char* func, uint32_t line) = 0;
  virtual absl::StatusOr<std::unique_ptr<protoDist::AllStats>>
  readTelemetry() = 0;
  virtual absl::Status processHighFrequencyTelemetry() = 0;
  virtual ~TelemetryProvider() = default;
};

}  // namespace gpuviz

#endif  // TELEMETRY_PROVIDER_H_

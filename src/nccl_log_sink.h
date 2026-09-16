/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_LOG_SINK_H_
#define NCCL_LOG_SINK_H_

#include <memory>

#include "absl/base/log_severity.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "include/nccl_net.h"
#include "telemetry_provider.h"

namespace gpuviz {

// Log sink class that captures all the ABSL_LOG() generated mainly by ACS
// library and pipe it into NcclDebugLogger and aggregator's log message
// storage.
class NcclLogSink : absl::LogSink {
 public:
  NcclLogSink(std::shared_ptr<TelemetryProvider> provider,
              ncclDebugLogger_t log_function)
      : telemetry_provider_(provider) {
    absl::AddLogSink(this);
  }

  ~NcclLogSink() override { absl::RemoveLogSink(this); }

  void Send(const absl::LogEntry& entry) override {
    ncclDebugLogLevel nccl_debug_log_level =
        ConvertAbslSeverityToNcclDebugLogLevel(entry.log_severity());

    std::string log_message(entry.text_message_with_prefix_and_newline());
    if (telemetry_provider_ != nullptr) {
      telemetry_provider_->addLogMsg(log_message, nccl_debug_log_level,
                                     __PRETTY_FUNCTION__, __LINE__);
    }
  }

 private:
  std::shared_ptr<TelemetryProvider> telemetry_provider_;

  // Converts the enum of absl::LogSeverity to enum of ncclDebugLogLevel.
  static ncclDebugLogLevel ConvertAbslSeverityToNcclDebugLogLevel(
      const absl::LogSeverity log_severity) {
    switch (log_severity) {
      case absl::LogSeverity::kInfo:
        return NCCL_LOG_INFO;
      // NCCL regards all errors as WARN. Therefore, we consider everything
      // above warning in absl as NCCL_LOG_WARN.
      case absl::LogSeverity::kWarning:
        return NCCL_LOG_WARN;
      case absl::LogSeverity::kError:
        return NCCL_LOG_WARN;
      case absl::LogSeverity::kFatal:
        return NCCL_LOG_WARN;
      // absl::LogSeverity's scope should be relatively stable. In case new
      // severity comes up, we consider it as INFO.
      default:
        return NCCL_LOG_INFO;
    }
  }
};

}  // namespace gpuviz

#endif  // NCCL_LOG_SINK_H_

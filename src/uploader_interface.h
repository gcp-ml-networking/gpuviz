/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef UPLOADER_INTERFACE_H_
#define UPLOADER_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "google/protobuf/message.h"

namespace gpuviz {

class UploaderInterface {
 public:
  virtual absl::Status Upload(const google::protobuf::Message& data,
                              bool ignore_next_execute_time = false) = 0;
  virtual absl::Time NextUploadTime() = 0;
  virtual size_t GetMaxMessageByteSize() = 0;
  virtual bool IsDead() = 0;
  virtual ~UploaderInterface() {};
};

}  // namespace gpuviz

#endif  // UPLOADER_INTERFACE_H_
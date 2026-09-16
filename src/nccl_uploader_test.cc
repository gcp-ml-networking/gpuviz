// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_uploader.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"
#include "src/proto/nccl_telemetry_proto.pb.h"

namespace gpuviz {
namespace {

using namespace ::testing;
using ::testing::Test;

// Generates fake proto to upload.
protoDist::ncclStatsTCPConnection GenerateFakeProtobuf(std::string local,
                                                       std::string remote) {
  protoDist::ncclStatsTCPConnection connection;
  connection.set_local_endpoint(std::move(local));
  connection.set_remote_endpoint(std::move(remote));
  return connection;
}

// Mock client class of interface Client
class MockClient : public Client {
 public:
  MockClient() {}
  ~MockClient() = default;
  MOCK_METHOD(absl::Status, SendMessage, (const google::protobuf::Message&),
              (override));
  MOCK_METHOD(bool, IsDead, (), (override));
  MOCK_METHOD(absl::Duration, GetMinWait, (), (override));
  MOCK_METHOD(size_t, GetMaxMessageByteSize, (), (override));
};

class NcclUploaderTest : public Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test happy case: uploader can repeatedly upload successfully, and the
// NextUploadTime should be current_time + default min_wait.
TEST_F(NcclUploaderTest, TestRepeatedlyWriteSucceed) {
  absl::Duration min_wait = absl::Milliseconds(100);
  std::unique_ptr<NcclUploader> uploader_ = std::make_unique<NcclUploader>(
      nullptr,
      [min_wait]() {
        std::unique_ptr<MockClient> mock_client_ =
            std::make_unique<MockClient>();
        EXPECT_CALL(*mock_client_, SendMessage(_))
            .Times(5)
            .WillRepeatedly(Return(absl::OkStatus()));
        EXPECT_CALL(*mock_client_, GetMinWait()).WillOnce(Return(min_wait));
        EXPECT_CALL(*mock_client_, IsDead())
            .Times(4)
            .WillRepeatedly(Return(false));
        return mock_client_;
      },
      nullptr, NcclUploader::BackOffParameters(),
      NcclUploader::BackOffParameters(), 0);
  absl::Time current_time = gpuviz::utils::get_current_monotonic_time();
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE((uploader_->Upload(GenerateFakeProtobuf("1", "2"))).ok());
    current_time = gpuviz::utils::get_current_monotonic_time();
    absl::Time next_time = uploader_->NextUploadTime();
    // Use EXPECT_LE here because actual current_time here is calculated within
    // Upload() call, which is earlier than the current_time here.
    EXPECT_LE(uploader_->NextUploadTime() - current_time, min_wait);
    absl::Status status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_THAT(status.message(),
                StrEq(absl::StrCat(
                    "Execute the Upload function too early, please wait till: ",
                    uploader_->NextUploadTime())));
    absl::SleepFor(next_time - current_time);
    EXPECT_FALSE(uploader_->IsDead());
  }
}

// Test the case when we have repeatedly Upload failure but the grpc client
// stays alive all the time. After the # of failed attempts reaches the
// BackOffParameters.max_attempts, the uploader will give up on retry and be
// dead.
TEST_F(NcclUploaderTest, TestWriteFailRepeatedlyTillUploaderDead) {
  absl::Duration min_wait = absl::Milliseconds(100);
  // We will call uploader_.Upload(message) 3 times. The first 2 attempts will
  // both fail. Then uploader_ will be dead. The 3rd attempt will return early
  // with InternalError.
  auto uploader_ = std::make_unique<NcclUploader>(
      nullptr,
      [&min_wait]() {
        std::unique_ptr<MockClient> mock_client_ =
            std::make_unique<MockClient>();
        // Will be called exactly 3 times: after the 1st attempt failed, before
        // the 2nd attempt started, after the 2nd attempt failed.
        EXPECT_CALL(*mock_client_, IsDead())
            .Times(3)
            .WillRepeatedly(Return(false));
        // Very Important to check Times(2) here, because after the first 2
        // attempts failed, uploader will be dead, and the 3rd call to
        // uploader_.Upload() will return early before calling
        // client_.SendMessage().
        EXPECT_CALL(*mock_client_, SendMessage(_))
            .Times(2)
            .WillOnce(Return(absl::UnavailableError("failed to write 1")))
            .WillOnce(Return(absl::UnavailableError("failed to write 2")));
        // Will be called only once during mock_client_'s lifecycle.
        EXPECT_CALL(*mock_client_, GetMinWait())
            .Times(1)
            .WillOnce(Return(min_wait));
        return mock_client_;
      },
      nullptr,
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 2},
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 2},
      0);

  // 1st failed attempt.
  absl::Status status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  absl::Time current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.message(), StrEq("failed to write 1"));
  EXPECT_FALSE(uploader_->IsDead());
  double exponential_delay = 100;
  // the next time - current_time must be smaller than exponential_delay in
  // milliseconds
  EXPECT_LE(uploader_->NextUploadTime() - current_time,
            absl::Milliseconds(exponential_delay));
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // 2nd failed attempt.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.message(), StrEq("failed to write 2"));
  EXPECT_TRUE(uploader_->IsDead());
  EXPECT_EQ(uploader_->NextUploadTime(), absl::InfiniteFuture());

  // 3rd failed attempt, return internal error without calling the grpc client.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              StrEq("Uploader is already dead, don't retry."));
}

// Test the case when we have repeatedly CreateClient failure.
// After the # of failed attempts reaches the BackOffParameters.max_attempts,
// the uploader will give up on retry and be dead.
TEST_F(NcclUploaderTest, TestRepeatedlyCreateClientFailTillUploaderDeath) {
  int num_attempts_create_client = 0;
  auto uploader_ = std::make_unique<NcclUploader>(
      nullptr,
      [&num_attempts_create_client]() {
        ++num_attempts_create_client;
        return absl::UnavailableError("failed to create client");
      },
      nullptr,
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 2},
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 2},
      0);

  // 1st failed attempt.
  absl::Status status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  absl::Time current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.message(), StrEq("failed to create client"));
  EXPECT_FALSE(uploader_->IsDead());
  double exponential_delay = 100;
  // the next time - current_time must be smaller than exponential_delay in
  // milliseconds
  EXPECT_LE(uploader_->NextUploadTime() - current_time,
            absl::Milliseconds(exponential_delay));
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // 2nd failed attempt
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.message(), StrEq("failed to create client"));
  // uploader_ now is dead.
  EXPECT_TRUE(uploader_->IsDead());
  EXPECT_EQ(uploader_->NextUploadTime(), absl::InfiniteFuture());

  // 3rd attempt will return early with internal error without trying to create
  // the client.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              StrEq("Uploader is already dead, don't retry."));
  EXPECT_EQ(num_attempts_create_client, 2);
}

// Test the case when we have write failure in the middle and grpc client is
// always alive. And ensure that we have the right NextUploadTime() returned.
TEST_F(NcclUploaderTest, TestFailInMiddleOfMultipleWrites) {
  absl::Duration min_wait = absl::Milliseconds(50);
  auto uploader_ = std::make_unique<NcclUploader>(
      nullptr,
      [min_wait]() {
        std::unique_ptr<MockClient> mock_client_ =
            std::make_unique<MockClient>();
        // grpc client is always alive.
        EXPECT_CALL(*mock_client_, IsDead()).WillRepeatedly(Return(false));
        // 4 attempts of write in total: success, failure, failure, success
        EXPECT_CALL(*mock_client_, SendMessage(_))
            .WillOnce(Return(absl::OkStatus()))
            .WillOnce(Return(absl::UnavailableError("failed to write 1")))
            .WillOnce(Return(absl::UnavailableError("failed to write 2")))
            .WillOnce(Return(absl::OkStatus()));
        EXPECT_CALL(*mock_client_, GetMinWait())
            .Times(1)
            .WillOnce(Return(min_wait));
        return mock_client_;
      },
      nullptr,
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 3},
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 3},
      0);

  // First write succeeds. The next upload time should be min_wait + current.
  absl::Status status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_TRUE(status.ok());
  absl::Time current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_LE(uploader_->NextUploadTime() - current_time, min_wait);
  EXPECT_FALSE(uploader_->IsDead());
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // Second write failed. The next upload time will be calculated by exponential
  // backoff + current.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_FALSE(uploader_->IsDead());
  current_time = gpuviz::utils::get_current_monotonic_time();
  double exponential_backoff = 100;
  EXPECT_LE(uploader_->NextUploadTime() - current_time,
            absl::Milliseconds(exponential_backoff));
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // Third write failed again. The next upload time will be calculated by
  // exponential backoff + current.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_FALSE(uploader_->IsDead());
  current_time = gpuviz::utils::get_current_monotonic_time();
  exponential_backoff = 100 * std::pow(2, 1);
  EXPECT_LE(uploader_->NextUploadTime() - current_time,
            absl::Milliseconds(exponential_backoff));
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // Fourth write succeeds now. The next upload time should be min_wait +
  // current.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(uploader_->IsDead());
  EXPECT_LE(uploader_->NextUploadTime() - current_time, min_wait);
}

// Test the case when we have write failure in the middle because the grpc
// client is dead. After the grpc client is dead, the uploader should try to
// create the grpc client again.
TEST_F(NcclUploaderTest, TestClientDeadInMiddleOfMultipleWrites) {
  absl::Duration min_wait = absl::Milliseconds(50);
  int num_of_create_of_client = 0;
  auto uploader_ = std::make_unique<NcclUploader>(
      nullptr,
      [min_wait, &num_of_create_of_client]() {
        num_of_create_of_client++;
        // The first grpc client created, we try to write 3 times: 1 success, 2
        // & 3 failure.
        if (num_of_create_of_client == 1) {
          std::unique_ptr<MockClient> mock_client_ =
              std::make_unique<MockClient>();
          EXPECT_CALL(*mock_client_, IsDead())
              .WillOnce(Return(false))  // before 2nd write attempt
              .WillOnce(Return(false))  // after 2nd write failure
              .WillOnce(Return(false))  // before 3rd write attempt
              .WillOnce(Return(true));  // after 3rd write failure
          EXPECT_CALL(*mock_client_, SendMessage(_))
              .WillOnce(Return(absl::OkStatus()))
              .WillOnce(Return(absl::UnavailableError("failed to write 1")))
              .WillOnce(Return(absl::UnavailableError("failed to write 2")));
          EXPECT_CALL(*mock_client_, GetMinWait())
              .Times(1)
              .WillOnce(Return(min_wait));
          return mock_client_;
        }
        // The second grpc client created, we try to write 2 times, both
        // succeed.
        if (num_of_create_of_client == 2) {
          std::unique_ptr<MockClient> mock_client_ =
              std::make_unique<MockClient>();
          EXPECT_CALL(*mock_client_, GetMinWait())
              .Times(1)
              .WillOnce(Return(min_wait));
          EXPECT_CALL(*mock_client_, SendMessage(_))
              .Times(2)
              .WillRepeatedly(Return(absl::OkStatus()));
          EXPECT_CALL(*mock_client_, IsDead())
              .Times(1)  // before second write attempt
              .WillOnce(Return(false));
          return mock_client_;
        }
        return std::make_unique<MockClient>();
      },
      nullptr,
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 3},
      NcclUploader::BackOffParameters{
          .base_delay = 100, .base = 2, .cap = 2000, .max_attempts = 3},
      0);

  // On 1st client:
  // 1st write succeed
  absl::Status status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_TRUE(status.ok());
  absl::Time current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_LE(uploader_->NextUploadTime() - current_time, min_wait);
  EXPECT_FALSE(uploader_->IsDead());
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // 2nd write failure
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  current_time = gpuviz::utils::get_current_monotonic_time();
  double exponential_backoff = 100;
  EXPECT_LE(uploader_->NextUploadTime() - current_time,
            absl::Milliseconds(exponential_backoff));
  EXPECT_FALSE(uploader_->IsDead());
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // 3rd write failure
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  current_time = gpuviz::utils::get_current_monotonic_time();
  exponential_backoff = 100 * std::pow(2, 1);
  EXPECT_LE(uploader_->NextUploadTime() - current_time,
            absl::Milliseconds(exponential_backoff));
  // even though client is dead, but the uploader remains alive
  EXPECT_FALSE(uploader_->IsDead());
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // On 2nd client
  // 1st write succeed.
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(num_of_create_of_client, 2);
  current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_LE(uploader_->NextUploadTime() - current_time, min_wait);
  absl::SleepFor(uploader_->NextUploadTime() - current_time);

  // 2nd write succeed
  status = uploader_->Upload(GenerateFakeProtobuf("1", "2"));
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(num_of_create_of_client, 2);
  current_time = gpuviz::utils::get_current_monotonic_time();
  EXPECT_LE(uploader_->NextUploadTime() - current_time, min_wait);
}

}  // namespace
}  // namespace gpuviz

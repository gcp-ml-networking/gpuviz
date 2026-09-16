// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "utils.h"

#include <sys/stat.h>

#include <cassert>
#include <memory>
#include <queue>

#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "curl/curl.h"
#include "curl/easy.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "src/proto/nccl_telemetry_proto.pb.h"

#define assertm(exp, msg) assert(((void)msg, exp))

using google::protobuf::RepeatedPtrField;

// const of OK response code from http get request.
constexpr long kHttpGetOk = 200;
namespace {
// Util function borrowed from Google internal library.
// For internal user's reference, please search "repeated_field_util.h".
// Put in anonymous namespace in this file because they are not used outside
// now.

// Shrink the RepeatedPtrField internal array to the new_size.
template <typename T>
inline void Truncate(RepeatedPtrField<T>* array, int new_size) {
  const int size = array->size();
  DCHECK_GE(size, new_size);
  array->DeleteSubrange(new_size, size - new_size);
}

// Remove the entries in ReteaedPtrField internal array if Pred satisfied.
template <typename T, typename Pred>
int RemoveIf(RepeatedPtrField<T>* array, const Pred& pr) {
  T** const begin = array->mutable_data();
  T** const end = begin + array->size();
  T** write = begin;
  while (write < end && !pr(*write)) ++write;
  if (write == end) return 0;
  // 'write' is positioned at first element to be removed.
  for (T** scan = write + 1; scan < end; ++scan) {
    if (!pr(*scan)) std::swap(*scan, *write++);
  }
  Truncate(array, write - begin);
  return end - write;
}

// Callback function for curl to write the response to the output string.
// Input: contents: the response data, size: the size of each element, nmemb:
// the number of elements, output: the output string.
// Returns: the number of bytes processed.
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            std::string* output) {
  size_t total_size = size * nmemb;
  output->append((char*)contents, total_size);
  return total_size;
}

static absl::StatusOr<std::string> CurlHttpGet(const std::string& url,
                                               const std::string& header) {
  CURL* curl;
  CURLcode res;
  std::string read_buffer;
  curl = curl_easy_init();
  if (curl == nullptr) {
    ABSL_LOG(ERROR) << "Failed to initialize curl.";
    return absl::InternalError("Failed to initialize curl.");
  }

  // Set URL.
  res = curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  if (res != CURLE_OK) {
    ABSL_LOG(ERROR) << "Failed to set URL: " << url;
    curl_easy_cleanup(curl);
    return absl::InternalError(absl::StrCat(
        "Failed to set URL: ", url, " with error: ", curl_easy_strerror(res)));
  }

  // Set header.
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, header.c_str());
  res = curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  if (res != CURLE_OK || headers == nullptr) {
    ABSL_LOG(ERROR) << "Failed to set header: " << header;
    curl_easy_cleanup(curl);
    if (headers != nullptr) {
      curl_slist_free_all(headers);
    }
    return absl::InternalError(
        absl::StrCat("Failed to set header: ", header,
                     " with error: ", curl_easy_strerror(res)));
  }

  // Set the write callback function and its data.
  // No need to check the return value as they will both return CURLE_OK.
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &read_buffer);

  res = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  curl_slist_free_all(headers);
  if (res != CURLE_OK) {
    ABSL_LOG(ERROR) << "curl_easy_perform() failed: "
                    << curl_easy_strerror(res);
    return absl::InternalError(absl::StrCat(
        "curl_easy_perform() failed with error: ", curl_easy_strerror(res)));
  }

  // Parse the http get response code, following example:
  // https://curl.se/libcurl/c/CURLINFO_RESPONSE_CODE.html
  long response_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
  if (response_code != kHttpGetOk) {
    return absl::InternalError(
        absl::StrFormat("Get a non-OK response code: %d when performing http "
                        "get request of %s, with full output: %s",
                        response_code, url, read_buffer));
  }
  ABSL_VLOG(1) << "Got metadata for key: " << url
               << " and its value is: " << read_buffer;
  return read_buffer;
}
}  // namespace

namespace gpuviz {
namespace utils {
absl::Time get_current_monotonic_time() {
  struct timespec ts;
  [[maybe_unused]] int result = clock_gettime(CLOCK_MONOTONIC, &ts);
  assertm(result != -1, "clock_gettime failed!");
  return absl::TimeFromTimespec(ts);
}

uint64_t get_current_monotonic_time_milliseconds() {
  struct timespec ts;
  [[maybe_unused]] int result = clock_gettime(CLOCK_MONOTONIC, &ts);
  assertm(result != -1, "clock_gettime failed!");
  uint64_t milliseconds =
      (uint64_t)ts.tv_sec * SECOND_TO_MILLISECOND_MULTIPLIER;
  milliseconds += (ts.tv_nsec / MILLISECOND_TO_NANOSECOND_MULTIPLIER);
  return milliseconds;
}

std::pair<std::string, std::string> ConvertSockaddrStorageToString(
    const sockaddr_storage& sockaddr_storage) {
  // This should be large enough for both IPv4 and IPv6
  char ipString[INET6_ADDRSTRLEN];

  if (sockaddr_storage.ss_family == AF_INET) {
    // IPv4
    const sockaddr_in* sockaddr_in_ptr =
        reinterpret_cast<const sockaddr_in*>(&sockaddr_storage);
    inet_ntop(AF_INET, &(sockaddr_in_ptr->sin_addr), ipString, INET_ADDRSTRLEN);
    uint16_t port = ntohs(sockaddr_in_ptr->sin_port);
    return std::pair<std::string, std::string>(std::string(ipString),
                                               std::to_string(port));
  } else if (sockaddr_storage.ss_family == AF_INET6) {
    // IPv6
    const sockaddr_in6* sockaddr_in6_ptr =
        reinterpret_cast<const sockaddr_in6*>(&sockaddr_storage);
    inet_ntop(AF_INET6, &(sockaddr_in6_ptr->sin6_addr), ipString,
              INET6_ADDRSTRLEN);
    uint16_t port = ntohs(sockaddr_in6_ptr->sin6_port);
    return std::pair<std::string, std::string>(std::string(ipString),
                                               std::to_string(port));
  } else {
    return std::pair<std::string, std::string>("", "");
  }
}
std::string getEndpointString(std::pair<std::string, std::string> ip_port) {
  return ip_port.first + ":" + ip_port.second;
}

// Keep the protoDist::StatsDistribution of msg iff the dist_type of the
// histogram is in the unordered_set.
void filterHistogramsByDistType(
    std::unique_ptr<protoDist::AllStats>& msg,
    const std::unordered_set<protoDist::DistType>& dist_types) {
  for (protoDist::ConnectionStats& conn_stats : *msg->mutable_conn_stats()) {
    RemoveIf(conn_stats.mutable_histogram(),
             [&dist_types](const protoDist::StatsDistribution* entry) {
               return !entry->has_dist_type() ||
                      dist_types.find(entry->dist_type()) == dist_types.end();
             });
  }
}

absl::StatusOr<std::string> getMdsAttributeByKey(absl::string_view key) {
  return CurlHttpGet(
      absl::StrCat("http://metadata.google.internal/computeMetadata/v1/", key),
      "Metadata-Flavor: Google");
}

std::string getProcessIdentifier() {
  struct stat proc_self_stat;
  int ret_val = stat("/proc/self/", &proc_self_stat);
  if (ret_val != 0) {
    // Process start time is not available, replace with 0 instead
    proc_self_stat.st_mtime = 0;
  }
  return absl::StrFormat(
      "%d:%u:%u", getpid(),
      static_cast<uint64_t>(proc_self_stat.st_mtim.tv_sec) *
              SECOND_TO_MICROSECOND_MULTIPLIER +
          proc_self_stat.st_mtim.tv_nsec / MICROSECOND_TO_NANOSECOND_MULTIPLIER,
      absl::Uniform<uint64_t>(absl::BitGen(), 0,
                              std::numeric_limits<uint64_t>::max()));
}
}  // namespace utils
}  // namespace gpuviz
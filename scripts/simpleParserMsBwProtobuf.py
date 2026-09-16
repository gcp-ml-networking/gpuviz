# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import argparse
from collections import defaultdict
import copy
from datetime import datetime
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# To generate millisecond_bw_file_format_pb2.py, download and install protobuf
# from protobuf.dev/downloads/ and run the following command:
# protoc --proto_path=$PWD --python_out=$PWD millisecond_bw_file_format.proto
sys.path.append(os.getcwd() + "/../build/src/proto")
import millisecond_bw_file_format_pb2


FILE_NAME_PREFIX = "millisecond_bandwidth_"
FILE_NAME_SUFFIX = ".log"
PARSED_CONNECTIONS_FILE_NAME = FILE_NAME_PREFIX + "parsed" + FILE_NAME_SUFFIX

UNIT_TO_NANO = 10**9
MILLI_TO_NANO = 10**6

DIRECTIONS = ["Tx", "Rx"]


def get_time_stamp_sec(time_stamp):
  return time_stamp.seconds + time_stamp.nanos / UNIT_TO_NANO


def get_start_and_end_datetime(raw_start_time, interval_ms):
  start_datetime = datetime.fromtimestamp(raw_start_time)
  raw_end_time = raw_start_time + interval_ms * MILLI_TO_NANO / UNIT_TO_NANO
  end_datetime = datetime.fromtimestamp(raw_end_time)
  return start_datetime, end_datetime


def setup_nic_bw_heatmap_data(start_time, nic_data, direction):
  list_nic_data_values = [
      nic_datum[direction] for nic_datum in nic_data.values()
  ]
  heatmap_data = np.array(list_nic_data_values)
  yticks = [ip_addr for ip_addr in nic_data.keys()]
  heatmap = sn.heatmap(heatmap_data, yticklabels=yticks, cmap="viridis")
  # Interval is the length of Tx/Rx bandwidth data of each NIC Data
  interval_ms = len(list_nic_data_values[0])
  start_datetime, end_datetime = get_start_and_end_datetime(
      start_time, interval_ms
  )
  plt.xticks(ticks=[0, interval_ms], labels=[start_datetime, end_datetime])
  plt.yticks(rotation=0)
  plt.subplots_adjust(bottom=0.5)


def plot_nic_bw_heatmap_figs(fig_dir, connections):
  fig_idx = 0
  for dict_batch in connections.values():
    if not dict_batch["nic_data"]:
      print(
          "Warning: No heatmap is plotted for this connection because NIC data"
          " is empty."
      )
      continue

    fig_idx += 1
    for direction in DIRECTIONS:
      plt.figure(figsize=(10, 10))
      setup_nic_bw_heatmap_data(
          dict_batch["time_stamp"], dict_batch["nic_data"], direction
      )
      plt.xlabel("Time")
      plt.ylabel("NIC IP Address")
      plt.title(direction + " Bandwidth (Bytes per Second)")
      filename = os.path.join(
          fig_dir,
          "heatmap_" + str(fig_idx) + "_" + direction + "_time_series.png",
      )
      plt.savefig(filename)
      plt.close()


def setup_nic_bw_line_chart_data(start_time, nic_data, direction):
  # Interval is the length of Tx/Rx bandwidth data of each NIC Data
  dict_nic_data = {
      ip_addr: nic_datum[direction] for ip_addr, nic_datum in nic_data.items()
  }
  for ip_addr, nic_datum in dict_nic_data.items():
    plt.plot(nic_datum, label=ip_addr)
  interval_ms = len(list(dict_nic_data.values())[0])
  start_datetime, end_datetime = get_start_and_end_datetime(
      start_time, interval_ms
  )
  plt.xticks(ticks=[0, interval_ms], labels=[start_datetime, end_datetime])
  plt.legend()
  plt.subplots_adjust(bottom=0.5)


def plot_nic_bw_line_chart_figs(fig_dir, connections):
  fig_idx = 0
  for dict_batch in connections.values():
    if not dict_batch["nic_data"]:
      print(
          "Warning: No line chart is plotted for this connection because NIC"
          " data is empty."
      )
      continue

    fig_idx += 1
    for direction in DIRECTIONS:
      plt.figure(figsize=(10, 10))
      setup_nic_bw_line_chart_data(
          dict_batch["time_stamp"], dict_batch["nic_data"], direction
      )
      plt.xlabel("Time")
      plt.ylabel("Bandwidth")
      plt.title(direction + " Bandwidth (Bytes per Second)")
      filename = os.path.join(
          fig_dir,
          "line_chart_" + str(fig_idx) + "_" + direction + "_time_series.png",
      )
      plt.savefig(filename)
      plt.close()


def setup_per_nic_bw_line_chart_data(start_time, ip_addr, nic_datum, direction):
  # Interval is the length of Tx/Rx bandwidth data of each NIC Data
  interval_ms = len(nic_datum[direction])
  start_datetime, end_datetime = get_start_and_end_datetime(
      start_time, interval_ms
  )
  plt.plot(nic_datum[direction], label=ip_addr)
  plt.xticks(ticks=[0, interval_ms], labels=[start_datetime, end_datetime])
  plt.legend()
  plt.subplots_adjust(bottom=0.5)


def plot_per_nic_bw_line_chart_figs(fig_dir, connections):
  for dict_batch in connections.values():
    for ip_addr, nic_datum in dict_batch["nic_data"].items():
      for direction in DIRECTIONS:
        plt.figure(figsize=(10, 10))
        setup_per_nic_bw_line_chart_data(
            dict_batch["time_stamp"], ip_addr, nic_datum, direction
        )
        plt.xlabel("Time")
        plt.ylabel("Bandwidth")
        plt.title(direction + " Bandwidth (Bytes per Second)")
        filename = os.path.join(
            fig_dir,
            "line_chart_per_nic"
            + ip_addr
            + "_"
            + direction
            + "_time_series.png",
        )
        plt.savefig(filename)
        plt.close()


def setup_nic_bw_every_1sec_line_chart_data(
    start_time, nic_data, direction, time_idx
):
  dict_nic_data = {
      ip_addr: nic_datum[direction][time_idx * 1000 : (time_idx + 1) * 1000]
      for ip_addr, nic_datum in nic_data.items()
  }
  for ip_addr, nic_datum in dict_nic_data.items():
    plt.plot(nic_datum, label=ip_addr)
  # Interval is the length of Tx/Rx bandwidth data of each NIC Data
  interval_ms = len(list(dict_nic_data.values())[0])
  start_datetime, end_datetime = get_start_and_end_datetime(
      start_time + time_idx, interval_ms
  )
  plt.xticks(ticks=[0, interval_ms], labels=[start_datetime, end_datetime])
  plt.legend()
  plt.subplots_adjust(bottom=0.5)


def plot_nic_bw_every_1sec_line_chart_figs(fig_dir, connections):
  for dict_batch in connections.values():
    if not dict_batch["nic_data"]:
      print(
          "Warning: No line chart is plotted for this connection because NIC"
          " data is empty."
      )
      continue

    size_of_nic_datum = len(list(dict_batch["nic_data"].values())[0]["Tx"])
    num_of_seconds = size_of_nic_datum // 1000
    if size_of_nic_datum % 1000 > 0:
      num_of_seconds += 1
    for time_idx in range(num_of_seconds):
      for direction in DIRECTIONS:
        plt.figure(figsize=(10, 10))
        setup_nic_bw_every_1sec_line_chart_data(
            dict_batch["time_stamp"],
            dict_batch["nic_data"],
            direction,
            time_idx,
        )
        plt.xlabel("Time")
        plt.ylabel("Bandwidth")
        plt.title(direction + " Bandwidth (Bytes per Second)")
        filename = os.path.join(
            fig_dir,
            "line_chart_time"
            + str(time_idx)
            + "sec"
            + "_"
            + direction
            + "_time_series.png",
        )
        plt.savefig(filename)
        plt.close()


def setup_every_5nic_bw_every_1sec_line_chart_data(
    start_time, nic_data, direction, time_idx, ip_addr_list
):
  dict_nic_data = {
      ip_addr: nic_data[ip_addr][direction][
          time_idx * 1000 : (time_idx + 1) * 1000
      ]
      for ip_addr in ip_addr_list
  }
  for ip_addr, nic_datum in dict_nic_data.items():
    plt.plot(nic_datum, label=ip_addr)
  # Interval is the length of Tx/Rx bandwidth data of each NIC Data
  interval_ms = len(list(dict_nic_data.values())[0])
  start_datetime, end_datetime = get_start_and_end_datetime(
      start_time + time_idx, interval_ms
  )
  plt.xticks(ticks=[0, interval_ms], labels=[start_datetime, end_datetime])
  plt.legend()
  plt.subplots_adjust(bottom=0.5)


def plot_every_5nic_bw_every_1sec_line_chart_figs(fig_dir, connections):
  num_of_ip_addr_per_group = 5
  for dict_batch in connections.values():
    if not dict_batch["nic_data"]:
      print(
          "Warning: No line chart is plotted for this connection because NIC"
          " data is empty."
      )
      continue

    size_of_nic_datum = len(list(dict_batch["nic_data"].values())[0]["Tx"])
    num_of_seconds = size_of_nic_datum // 1000
    if size_of_nic_datum % 1000 > 0:
      num_of_seconds += 1

    ip_addr_list = list(dict_batch["nic_data"].keys())
    num_of_ip_groups = len(ip_addr_list) // num_of_ip_addr_per_group
    if len(ip_addr_list) % num_of_ip_addr_per_group > 0:
      num_of_ip_groups += 1

    for time_idx in range(num_of_seconds):
      for ip_addr_group_idx in range(num_of_ip_groups):
        for direction in DIRECTIONS:
          plt.figure(figsize=(10, 10))
          setup_every_5nic_bw_every_1sec_line_chart_data(
              dict_batch["time_stamp"],
              dict_batch["nic_data"],
              direction,
              time_idx,
              ip_addr_list[
                  ip_addr_group_idx
                  * num_of_ip_addr_per_group : (ip_addr_group_idx + 1)
                  * num_of_ip_addr_per_group
              ],
          )
          plt.xlabel("Time")
          plt.ylabel("Bandwidth")
          plt.title(direction + " Bandwidth (Bytes per Second)")
          filename = os.path.join(
              fig_dir,
              "line_chart_time"
              + str(time_idx)
              + "sec"
              + "_"
              + ip_addr_list[ip_addr_group_idx * num_of_ip_addr_per_group]
              + "_"
              + direction
              + "_time_series.png",
          )
          plt.savefig(filename)
          plt.close()


def parse_data(data):
  ms_bw_info = millisecond_bw_file_format_pb2.MillisecondBandwidthInfo()
  delimiter = b"==END=="
  list_batches = []

  # The last item is either an empty string or a partially written protobuf
  # serialization.
  messages = data.split(delimiter)[:-1]

  for msg_data in messages:
    try:
      ms_bw_info.ParseFromString(msg_data)
      for batch in ms_bw_info.batches:
        dict_batch = defaultdict(lambda: defaultdict(int))
        dict_batch["process_ids"] = [batch.process_id]
        dict_batch["time_stamp"] = get_time_stamp_sec(batch.time_stamp)
        dict_batch["nic_data"] = defaultdict(lambda: defaultdict(int))
        dict_nic_data = dict_batch["nic_data"]
        for nic_datum in batch.nic_data:
          dict_nic_data[nic_datum.ip_address] = defaultdict(
              lambda: defaultdict(int)
          )
          # Change the unit from per-millisecond to per-second
          dict_nic_data[nic_datum.ip_address]["Tx"] = [
              data * UNIT_TO_NANO / MILLI_TO_NANO
              for data in nic_datum.tx_bandwidth_data
          ]
          dict_nic_data[nic_datum.ip_address]["Rx"] = [
              data * UNIT_TO_NANO / MILLI_TO_NANO
              for data in nic_datum.rx_bandwidth_data
          ]
        list_batches.append(dict_batch)
    except:
      pass
  return list_batches


def get_filename(path):
  # Target the file name instead of any directories with the same prefix.
  prefix_idx = path.rfind(FILE_NAME_PREFIX)
  return path[prefix_idx + len(FILE_NAME_PREFIX) : -len(FILE_NAME_SUFFIX)]


def read_data(directory):
  # Main dictionary to hold data aggregated per connection.
  connections = defaultdict(lambda: defaultdict(list))

  pattern = os.path.join(directory, FILE_NAME_PREFIX + "*" + FILE_NAME_SUFFIX)
  txt_log_files = glob.glob(pattern)

  for path in txt_log_files:
    data = open(path, "rb").read()
    file_name = get_filename(path)
    connections[file_name] = parse_data(data)

  return connections


def print_parsed_log_to_file(parsed_log_dir, connections):
  if not parsed_log_dir:
    return

  if not os.path.exists(parsed_log_dir):
    os.makedirs(parsed_log_dir)
  parsed_log_filename = parsed_log_dir + "/" + PARSED_CONNECTIONS_FILE_NAME
  with open(parsed_log_filename, "w") as f:
    json_output = json.dumps(connections, indent=2)
    print(json_output, file=f)


def merge_batches_in_the_same_process(connections):
  merged_connections = defaultdict(lambda: defaultdict(int))
  for file_name, list_batches in connections.items():
    rotation_idx_delimiter_pos = file_name.rfind("_")
    rotation_idx = int(file_name[rotation_idx_delimiter_pos + 1 :])
    file_name_trimmed = file_name[:rotation_idx_delimiter_pos]

    if rotation_idx == 0:
      # Initialize the merged batch dictionary with an empty dictionary
      dict_batch = defaultdict(lambda: defaultdict(list))
      dict_batch["process_ids"] = list_batches[0]["process_ids"]
      dict_batch["time_stamp"] = list_batches[0]["time_stamp"]
      dict_batch["nic_data"] = defaultdict(lambda: defaultdict(list))
    else:
      # Get the existing merged batch dictionary for the same batch list parsed
      # in the previous rotation log file.
      dict_batch = merged_connections[file_name_trimmed]

    for batch in list_batches:
      for ip_address, dict_nic_datum in batch["nic_data"].items():
        if not ip_address in dict_batch["nic_data"].keys():
          dict_batch["nic_data"][ip_address] = copy.deepcopy(dict_nic_datum)
          continue
        for direction in DIRECTIONS:
          dict_batch["nic_data"][ip_address][direction] += dict_nic_datum[
              direction
          ]
    merged_connections[file_name_trimmed] = dict_batch
  return merged_connections


def get_intervals_across_connections(connections):
  """Calculates and merges time intervals from all NICs data in all connections.

  This function extracts time intervals for all NICs in all connections based
  on the timestamp and the maximum Tx data length across NICs. The function then
  merges overlapping or adjacent intervals to produce a consolidated list of
  non-overlapping time intervals.

  Args:
    connections (dict): A dictionary representing network connections. The keys
      are connection identifiers, and the values are dictionaries containing: -
      "time_stamp" (int): The starting timestamp of the connection in seconds. -
      "nic_data" (dict): A dictionary of NIC data, where keys are NIC
      identifiers, and values are dictionaries containing "Tx" (list) data.

  Returns:
    list: A list of tuples, where each tuple represents a non-overlapping time
      interval (start_sec, end_sec) in seconds.
  """
  intervals = []
  for dict_batch in connections.values():
    start_sec = dict_batch["time_stamp"]
    max_len_ms = max(
        len(nic_datum.get("Tx", []))
        for nic_datum in dict_batch["nic_data"].values()
    )
    end_sec = start_sec + max_len_ms * MILLI_TO_NANO / UNIT_TO_NANO
    intervals.append((start_sec, end_sec))
  if not intervals:
    return []

  # Sort the intervals by their start time
  intervals.sort(key=lambda x: x[0])
  # Merge the overlapping/adjacent intervals
  merged_intervals = []
  # Start with the first interval
  current_start, current_end = intervals[0]

  for next_start, next_end in intervals[1:]:
    # If the current interval overlaps with or is adjacent to the next one
    if next_start <= current_end:
      # Merge by taking the maximum end time
      current_end = max(current_end, next_end)
    else:
      # Found a gap, save the merged interval and start a new one
      merged_intervals.append((current_start, current_end))
      current_start, current_end = next_start, next_end

  # Add the last merged interval
  merged_intervals.append((current_start, current_end))

  return merged_intervals


def merge_batches_with_overlapping_time_interval(connections):
  intervals = get_intervals_across_connections(connections)
  merged_connections = defaultdict(lambda: defaultdict(int))
  for start, _ in intervals:
    merged_connections[start] = defaultdict(lambda: defaultdict(int))
    merged_connections[start]["process_ids"] = []
    merged_connections[start]["time_stamp"] = start
    merged_connections[start]["nic_data"] = defaultdict(
        lambda: defaultdict(int)
    )

  for dict_batch in connections.values():
    timestamp = dict_batch["time_stamp"]
    list_process_id = dict_batch["process_ids"]
    nic_data = dict_batch["nic_data"]
    for start, end in intervals:
      if timestamp < end:
        dict_time_data = merged_connections[start]
        dict_time_data["process_ids"] += list_process_id
        for ip_addr, nic_datum in nic_data.items():
          nic_interval = len(nic_datum["Tx"])
          diff_before_start = int(
              round(timestamp - start, 3) * UNIT_TO_NANO / MILLI_TO_NANO
          )
          diff_after_end = int(
              (round(end - timestamp, 3) * UNIT_TO_NANO // MILLI_TO_NANO)
              - nic_interval
          )
          new_tx_data = (
              [0] * diff_before_start + nic_datum["Tx"] + [0] * diff_after_end
          )
          new_rx_data = (
              [0] * diff_before_start + nic_datum["Rx"] + [0] * diff_after_end
          )
          if not ip_addr in dict_time_data["nic_data"].keys():
            dict_time_data["nic_data"][ip_addr] = defaultdict(
                lambda: defaultdict(int)
            )
            dict_time_data["nic_data"][ip_addr]["Tx"] = new_tx_data
            dict_time_data["nic_data"][ip_addr]["Rx"] = new_rx_data
          else:
            for idx in range(len(new_tx_data)):
              dict_time_data["nic_data"][ip_addr]["Tx"][idx] += new_tx_data[idx]
              dict_time_data["nic_data"][ip_addr]["Rx"][idx] += new_rx_data[idx]
        break

  return merged_connections


def merge_connections(connections):
  merged_connections = merge_batches_in_the_same_process(connections)
  merged_connections = merge_batches_with_overlapping_time_interval(
      merged_connections
  )
  return merged_connections


def print_merged_data_to_console(connections):
  json_output = json.dumps(connections, indent=2)
  print(json_output)


def main(args):
  directory = args.source_dir
  fig_dir = args.fig_dir + "/"
  parsed_log_dir = args.parsed_log_dir
  connections = read_data(directory)

  print_parsed_log_to_file(parsed_log_dir, connections)
  merged_connections = merge_connections(connections)
  print_merged_data_to_console(merged_connections)

  # Plot a heat-map for every connection.
  plot_nic_bw_heatmap_figs(fig_dir, merged_connections)

  # Plot a line-chart for every connection.
  plot_nic_bw_line_chart_figs(fig_dir, merged_connections)

  # Plot a line-chart for each IP address
  plot_per_nic_bw_line_chart_figs(fig_dir, merged_connections)

  # Plot a line-chart for every connection every 1 second
  plot_nic_bw_every_1sec_line_chart_figs(fig_dir, merged_connections)

  # Plot a line-chart for every 5 IP address every 1 second
  plot_every_5nic_bw_every_1sec_line_chart_figs(fig_dir, merged_connections)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--source_dir",
      type=str,
      default="/tmp",
      help="Directory where log files are stored.",
  )
  parser.add_argument(
      "--parsed_log_dir",
      type=str,
      default="",
      help="Directory where parsed millisecond bandwidth data log are stored.",
  )
  parser.add_argument(
      "--fig_dir",
      type=str,
      default="/tmp/figures",
      help="Directory where figures are stored.",
  )
  main(parser.parse_args())

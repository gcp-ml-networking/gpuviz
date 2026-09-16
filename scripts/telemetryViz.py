# Copyright 2026 Google LLC
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import argparse
import glob
import itertools
import os
import sys
import matplotlib.markers as mmarkers
import pandas as pd

sys.path.append(os.getcwd() + "/../build/src/proto")
import nccl_telemetry_proto_pb2
from nccl_telemetry_proto_pb2 import DistType
from nccl_telemetry_proto_pb2 import ncclStatsConnectionType
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from collections import namedtuple

# Label frequency for time-series plots.
# Display every third label (adjust condition as needed).
YLABEL_FREQUENCY = 3  # y-axis labels i.e. bucket bounds
XLABEL_FREQUENCY = 1  # x-axis labels i.e. time stamps.

# Filtering out the lowest 10**-5th percentile in CCDF Plots
PERCENTILE_THRESHOLD = 10**-5

# Create estimated bw graphs for these distribution types
EST_BW_FOR_DIST = [DistType.protoSendMessageSize, DistType.protoRecvMessageSize]

NANOSECONDS_TO_MICROSECONDS = 10**-3
BYTES_TO_GBITS = 8 * 10**-9
TIME_UNIT = "Nanoseconds"

UNIT_LABELS = {
    DistType.protoSendLatencySW: TIME_UNIT,
    DistType.protoRecvLatencySW: TIME_UNIT,
    DistType.protoSendLatencyNetHW: TIME_UNIT,
    DistType.protoRecvLatencyNetHW: TIME_UNIT,
    DistType.protoRecvReadylatency: TIME_UNIT,
    DistType.protoSendMessageSize: "Bytes",
    DistType.protoRecvMessageSize: "Bytes",
    DistType.protoSendBandwidth: "Bytes/s",
    DistType.protoRecvBandwidth: "Bytes/s",
    DistType.protoBandwidthVariance: "Bytes/s",
    DistType.protoLatencyVariance: TIME_UNIT,
}


ALL_MARKER_STYLES = [
    m for m in mmarkers.MarkerStyle.markers.keys() if isinstance(m, str)
]
MARKER_CYCLE = itertools.cycle(ALL_MARKER_STYLES)

# Types of distributions/histograms we expect
DIST_TYPE_LABELS = {
    DistType.protoSendLatencySW: "Software Send Latency",
    DistType.protoRecvLatencySW: "Software Receive Latency",
    DistType.protoSendLatencyNetHW: "Network Hardware Send Latency",
    DistType.protoRecvLatencyNetHW: "Network Hardware Receive Latency",
    DistType.protoRecvReadylatency: "Receive Ready Latency",
    DistType.protoSendMessageSize: "Send Message Size",
    DistType.protoRecvMessageSize: "Receive Message Size",
    DistType.protoSendBandwidth: "High Frequency Send Bandwidth",
    DistType.protoRecvBandwidth: "High Frequency Recv Bandwidth",
    DistType.protoBandwidthVariance: "High Frequency Bandwidth Variance",
    DistType.protoLatencyVariance: "High Frequency Latency Variance",
}


ConnectionData = namedtuple(
    "ConnectionData", ["time_stamp", "counts", "bounds", "min", "max", "avg"]
)


# Prints all connection identifier data.
def print_conn_id(conn):
  print(
      "================== Connection Identification Information"
      " ========================="
  )
  conn_id = conn.connection_id
  print("Plug in name:", conn_id.nccl_plugin_name)
  print("GPU PCI ADDR:", conn_id.gpu_pci_addr)
  # print(conn_id.nccl_plugin_type)
  conn_info = conn_id.connection
  # print(conn_info.conn_type)
  extra_local, extra_remote = "", ""
  if conn_info.conn_type == ncclStatsConnectionType.EntityRDMAConnection:
    conn_type = "RDMA"
    stats_conn = conn_info.rdma_conn
    extra_local, extra_remote = (
        f" (QPN: {stats_conn.local_qpn})",
        f"(QPN: {stats_conn.remote_qpn})",
    )
  elif conn_info.conn_type == ncclStatsConnectionType.EntityTCPConnection:
    conn_type = "TCP"
    stats_conn = conn_info.tcp_conn
  else:
    conn_type = ""
  print("Connection Type:", conn_type)
  print("Local End-point:", stats_conn.local_endpoint, extra_local)
  print("Remote End-point:", stats_conn.remote_endpoint, extra_remote)
  print("Time Stamp:", datetime.fromtimestamp(conn.time_stamp.seconds))
  print("================== End Information =========================")


# The string concatenation of local and remote end-point is used as key to store data for a particular connection in a dictionary.
def get_connection_name(conn):
  conn_type = conn.connection_id.connection.conn_type
  if conn_type == ncclStatsConnectionType.EntityNVLConnection:
    # TODO
    return ""
  elif conn_type == ncclStatsConnectionType.EntityPCIConnection:
    # TODO
    return ""
  elif conn_type == ncclStatsConnectionType.EntityTCPConnection:
    connection = conn.connection_id.connection.tcp_conn
  elif conn_type == ncclStatsConnectionType.EntityRDMAConnection:
    connection = conn.connection_id.connection.rdma_conn
  else:
    # TODO
    return ""
  return connection.local_endpoint + "_" + connection.remote_endpoint


# For the given connection add data for current time-stamp.
def add_timestamped_data(connections, conn):
  name = get_connection_name(conn)
  for hist in conn.histogram:
    max_bucket = hist.max_bucket
    base = hist.base
    scale = hist.scale_factor
    connections[name][hist.dist_type].append(
        ConnectionData(
            conn.time_stamp.seconds + conn.time_stamp.nanos / 10**9,
            [curr for curr in hist.bucket_counts],
            [(base**i) * scale for i in range(len(hist.bucket_counts))],
            hist.min,
            hist.max,
            hist.avg,
        )
    )


def parse_data(connections, data):
  all_stats = nccl_telemetry_proto_pb2.AllStats()
  delimiter = b"==END=="
  dict_ips = defaultdict(lambda: defaultdict(int))
  unique_conns = set()

  messages = data.split(delimiter)[:-1]
  for msg_data in messages:
    try:
      all_stats.ParseFromString(msg_data)
      for conn in all_stats.conn_stats:
        conns_in_file(unique_conns, dict_ips, conn)
        add_timestamped_data(connections, conn)
    except:
      pass
  return dict_ips


def get_ip_addr(ip_port):
  return ip_port.rsplit(":", 1)[0]


def conns_in_file(unique_conns, dict_ips, conn):
  name = get_connection_name(conn)
  (src, dst) = [get_ip_addr(x) for x in name.split("_")]

  key = src + "_" + dst
  if key not in unique_conns:
    unique_conns.add(key)

    dict_ips["Sources"][src] += 1
    dict_ips["Destinations"][dst] += 1


def write_ip_data(path, dict_ips):
  directory = os.path.dirname(path)
  f = open(os.path.join(directory, "conns_per_process.csv"), "a")
  f.write(path.split("/")[-1] + "\n")

  for key, value in dict_ips.items():
    df = pd.DataFrame(
        list(value.items()), columns=[key + " IPs", "Number of Connections"]
    )
    df.to_csv(f, index=False)
  f.close()


def read_data(directory):
  # Main dictionary to hold data aggregated per connection.
  connections = defaultdict(lambda: defaultdict(list))

  pattern = os.path.join(directory, "exporter*.log")
  txt_log_files = glob.glob(pattern)

  for path in txt_log_files:
    data = open(path, "rb").read()
    dict_ips = parse_data(connections, data)
    if dict_ips:
      write_ip_data(path, dict_ips)

  for name in connections.keys():
    for dist_type in DIST_TYPE_LABELS.keys():
      connections[name][dist_type].sort(key=lambda x: x.time_stamp)

  return connections


# For a given distribution of type dist_type make a heat-map or time-series chart showing change in distribution over time.
def make_time_series_plot(list_of_timestamped_data):
  times = [
      datetime.fromtimestamp(el.time_stamp) for el in list_of_timestamped_data
  ]
  data = np.array([hist.counts[::-1] for hist in list_of_timestamped_data])

  yticks = [int(el) for el in list_of_timestamped_data[0].bounds[::-1]]
  heatmap = sn.heatmap(data.T, yticklabels=yticks, cmap="viridis")

  yticklabels = heatmap.get_yticklabels()
  for i, label in enumerate(yticklabels):
    if i % YLABEL_FREQUENCY != 0:
      label.set_visible(False)

  xlabels = [times[i] for i in range(0, len(times), XLABEL_FREQUENCY)]
  plt.xticks(
      ticks=range(0, len(times), XLABEL_FREQUENCY), labels=xlabels, rotation=90
  )
  plt.subplots_adjust(bottom=0.5)


# Make CDF Plot
def CDF_plot(list_of_timestamped_data):
  frequencies = sum(
      [np.array(hist.counts) for hist in list_of_timestamped_data]
  )

  est_latencies = []
  bucket_low = 0
  for bucket_high in list_of_timestamped_data[0].bounds:
    bucket_avg = (bucket_high - bucket_low) / 2.0
    bucket_avg *= NANOSECONDS_TO_MICROSECONDS
    est_latencies.append(bucket_avg)
    bucket_low = bucket_high

  cdf = np.cumsum(frequencies) / sum(frequencies)
  ccdf = 1 - cdf

  percentile_threshold = np.percentile(ccdf, PERCENTILE_THRESHOLD)

  filtered_latencies = [
      est_latencies[i]
      for i in range(len(est_latencies))
      if ccdf[i] > percentile_threshold
  ]
  filtered_ccdf = [ci for ci in ccdf if ci > percentile_threshold]

  marker = next(MARKER_CYCLE)
  plt.plot(filtered_latencies, filtered_ccdf, marker=marker)


def bandwidth_plot(list_of_timestamped_data):
  data_per_sec = []
  times = []

  prev_time = list_of_timestamped_data[0].time_stamp
  total_data = 0

  for i in range(1, len(list_of_timestamped_data)):
    el = list_of_timestamped_data[i]
    curr_time = el.time_stamp

    avg_msg_size = el.avg * BYTES_TO_GBITS
    total_data += sum(el.counts) * avg_msg_size
    if curr_time != prev_time:
      times.append(datetime.fromtimestamp(curr_time))
      data_per_sec.append(total_data / (curr_time - prev_time))

      prev_time = curr_time
      total_data = 0

  plt.plot(times, data_per_sec)


def all_conns_in_one_plot(connections, dist_type, plotting_func):
  labels = []
  for key in connections.keys():
    if connections[key][dist_type]:
      plotting_func(connections[key][dist_type])
      labels.append(key)
  return labels


def make_conn_stat_plot(list_of_timestamped_data, stat_type):
  times = [datetime.fromtimestamp(el[0]) for el in list_of_timestamped_data]
  data = [getattr(el, stat_type) for el in list_of_timestamped_data]
  plt.plot(times, data)


def time_series_figs(fig_dir, connections):
  for key in connections.keys():
    for dist_type in DIST_TYPE_LABELS.keys():
      if connections[key][dist_type]:
        # print("Connection Id:", key)
        plt.figure(figsize=(10, 10))
        make_time_series_plot(connections[key][dist_type])
        plt.xlabel("Time")
        plt.ylabel(UNIT_LABELS[dist_type])
        plt.title(DIST_TYPE_LABELS[dist_type])
        filename = os.path.join(
            fig_dir, key + "_" + str(dist_type) + "_time_series.png"
        )
        plt.savefig(filename)
        plt.close()


def cdf_figs(fig_dir, connections):
  for dist_type in DIST_TYPE_LABELS.keys():

    labels = all_conns_in_one_plot(connections, dist_type, CDF_plot)

    if labels:
      plt.xlabel(UNIT_LABELS[dist_type])
      plt.ylabel("Cumulative Probability")
      plt.yscale("log")
      plt.title(DIST_TYPE_LABELS[dist_type])
      plt.grid(True)
      plt.legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
      filename = os.path.join(fig_dir, str(dist_type) + "_CCDF.png")
      plt.savefig(filename, bbox_inches="tight")
      plt.clf()


def bandwidth_figs(fig_dir, connections):
  for dist_type in EST_BW_FOR_DIST:
    labels = all_conns_in_one_plot(connections, dist_type, bandwidth_plot)
    if labels:
      plt.xticks(rotation=90)
      plt.xlabel("Time")
      plt.ylabel("Bandwidth in Gb/sec")

      plt.title("Bandwidth of " + DIST_TYPE_LABELS[dist_type])
      plt.legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
      filename = os.path.join(fig_dir, str(dist_type) + "_bandwidth.png")
      plt.savefig(filename, bbox_inches="tight")
      plt.clf()


def other_stats_figs(fig_dir, connections):
  for stat_type in ["min", "max", "avg"]:
    for dist_type in DIST_TYPE_LABELS.keys():
      labels = all_conns_in_one_plot(
          connections, dist_type, lambda x: make_conn_stat_plot(x, stat_type)
      )
      if labels:
        plt.xticks(rotation=90)
        plt.xlabel("Time")
        plt.ylabel(UNIT_LABELS[dist_type])
        plt.legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(stat_type + " of " + DIST_TYPE_LABELS[dist_type])
        filename = os.path.join(
            fig_dir, str(dist_type) + "_" + stat_type + ".png"
        )
        plt.savefig(filename, bbox_inches="tight")
        plt.clf()


def main(args):
  directory = args.source_dir
  fig_dir = args.fig_dir + "/"

  connections = read_data(directory)

  # Plotting a heat-map for every connections and every distribution type.
  time_series_figs(fig_dir, connections)

  # CDF plots for each distribution type.
  cdf_figs(fig_dir, connections)

  # Bandwidth plots for each distribution type.
  bandwidth_figs(fig_dir, connections)

  # Plots of other stats.
  other_stats_figs(fig_dir, connections)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--source_dir",
      type=str,
      default="/tmp",
      help="Directory where log files are stored.",
  )
  parser.add_argument(
      "--fig_dir",
      type=str,
      default="/tmp/figures",
      help="Directory where figures are stored.",
  )
  main(parser.parse_args())

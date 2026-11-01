
from scapy.all import rdpcap, IP, IPv6, TCP, UDP
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse


def get_packet_info(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        protocol = int(packet[IP].proto)
    elif IPv6 in packet:
        src = packet[IPv6].src
        dst = packet[IPv6].dst
        protocol = int(packet[IPv6].nh)
    else:
        return None

    if TCP in packet:
        sport = int(packet[TCP].sport)
        dport = int(packet[TCP].dport)
        flags = int(packet[TCP].flags)
    elif UDP in packet:
        sport = int(packet[UDP].sport)
        dport = int(packet[UDP].dport)
        flags = 0
    else:
        return None

    return src, sport, dst, dport, protocol, flags


def extract_flows(pcap_file):
    packets = rdpcap(pcap_file)
    flows = defaultdict(list)

    for packet in packets:
        info = get_packet_info(packet)

        if info is None:
            continue

        src, sport, dst, dport, protocol, flags = info

        forward = (src, sport, dst, dport, protocol)
        reverse = (dst, dport, src, sport, protocol)
        key = min(forward, reverse)

        direction = 0 if forward == key else 1

        flows[key].append({
            "time": float(packet.time),
            "length": len(packet),
            "direction": direction,
            "flags": flags
        })

    rows = []

    for key, records in flows.items():
        records.sort(key=lambda x: x["time"])

        fwd = [r for r in records if r["direction"] == 0]
        bwd = [r for r in records if r["direction"] == 1]

        times = np.array([r["time"] for r in records])
        lengths = np.array([r["length"] for r in records])

        iat = np.diff(times)
        fwd_lengths = np.array([r["length"] for r in fwd])
        bwd_lengths = np.array([r["length"] for r in bwd])

        duration = max(0.0, times[-1] - times[0])

        rows.append({
            "src_ip": key[0],
            "src_port": key[1],
            "dst_ip": key[2],
            "dst_port": key[3],
            "protocol": key[4],
            "flow_duration": duration * 1000000,
            "tot_fwd_pkts": len(fwd),
            "tot_bwd_pkts": len(bwd),
            "totlen_fwd_pkts": int(fwd_lengths.sum()) if len(fwd) else 0,
            "totlen_bwd_pkts": int(bwd_lengths.sum()) if len(bwd) else 0,
            "fwd_pkt_len_max": int(fwd_lengths.max()) if len(fwd) else 0,
            "fwd_pkt_len_min": int(fwd_lengths.min()) if len(fwd) else 0,
            "fwd_pkt_len_mean": float(fwd_lengths.mean()) if len(fwd) else 0,
            "bwd_pkt_len_max": int(bwd_lengths.max()) if len(bwd) else 0,
            "bwd_pkt_len_min": int(bwd_lengths.min()) if len(bwd) else 0,
            "bwd_pkt_len_mean": float(bwd_lengths.mean()) if len(bwd) else 0,
            "flow_byts_per_s": float(lengths.sum() / duration) if duration > 0 else 0,
            "flow_pkts_per_s": float(len(records) / duration) if duration > 0 else 0,
            "flow_iat_mean": float(iat.mean()) if len(iat) else 0,
            "flow_iat_std": float(iat.std()) if len(iat) > 1 else 0,
            "flow_iat_max": float(iat.max()) if len(iat) else 0,
            "flow_iat_min": float(iat.min()) if len(iat) else 0,
            "syn_count": sum(bool(r["flags"] & 0x02) for r in records),
            "ack_count": sum(bool(r["flags"] & 0x10) for r in records),
            "rst_count": sum(bool(r["flags"] & 0x04) for r in records),
            "fin_count": sum(bool(r["flags"] & 0x01) for r in records),
            "packet_count": len(records)
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="traffic_capture.pcap")
    parser.add_argument("--output", default="python_flows.csv")
    args = parser.parse_args()

    print("Reading PCAP:", args.input)

    df = extract_flows(args.input)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df.to_csv(args.output, index=False)

    print("\nFLOW EXTRACTION COMPLETED")
    print("Total flows:", len(df))
    print("Total features:", len(df.columns))
    print("Output:", args.output)
    print("\nExtracted columns:")
    print(df.columns.tolist())

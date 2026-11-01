
from scapy.all import sniff, get_if_list, IP, IPv6, TCP, UDP
from datetime import datetime
import threading

packet_count = 0
lock = threading.Lock()


def process_packet(packet):
    global packet_count

    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        protocol = packet[IP].proto

    elif IPv6 in packet:
        src = packet[IPv6].src
        dst = packet[IPv6].dst
        protocol = packet[IPv6].nh

    else:
        return

    if TCP in packet:
        protocol_name = "TCP"
        sport = packet[TCP].sport
        dport = packet[TCP].dport

    elif UDP in packet:
        protocol_name = "UDP"
        sport = packet[UDP].sport
        dport = packet[UDP].dport

    else:
        return

    with lock:
        packet_count += 1
        current_count = packet_count

    timestamp = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{timestamp}] "
        f"PACKET: {current_count} | "
        f"{protocol_name} | "
        f"{src}:{sport} -> {dst}:{dport} | "
        f"SIZE: {len(packet)} bytes"
    )


def main():

    print("=" * 65)
    print("       LIVE NETWORK TRAFFIC MONITOR")
    print("       CONTINUOUS PACKET CAPTURE")
    print("=" * 65)

    print("\nStarted:", datetime.now())

    print("\nAVAILABLE NETWORK INTERFACES")

    interfaces = get_if_list()

    for i, interface in enumerate(interfaces):
        print(f"{i + 1}. {interface}")

    print("\nStarting continuous packet capture...")
    print("Generate network traffic by browsing websites.")
    print("Press CTRL+C to stop.\n")

    try:

        sniff(
            prn=process_packet,
            store=False
        )

    except PermissionError:

        print("ERROR: Administrator permission required.")

    except KeyboardInterrupt:

        print("\n" + "=" * 65)
        print("LIVE CAPTURE STOPPED")
        print("=" * 65)

        print("Total packets captured:", packet_count)

    except Exception as e:

        print("Capture error:", str(e))


if __name__ == "__main__":
    main()

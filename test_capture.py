
from scapy.all import sniff, IP, TCP, UDP, conf

iface = conf.ifaces.dev_from_index(14)

if iface is None:
    raise RuntimeError("Wi-Fi interface not found")

print("Using interface:", iface.name)
print("Starting live packet capture...")
print("Press CTRL+C to stop.")

def process_packet(packet):
    if IP not in packet:
        return

    ip = packet[IP]

    if TCP in packet:
        protocol = "TCP"
    elif UDP in packet:
        protocol = "UDP"
    else:
        protocol = "OTHER"

    print(f"{ip.src} -> {ip.dst} | {protocol} | {len(packet)} bytes")

sniff(iface=iface, prn=process_packet, store=False)

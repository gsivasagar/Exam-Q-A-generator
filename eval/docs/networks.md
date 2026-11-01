# Computer Networks

## Transport Layer
TCP provides reliable, ordered delivery and begins a connection with a three-way handshake using SYN, SYN-ACK and ACK segments. Sequence numbers and acknowledgements let TCP detect lost packets and retransmit them. UDP is a connectionless protocol with no delivery guarantee, so it is preferred for live video calls and online games where speed matters more than completeness. TCP flow control uses a sliding window so that a fast sender does not overwhelm a slow receiver. Congestion control algorithms such as slow start reduce the sending rate when packets are dropped.

## Addressing
An IPv4 address has 32 bits written as four decimal octets, while an IPv6 address has 128 bits. A subnet mask such as 255.255.255.0 separates the network portion of an address from the host portion. CIDR notation like 192.168.1.0/24 states how many leading bits identify the network. The private ranges 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16 are not routed on the public internet. DHCP automatically assigns an IP address, gateway and DNS server to devices joining a network. ARP maps a known IP address to the MAC address of a device on the local network.

## Application Layer
DNS translates human-readable domain names into IP addresses using a hierarchy of root, top-level domain and authoritative servers. HTTP is a request-response protocol, and HTTPS adds encryption using TLS. The status code 404 means the requested resource was not found, while 500 signals an internal server error. A 301 response tells the client that a page has moved permanently to a new address. SMTP sends email between servers and IMAP lets clients read mail stored on a server.

## Reference Models
The OSI model has seven layers: physical, data link, network, transport, session, presentation and application. The TCP/IP model groups these into four layers. Routers forward packets at the network layer using routing tables, whereas switches forward frames at the data link layer using MAC addresses. A firewall filters traffic according to rules about addresses, ports and protocols. A VPN creates an encrypted tunnel across an untrusted network.

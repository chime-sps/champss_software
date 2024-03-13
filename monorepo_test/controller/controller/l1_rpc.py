# Based on code in frb-master

# Hardware Identification
racks = [1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d"]
rack_subnets = {
    1: [6, 7],
    2: [8, 9],
    3: [6, 7],
    4: [8, 9],
    5: [6, 7],
    6: [8, 9],
    7: [6, 7],
    8: [8, 9],
    9: [6, 7],
    10: [8, 9],
    11: [6, 7],
    12: [8, 9],
    13: [6, 7],
}
nodes = range(10)
columns = [range(0, 256), range(1000, 1256), range(2000, 2256), range(3000, 3256)]

# Assign Globals
RPC_PORT = 5555
NUM_BEAMS_PER_IP = 4
L1_NODES = []
L1_BEAMS = []
L1_IP_ADDRS = {}
NODE_TO_BEAM_MAP = {}
BEAM_TO_NODE_MAP = {}
BEAM_TO_IP_MAP = {}

# Calculate all the node names
for rack in racks:
    for node in nodes:
        L1_NODES.append(f"cf{rack}n{node}")

# Remove nodes cfDn8 and cfDn9 from the list of L1 Nodes
L1_NODES.remove("cfdn9")
L1_NODES.remove("cfdn8")

# Calculate all the beam numbers
for column in columns:
    for beam in column:
        L1_BEAMS.append(beam)

# Calculate all the ip addresses
for node in L1_NODES:
    # Select rack number, e.g. cf1n3 --> 1
    L1_IP_ADDRS[node] = []
    rack = node[2]
    # Convert rack to int using the hex casting
    for subnet in rack_subnets[int(rack, 16)]:
        L1_IP_ADDRS[node].append(
            "10.{}.{}.{}".format(
                subnet, str(200 + int(rack, 16)), str(10 + int(node[-1]))
            )
        )

# Create node name to beam map
for i in range(len(L1_NODES)):
    NODE_TO_BEAM_MAP[L1_NODES[i]] = L1_BEAMS[i * 8 : (i * 8) + 8]

# Create beam number to node name map
for node in NODE_TO_BEAM_MAP.keys():
    for beam in NODE_TO_BEAM_MAP[node]:
        BEAM_TO_NODE_MAP[beam] = node

# Create the beam number to ip address map
for node in NODE_TO_BEAM_MAP.keys():
    beams = NODE_TO_BEAM_MAP[node]
    ip_addrs = L1_IP_ADDRS[node]
    for beam in beams:
        BEAM_TO_IP_MAP[beam] = ip_addrs[int(beams.index(beam) / NUM_BEAMS_PER_IP)]


def get_node_beams(node_name):
    return NODE_TO_BEAM_MAP[node_name]


def get_node_rows(node_name):
    return [beam % 1000 for beam in NODE_TO_BEAM_MAP[node_name]]


def get_beam_node(beam_id):
    return BEAM_TO_NODE_MAP[beam_id]


def get_beam_ip(beam_id):
    return BEAM_TO_IP_MAP[beam_id]

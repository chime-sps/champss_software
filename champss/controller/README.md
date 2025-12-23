Slow Pulsar Pipeline Controller
===============================
![Test suite status](https://github.com/chime-sps/controller/workflows/Tests/badge.svg)

This repository contains tools for managing CHAMPSS data acquisition via Remote Procedure Calls (RPC) with CHIME L1 nodes.

## Command Line Tools

### spsctl

Main L1 controller for dynamic beam tracking. Generates pointings and sends RPC commands to L1 nodes to configure beam acquisition. The default behaviour uses the CHAMPSS pointing map to determine the channelization for each beam across the sky (e.g. increasing chanellization when the Galactic Plane is transitting).  Acquisition is stopped via the `stopacq` command, or via Ctrl-C

**Usage:**
```bash
spsctl [OPTIONS] ROW [ROW...]
```

**Arguments:**
- `ROW`: Beam row(s) to record (0-223). Each row maps to 4 beams (e.g., row 0 → beams 0, 1000, 2000, 3000).

### spsctl_batched

Parallelized wrapper for `spsctl`. Spawns multiple jobs to handle many beam rows simultaneously and distributes data across multiple mount points.

**Usage:**
```bash
spsctl_batched $(seq 0 49) --batchsize 10      # 50 rows in 5 batches
spsctl_batched 0 10 20 30 --basepath /sps-archiver2/raw/ --basepath /sps-archiver3/raw/
```

### stopacq

Stop acquisition on specified beam rows.

**Usage:**
```bash
stopacq [OPTIONS] ROW [ROW...]
```


### sched-known-psrs

Automated scheduler that records data when known pulsars transit through CHIME's FRB beams. For each pulsar, it finds the closest beamrow at transit, and sends an spsctl call to that specific beamrow for the full pointing length +3min on each side, then stops it.  In the default behaviour, it reads pulsars from the timing_ops database, and checks the database every 10 minutes to update the pulsar list.

## RPC System

### Overview

Controller communicates with L1 nodes via ZeroMQ RPC on port 5555, using `set_spulsar_writer_params`.

### set_spulsar_writer_params

Configures slow pulsar data writing on L1 nodes.

**Parameters:**
- `beam_id` (int): Beam identifier
- `nfreq_out` (int): Output frequency channels. Max is 16384 from CHIME/FRB, can downsample to and power of 2 from 1024..16384.  Set to 0 to stop.
- `ntime_out` (int): Time samples per chunk.  Default is 1024, 1ms res.  Reducing to eg. 512 downsamples in time to 2ms
- `nbins_out` (int): Fixed to 5 levels for spshuff, bit reduction from 8-bit to ~2.4-bits per sample with Huffman coding.
- `base_path` (str): CHAMPSS network mount on L1 node
- `source` (str): Writer instance (`champss` or `slow`). Two streams can write in parallel, but do slow team independent runs theirs.


### Beam to IP Mapping

Helper functions in `controller/l1_rpc.py`:
- `get_beam_ip(beam_id)`: Returns IP address for a beam
- `get_beam_node(beam_id)`: Returns node name (cf{rack}n{node})
- `get_node_beams(node_name)`: Returns beams on a node
- `get_node_rows(node_name)`: Returns beam rows on a node

**System layout:**
- L1 nodes: cf{rack}n{node} where rack=1-9,a-d, node=0-9
- Each node handles 8 beams (2 beams per IP, 4 IPs per node)
- Beam columns: 0-255, 1000-1255, 2000-2255, 3000-3255



Slow Pulsar Pipeline Controller
===============================
![Test suite status](https://github.com/chime-sps/controller/workflows/Tests/badge.svg)

This repository contains the code for managing the Slow Pulsar search pipeline.

Command Line Usage
------------------

```
spsctl [OPTIONS] [ROW]...
```

  ROW is the beam row which to track (in the range of 0-223). Default: all.

Options:
  --host HOSTNAME  Generate the schedule only for beam(s) running on
                   HOSTNAME(s). (Repeat to include multiple hosts.)

# Interface and infrastructure

Leader : Davor
Second : Chitrang, Shiny

## Overview

This work package includes all the supporting services required to run the
science pipeline and provide it with effective data storage and compute resource
management capabilities, as required by the infrastructure assumptions we give
to the pipeline components:
- Data products are available where and when they are needed.
- Metadata is recorded in a database.
- Processing components are managed effectively on the compute cluster.
- Components can subscribe to be notified about events of interest.


## Components

The following principles must be implemented by all facets of the
infrastructure, and they must in turn enable it for the science pipeline:

- robustness: the system needs to be able to operate correctly even in the case
  of hardware faults, software crashes, or human (operator) error.
- scalability: the system should keep performing even as load increases, and
  smoothly handle increases or decreases in available resouces.
- operability and maintainability: the system should be easy to keep running
  smoothly, understandable for troubleshooting problems, and simple enough so
  even new members can extend and run it in the future.

### Storage

We estimate data will require:

- L1: 45 TB/day (1.4 PB/month), at a rate of 4.2 Gb/s
- RFI: 3.7 TB/day
- Hhat: 1.1 PB for daily stacks + 2 x 1.1 PB for monthly and cumulative stacks

### Database

Event rate:
- L1 beam DM adjustments: ??
- pointings/day: 165,537


### Cluster Orchestration

Because the data rate varies depending on the part of the sky currently
overhead, we will also have varying intensity of computing, depending on which
patch of the sky is being processed. This means that we will need intelligent
scheduling of non-real time components, depending on backlog of data (e.g., Hhat
stacks to check), while keeping the realtime components running with resources
that they need to keep up with the data rate.


### Event Logging and Aggregation

Cluster activity should be collected in a single system where it is easily
searchable and queryable. While full-text search is probably not required, it is
essential that operators and developers can focus only on certain hosts,
components, and time ranges, and can access them in a format where they can be
processed by general utilities for text processing.


### Monitoring and Alerting

- Grafana for visualizations where possible (Prometheus and other backends if appropriate)
  - Prometheus is optimized for large-scale metrics, and its computation is free
    to interpolate between time points (it aims for "statistically accuracy",
    not "single-event precision")
  - For event-level precision, Grafana can use other backends, such as Postgress
    (optionally with TimescaleDB extension for time-series features) or Loki
    (Grafana's own log aggregation database)
- data-quality and science-specific dashboards:
  - some visualizations can't be done with built-in Grafana visualization
    widgets. Grafana can be extended with custom widgets, and we have used a few
    in FRB and CHIME, but the API is not simple and I'm not sure if it has all
    the interactivity.
  - possible alternative is Dash, which can produce web dashboards written
    entirely in Python. CHIME uses it for their Theremin quicklooks, too.


### Operations

Simple web and command-line interfaces for:
- applying software updates
- taking down individual hosts for maintenance
- restarting/rebooting hosts
- recovering from power outage


### Data Transfer Off-Site

- Automated tracking of data product and their copying to Compute Canada


## Interfaces

- web interfaces for operation and monitoring of cluster performance
- science-focused quicklook dashboards, extensible in Python by team members as
  they find need for them in their analyses


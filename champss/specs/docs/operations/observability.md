# Observability Stack Notes (as of Jan 29 2024)

## All SPS Nodes

The observabiity stack, deployed through Docker Swarm, is running on the CHIME telescope site node `ss1.chime` (formerly `psr-archiver.chime`). This node also serves as a storage node for CHIME SPS data processing due to the available HDDs and SSDs. There will be an `ss2.chime` (now called `sps-archiver.chime`) which will also export metrics (as will `sps-compute1.chime and sps-compute2.chime`), but not be used to host the metric collectors like `ss1.chime`. `ss1.chime` will be an equivalently-sized storage node for SPS as well.

## All Tools and Their Purpose

The stack consists of:
- Traefik (a reverse-proxy to join all observability tools under one URL with different paths)
    - Mimir (a scalable highly available object storage for all observability data)
        - Grafana (queries Prometheus and Loki to visualize all data in one place)
            - Prometheus (collect metrics from agents and allows querying through PromQL)
                - Node_Expoter (agent to export node metrics)
            - Loki (collect logs from agents and allows querying through LogQL)
                - Promtail (agent to export node logs)
            - cAdvisor (agent to export Docker container metrics and logs)

## Public CHIME SPS URL

However, you may notice that anything going through the Traefik reverse proxy (e.g. Grafana) is not accessible (despite Prometheus stating it is up and working, which it technically is). This is because Traefik is currently setup to use TLS with SSL certificates for secure connections, since all of this will actualy be available (after a credentials login) to a **public** URL, `sps.chimenet.ca`. This will be setup very soon. As such, anything here referencing `ss1.chime` URL will become synonymous with `sps.chimenet.ca`. 

If you want to temporarily test things without TLS, you must look in the `configs/` files and the Docker Compose YML file (which mounts those `configs/` files) and remove any mentions of `letsencrypt`, `websecure`, and `tls`.

## Proxy to CHIME Site

Currently, each of these tools is exported on a port on `ss1.chime` only, which will require an internet
connection to the CHIME telescope to access, which can be done from anywhere, by running on your
local computer:
```
sshuttle --dns -NHr <your_chime_username>@login.chimenet.ca 0/0
```

## Exposed Ports

The ports of each service for now:
```
Prometheus: 9090
Loki (read and write): 3100, 7946, 9095
Grafana: 3000
NodeExporter: 9093
Promtail: (not set up)
cAdvisor: (not set up)
Mimir: 8080, 7946, 9095 (composed of multiple containers for scaling purposes)
Traefik: 8000
```

So, you then could access Grafana with `ss1.chime:3000`. However, with Traefik, this becomes `ss1.chime/grafana`. As explained earlier, this will not currently work.

## Missing Services / Migration from Docker Compose to Docker Swarm Stack

Promtail is currently not exposing a port simply due to its current configuration.

cAdvisor is not given a port because it is not yet included in this observability stack. cAdvisor, like other tools listed
here (Promtail, etc), are meant to run on each node and export data to `ss1.chime`'s global Prometheus and Loki, and loaded to its Grafana. Currently, we have these tools running on each node, but they were not deployed **from** `ss1.chime` as they should be, because there are issues extending this observability stack from a Docker Compose file (local node) to a Docker Swarm Stack file (multi-node).

## Neccessity of .chime / HTTP

It is important to always include the `.chime` in configuration files, otherwise the Docker containers will use their local IP and nothing else will be able to reach it. Also sometimes, it's better not to explicity put `http://` in the URL since Docker or other tools under-the-hood implement this, but you can try both to see if it works.

## Neccessity of Docker Networks in Configurations

It's also important they expose their ports to the host in the Docker Compose files so the containers and the user can all communicate together through `ss1.chime` domain. Or at least, for required container-to-container communication, attach the containers to the same Docker Network so they can directly communicate (actually neccessity for Loki and Mimir, the internal ports in the containers used by the shared Docker Network are different than the exposed ports to the host).

If the Docker Compose YML file mentions a `network` with `external: true`, it requires it to be already created. You can check if it exists, and then create one if not with:
```
docker network ls
docker network create loki
docker network create mimir
docker network create cloud-edge
docker network create cloud-public
docker network create cloud-socket-proxy
```

## Neccessity of / at end of Proxied URL

If using Traefik (and thus TLS is working), you have to include a `/` at the end, e.g. http://sps.chimenet.ca/dashboard/, not http://sps.chimenet.ca/dasboard (`dashboard` is the dashboard for Traefik). 

## A .env file is a requirement

There will need to be a `.env`` file in the same directory as the docker-compose.yml file. It requires some fields for now:
```
DOCKER_ENV_DOMAIN=<sps.chimenet.ca if setup, else ss1.chime>
MINIO_USERNAME=frbadmin
MINIO_PASSWD=<frbadmin password>
```

This will already be given as an `env` file, and after you enter sensitive information, copy it to `.env` with `cp env .env`, since 
`.env` is ignored when pushing to GitHub, and the Docker Compose files are instructed to use the `.env`.

MinIO referenced here is explained in the `storage.md` file. It is where everything here is stored on and thus is very important.

## Why Does Mimir Have So Many Containers?

Mimir acts as a scable way of reading/writing metrics, so as to not slow down data proceesing read/writes. As such, it requires its components to be dviided, so as to autoscale each required part (writer, reader, etc.) individually as requested. 









# Storage Stack Notes (as of Jan 29 2024)

## MinIO VS POSIX

For storage, we are migrating from a POSIX filesystem with a hierarchical folder structure, to a scalable cloud-native object storage called MinIO which uses a well-supported S3 API (as used in AWS).

Such a tool provides a simple way to group drives across different nodes, and group servers across different sites, into a distributed data pool with even more read/write speed than POSIX. Additionally, it provides customizable "parity" data for disk fault redundancy, to provide no data loss with minimal cost. Importantly, with data distribution there is overall reduced I/O load per drive, as well as easy and quick scaling to new storage nodes anywhere. It also provides simple, well-documented user API tools to make using MinIO almost not different than what users are used to with POSIX.

## Object Storage

A file in MinIO is composed of around 16MiB object chunks which are distributed evenly across a set of drives to reduce the read/write load on each drive. This is called data striping. Additionally, each object has rich metadata that allows for the contents to be built quickly into the file on a read, and parity data can be applied such that when a drive fails, your data is not lost. The amount of parity you use determines how many drives can fail without any data loss, but the more used means the more extra space is used for this fault-redundancy parity data. This is called "erasure coding" in MinIO, which is their algorithm to do this with minimal cost. Data is stored in a "bucket", and in this bucket, you can still create "folders" and "files" as usual, but underneath the user-interface, the data is stored differently.

## Read/Write

Although your data is still stored on your drives as usual, you need to request your data through HTTP since the MinIO object storage is joined and accessed through a server API. The result of these requests can still return file-like objects as per usual, providing high compatibility with pre-existing code. Each bucket can have different access levels (directly hooked up to LDAP), and tiering can be configured to move data between SSDs and HDDs (e.g. "hot" and "cold" tiers) given a set of rules.

## Example

For example, with SPS, we may want to write a day's worth of raw intensity data to the "hot" storage multi-node set of SSDs, and then after a day, move it o the "warm" storage multi-node set of HDDs, and then 7 days after that, have it removed. This can all be done automatically in MinIO with set rules through the easy-to-use web interface or CLI.

## Ansible

To use MinIO, you need to format your HDDs and SSDs on your desired nodes, as well as do some kernel tuning. This can be done remotely with Ansible using given YML files in the CHIME SPS `sps-ops` GitHub repository. Firstly, install `sshpass`` wherever you currently are, as it's a neccessity to SSH into nodes with Ansible and run these commands remotely from wherever:
```
sudo apt install sshpass
```

Assuming you have Python3 instaled, then install Poetry through pip3:
```
pip install poetry
```

Head into `sps-ops`, which contains all of the neccesary code for the observability and storage stacks. From there, run `poetry install` to install all needed dependencies (e.g. Ansible) and then `poetry shell` to enter the virtual environment containing all of these dependencies. Now, you can start Ansible playbooks, which automates node setup, given a specific YML file containing an Ansible task.

If you are triyng to start this automated process to modify nodes that are NOT on your current network (e.g. from your local computer to modify CHIME site nodes), you need to modify your ~/.ssh/config file:

```
Host *.chime
   User <your_username>
   IdentityFile ~/.ssh/id_rsa # The location of your SSH key
   ProxyJump login.chimenet.ca # The CHIME site gateway node
```

Now you can SSH directly into CHIME site nodes without going through the gateway, e.g. you can SSH direcly into `sps-compute1.chime` using proxy jumps. Thus, Ansible can as well.

In the `/ansible` directory of `sps-ops`, you can modify the `hosts` file to include all the nodes you'd like your `ansible-playbook` commands to modiy.
```
[storage] # Label for organization
ss1.chime # Current master node for observability and storage stacks
ss2.chime # Currently named sps-archiver.chime
```

Then you can run something like this (to specify: the core YML file, the `/roles/<core_yml_name>/tasks/<task_yml_file>` task YML file, specific nodes to modify, the user to become with sudo privledges, extra vars in the YML files, etc.):
```
ansible-playbook minio.yml -t minio-tune --limit=ss1.chime -u frbadmin -K --extra-vars '{"minio_use_nvme":true}' --ask-vault-password
ansible-playbook minio.yml -t minio-setup --limit=ss1.chime -u frbadmin -K --extra-vars '{"minio_use_nvme":true}' --ask-vault-password
```
The password for the Ansible Vault, containing encrypted secrets, is the frbadmin password.

**RUNNING THIS ON `sps-archiver.chime` (or anywhere for that matter) WILL WIPE ALL THE DRIVES.**

## Ports

After running `docker compose up -d` in the directory of the docker-compose.yml file for MinIO, two MinIO servers will be started, one for the HDDs and one for the SSDs (each MinIO server must have identical drives in its set). Additionally, one port is for the web console, and one port is for direct access to the data.

- SSDs Console: `http://ss1.chime:9101`
- SSDs S3 API: `http://ss1.chime:9100`
- HDDs Console: `http://ss1.chime:9001`
- HDDs S3 API: `http://ss1.chime:9000`

You can use the MinIO client to directly modify a server. 

```
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc --help
```

You need to setup some tiers so that data from the SSDs can be transferred to the HDDs server (but still are able to accessed from the SSDs endpoint!), and personal aliases for each server, as well as some buckets.

```
# Set your person aliases
./mc alias set hot http://ss1.chime:9100/
./mc alias set warm http://ss1.chime:9000/
./mc alias ls

# See your perso

# Raw data will be on both SSDs (for current day) and HDDs (afterwards)
./mc mb hot/data
./mc mb warm/data

# Raw data will be stored on HDDs (as to not slow down raw data read/write)
./mc mb warm/loki-data
./mc mb warm/loki-ruler
./mc mb warm/mimir

# See the buckets "data", "mimir", etc., that you created, and their contents
./mc ls hot -r
./mc ls warm -r

# Add a tier on the "hot" MinIO sever's "data" bucket of the "warm" MinIO server's "data" bucket named as "WARM-TIER"
./mc ilm tier add minio hot WARM-TIER --access-key frbadmin --secret-key <frbadmin_password> --bucket data --endpoint http://ss1.chime:9000
./mc ilm tier list hot

# Set some rule on the hot/data bucket to move its data to the warm/data bucket after some time
./mc ilm rule add --transition-days 1 --transition-tier "WARM-TIER" hot/data

# Set some rule on the warm/data bucket to delete its data after some time
./mc ilm rule add --expire-days 1 warm/data

# See rules applied
./mc ilm rule list hot/data
./mc ilm rule list warm/data

# Copy a day's raw intensity data to the hot/data bucket
nohup ./mc cp --recursive /data/chime/sps/raw/2024/01/08/ hot/data/chime/sps/raw/2024/01/08/ > mc_cp_log.txt 2>&1 &

# See progress
cat mc_cp_log.txt
```

Of course, downloading and uploading files will be done through Python with a wrapper in `sps-common`/`spshuff`.

In the future, we will probably have a third tier, a cold tier, which will be somewhere off-site. 

## Environmental Varialbes in Docker Compose Per Service

### Parity and Storage Classes

```
# The default storage class is STANDARD for objects in a bucket. You don't have to
# specify these values as they do have default values, but this can be modified to
# specify the amount of extra parity data used for drive failure redundancy
- MINIO_STORAGE_CLASS_STANDARD=EC:1
# The amount of parity data used for objects given the REDUCED storage class. Also
# doesn't need to be specified.
- MINIO_STORAGE_CLASS_RRS=EC:1
```

See [here](https://min.io/docs/minio/linux/operations/concepts/erasure-coding.html#minio-ec-erasure-set:~:text=Drive%20MinIO%20Cluster-,Parity,-Total%20Storage) for what each level of parity data means in terms of storage used.

See [here](https://min.io/docs/minio/linux/reference/minio-server/settings/storage-class.html#:~:text=in%20the%20deployment%3A-,Erasure%20Set%20Size,-Default%20Parity%20) for the default values depending on the number of drives used in your MinIO server.

### Prometheus and Loki

```
# These labels are simpled applied to PromQL querying, so that Prometheus data can be seen in the MinIO console
- MINIO_PROMETHEUS_AUTH_TYPE=public # No token needed to access our Prometheus instance
- MINIO_PROMETHEUS_URL=http://ss1.chime:9090/ # Where is Prometheus hosted?
- MINIO_PROMETHEUS_EXTRA_LABELS=instance="ss1.chime:9000" # There are multiple MinIO servers running that Prometheus picks up, specify the one for the given Docker Compose Service
# - MINIO_PROMETHEUS_JOB_ID= -> Do not specify this. Maybe a possible bug with MinIO. The default job id works "minio-job".
```

```
# Need this at the top of the Docker Compose file for Loki to work with MinIO
networks:
  minio:
    external: true
  loki:
    external: true

x-logging: &loki-logging
  driver: loki
  options:
    loki-url: "http://localhost:8000/loki/loki/api/v1/push"
    loki-retries: "5"
    loki-batch-size: "102400" # 100KB
    loki-tenant-id: "lokilocal"
```

## LDAP and Access Levels

Additionally, everyone can be given an account to access the MinIO object storage cluster, automatically inherited from LDAP. Each bucket can have different access/permission levels per user.

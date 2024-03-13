#!/bin/sh

# Check if NODE_ID and NODE_NAME environment variables are set
if [ -z  "$NODE_ID"  ] || [ -z  "$NODE_NAME"  ]; then
  echo  "NODE_ID and NODE_NAME environment variables must be set"
  exit  1
fi

# Check if /home directory exists and is writable
if [ ! -w  "/home"  ]; then
  echo  "/home directory does not exist or is not writable"
  exit  1
fi

# Echo the node id and node name to stdout
echo "============================="
echo  "Docker Swarm Node Metadata"
echo  "ID   : $NODE_ID"
echo  "NAME : $NODE_NAME"
echo  "FILE : /home/node-meta.prom"
echo "============================="

# Export NODE_ID and NODE_NAME for node export textfile collector
# Tell the user where the file will be written
echo  "Exporting NODE_ID and NODE_NAME to /home/node-meta.prom"
echo "node_meta{node_id=\"$NODE_ID\", container_label_com_docker_swarm_node_id=\"$NODE_ID\", node_name=\"$NODE_NAME\"} 1" > /home/node-meta.prom

# Release the hounds!
exec /bin/node_exporter  "$@"

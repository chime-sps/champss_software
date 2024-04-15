"""Script to delete stalling Docker Containers from the SPS Pipeline Docker Stack."""
import time
from datetime import datetime, timedelta

import docker

client = docker.from_env()

container_name_prefixes = [
    "pipeline_tiny",
    "pipeline_small",
    "pipeline_medium",
    "pipeline_large",
    "pipeline_huge",
]
seconds_per_check = 300
stalling_time = timedelta(minutes=45)


def main():
    """Entry function to delete stalling SPS Pipeline Docker Containers."""
    while True:
        containers = [
            container
            for container in client.containers.list(all=True)
            if any(prefix in container.name for prefix in container_name_prefixes)
        ]
        for container in containers:
            container_status = container.attrs["State"]["Status"]
            container_id = container.id
            container_name = container.name
            started_time = datetime.fromisoformat(
                container.attrs["State"]["StartedAt"].split(".", 1)[0]
            )
            current_time = datetime.now()
            elapsed_time = current_time - started_time

            if container_status == "running" and elapsed_time > stalling_time:
                print(
                    f"Force removing container {container_name} {container_id} as it"
                    " has been running for more than 45 minutes."
                )
                container.remove(force=True)

        time.sleep(seconds_per_check)

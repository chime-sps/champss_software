"""Script to delete exited Docker Containers from the SPS Pipeline Docker Stack."""
import time

import docker

client = docker.from_env()

container_name_prefixes = [
    "pipeline_tiny",
    "pipeline_small",
    "pipeline_medium",
    "pipeline_large",
    "pipeline_huge",
]
seconds_per_check = 10


def main():
    """Entry function to delete exited SPS Pipeline Docker Containers."""
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
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            if container_status == "exited":
                print(
                    f"[{timestamp}] Pipeline container has exited, deleting container"
                    f" {container_name} {container_id}."
                )
                container.remove(force=True)

        time.sleep(seconds_per_check)

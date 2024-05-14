import docker
import time
from datetime import datetime

client = docker.from_env()

seconds_per_check = 600


def main():
    while True:
        """Entry function to delete replaced and un-used Docker Images."""
        images = {}

        for image in client.images.list():
            # Only delete duplicate images that are not the most recent of them
            try:
                image_name = image.attrs["RepoDigests"][0].split("@")[0]
            except Exception as e:
                try:
                    image_name = image.attrs["RepoTags"][0].split(":")[0]
                except Exception as e:
                    print(e)

            if image_name not in images:
                images[image_name] = []

            image_id = image.attrs["Id"]
            image_timestamp_string = image.attrs["Created"].split("Z", 1)[0].split(".",1)[0]
            image_timestamp = datetime.fromisoformat(
                image_timestamp_string
            )
            
            images[image_name].append({"id": image_id, "timestamp": image_timestamp})

        for image_name in images.keys():
            # Sort by most recently created image to oldest image
            sorted_images = sorted(
                images[image_name],
                key=lambda x: x["timestamp"],
                reverse=True,
            )

            for image_index, image in enumerate(sorted_images[1:]):
                # No force, only if they are not in use
                image_id = image["id"]
                print(f"Removing image {image_name} - {image_id}...")
                try:
                    client.images.remove(images[image_name][image_index]["id"])
                except Exception as e:
                    print(e)

        time.sleep(seconds_per_check)


if __name__ == "__main__":
    main()

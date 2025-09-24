import argparse
import json
import sys
from datetime import datetime

from sps_databases import db_utils, models

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=27017, help="The port of the database.")
parser.add_argument("--name", type=str, default="sps", help="The name of the database.")
parser.add_argument(
    "--host", type=str, default="sps-archiver1", help="The name of the database."
)
parser.add_argument(
    "--count",
    type=int,
    default=0,
    help="The number of dummy observations added to the db.",
)
args = parser.parse_args()

# Connect to the database
db = db_utils.connect(host=args.host, port=args.port, name=args.name)

if "pointings" in db.list_collection_names():
    print("The database is already populated")
    sys.exit(1)

# Add pointings
print("Load the pointings map")
pointings = json.load(open("./pointings_map_v1-3.json"))
for p in pointings:
    p["search_algorithm"] = models.SearchAlgorithm.power_spec.value

print("Insert pointings into the database")
rc = db.pointings.insert_many(pointings)
assert rc.acknowledged
for id, p in zip(rc.inserted_ids, pointings):
    p["id"] = id

# Add observations for some of the pointings
if args.count != 0:
    observations = []
    for p in pointings[: args.count]:
        obs = {
            "pointing_id": p["id"],
            "datetime": datetime.utcnow(),
            "datapath": "/path/to/data",
            "status": models.ObservationStatus.scheduled.value,
        }
        observations.append(obs)

    print("Add some observations into the database")
    rc = db.observations.insert_many(observations)
    assert rc.acknowledged
    for id, obs in zip(rc.inserted_ids, observations):
        obs["id"] = id
else:
    print("No dummy observations will be inserted into the database.")
print("Done")

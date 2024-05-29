import datetime as dt
import math

import pytz
from pymongo.errors import ServerSelectionTimeoutError
from sps_databases import db_utils


def test_datetimes():
    try:
        db = db_utils.connect()
        now = dt.datetime.utcnow().replace(tzinfo=pytz.UTC)
        db.test.insert_one({"date": now})
        db_now = db.test.find_one()["date"]
        assert db_now.tzinfo
        assert math.isclose(now.timestamp(), db_now.timestamp())
    except ServerSelectionTimeoutError as error:
        print("Not connected to SPS database to run tests")

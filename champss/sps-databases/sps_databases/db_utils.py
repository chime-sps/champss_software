from pymongo import MongoClient

db = None


def connect(host="localhost", port=27017, name="sps"):
    global db
    if db is None:
        client = MongoClient(host, port, tz_aware=True, serverSelectionTimeoutMS=2000)
        db = getattr(client, name)
    return db

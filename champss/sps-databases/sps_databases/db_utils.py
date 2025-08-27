from pymongo import MongoClient

db = None


def connect(host="sps-archiver1", port=27017, name="sps"):
    global db
    if db is None:
        client = MongoClient(host, port, tz_aware=True, serverSelectionTimeoutMS=2000)
        db = getattr(client, name)
    return db

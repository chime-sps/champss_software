# sps-databases
![Test suite status](https://github.com/chime-sps/sps-databases/workflows/Tests/badge.svg)
Database layer for SPS

## Install the Database Server

If the Mongo container is not already running, you can start your own:

1. Check if it started:
   ```
   $ docker container ls
   CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                                  NAMES
   fd11cab32e6d        mongo               "docker-entrypoint.s…"   7 weeks ago         Up 3 seconds        0.0.0.0:27017-27019->27017-27019/tcp   sps-db-mongo
   ```

2. Pull the docker Image for the MongoDB server:
   ```
   docker pull mongo
   ```

3. Start the MongoDB server:
   ```
   docker run --name sps-db-mongo -p 27017:27017 mongo
   ```
   One can force the container to save their contents to a specific folder `/your/folder` by adding `-v /your/folder/db:/data/db` to that command. This prevents the data from being deleted if the container is deleted.

## Populate the database

```
$ cd sps_databases/utilities
$ python populate_db.py #(Note: it will take a few minutes to load all the pointings into the database.)
```
This script allows the use of options `--host` and `--port` to specify the docker instance. The options `--name` allows the use of a different name for the database which prevents interfering with the database entries of other people.

# Connecting to the database

The `sps_databases` package provides a set of methods for accessing the database, although you can also use MongoDB directly using the `pymongo` package.

## Read from the database

```python
from sps_databases import db_api
# assuming you already have a `pointing_id`:
print (db_api.get_pointing(pointing_id))
```

Will output something like:
```
Pointing(_id=ObjectId('60147308fd86abb1e4b7e5b2'), ra=300.00720109575036, dec=4.540456929400711, beam_row=23, length=368640, ne2001dm=133.19571311122226, ymw16dm=104.6781383116166, maxdm=300.30837821026887, nchans=2048, search_algorithm=<SearchAlgorithm.power_spec: 1>)
```

## Connect to a database on another port
If your database is running on a host other than "sps-archiver1" and the default port 27017, before calling any functions in `db_api`, first call `db_utils.connect`:
```python
from sps_databases import db_api, db_utils

db_utils.connect("somehost", port=12345, name="myname")
db_api(get_pointing(pointing_id))
# Pointing(_id=ObjectId('60147308fd86abb1e4b7e5b2'), ra=300.00720109575036, dec=4.540456929400711, beam_row=23, length=368640, ne2001dm=133.19571311122226, ymw16dm=104.6781383116166, maxdm=300.30837821026887, nchans=2048, search_algorithm=<SearchAlgorithm.power_spec: 1>)
```
Note that when running db_utils.connect multiple times with different settings only the first connection will be established.
When the database is running on the same machine, you can use "localhost" as the host.

## Access with `pymongo`

To connect manually to the database, inside python shell do:

```
In [1]: import pymongo

In [2]: client = pymongo.MongoClient()

In [3]: db = client.sps

In [4]: client.sps.list_collection_names()
Out [4]: ['pointings', 'observations']

In [5]: client.sps.pointings.find_one()
Out[5]:
{'_id': ObjectId('60147308fd86abb1e4b77d4e'),
 'beam_row': 0,
 'ra': 359.9990711484846,
 'dec': -10.0316648855797,
 'length': 368640,
 'ne2001dm': 31.875682121784383,
 'ymw16dm': 22.223016520521156,
 'maxdm': 113.75136424356876,
 'nchans': 1024,
 'search_algorithm': 1}
```

This method gives you direct access to all capabilities of MongoDB, but
obviously it also requires closer knowledge of its APIs than just going through
`sps_databases`.

## Making queries using Mongo operators
```python
# Let's try to get all pointings with 300 < RA < 310 and 4 < Dec < 10: 
In[6]: list(client.sps.pointings.find({"ra": {"$gt": 300, "$lt": 310}, "dec": {"$gt": 4, "$lt": 10}}))[:3]
Out[6]:
[{'_id': ObjectId('60147308fd86abb1e4b7e5b2'),
  'beam_row': 23,
  'ra': 300.00720109575036,
  'dec': 4.540456929400711,
  'length': 368640,
  'ne2001dm': 133.19571311122226,
  'ymw16dm': 104.6781383116166,
  'maxdm': 300.30837821026887,
  'nchans': 2048,
  'search_algorithm': 1},
 {'_id': ObjectId('60147308fd86abb1e4b7e5b3'),
  'beam_row': 23,
  'ra': 300.32802954208114,
  'dec': 4.539885795431418,
  'length': 368640,
  'ne2001dm': 129.75505489042774,
  'ymw16dm': 102.38311432463907,
  'maxdm': 295.9840817661643,
  'nchans': 2048,
  'search_algorithm': 1},
 {'_id': ObjectId('60147308fd86abb1e4b7e5b4'),
  'beam_row': 23,
  'ra': 300.64885773299955,
  'dec': 4.5393165393294925,
  'length': 368640,
  'ne2001dm': 126.7378538445316,
  'ymw16dm': 100.23268208159362,
  'maxdm': 292.37305164424026,
  'nchans': 2048,
  'search_algorithm': 1}]
```

For the different types of queries that you can do with MongoDB, see the [tutorial](https://docs.mongodb.com/manual/tutorial/query-documents/).

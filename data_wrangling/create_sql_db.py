""" Run this file from command line to read csv files into sql databases """

import sqlite3
import csv

############################################################
#      Strings containing commands for creating tables     #
############################################################

nodes_table = """
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT
);"""

nodes_tags_table = """
CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id)
);"""

ways_table = """
CREATE TABLE ways (
    id INTEGER PRIMARY KEY NOT NULL,
    user TEXT,
    uid INTEGER,
    version TEXT,
    changeset INTEGER,
    timestamp TEXT
);"""

ways_tags_table = """
CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id)
);"""

ways_nodes_table = """
CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);"""

############################################################
#                    Other variables                       #
############################################################


table_names = ["nodes", "nodes_tags", "ways", "ways_tags", "ways_nodes"]

table_creation_commands = {
    "nodes_table": nodes_table, 
    "nodes_tags_table": nodes_tags_table,
    "ways_table": ways_table,
    "ways_tags_table": ways_tags_table, 
    "ways_nodes_table": ways_nodes_table
}

# Data insertion commands
insert_nodes = """
    INSERT INTO nodes (id,lat,lon,user,uid,version,changeset,timestamp) 
    VALUES(?,?,?,?,?,?,?,?);"""
insert_nodes_tags = """
    INSERT INTO nodes_tags (id,key,value,type)
    VALUES(?,?,?,?) """
insert_ways = """
    INSERT INTO ways (id,user,uid,version,changeset,timestamp)
    VALUES(?,?,?,?,?,?)"""
insert_ways_tags = """
    INSERT INTO ways_tags (id,key,value,type)
    VALUES(?,?,?,?)"""
insert_ways_nodes = """
    INSERT INTO ways_nodes (id,node_id,position)
    VALUES(?,?,?)"""

# Used below for reading data from csv files to database tables
d = {"nodes": 
        {"f": "nodes.csv", "insert_cmd": insert_nodes},
    "nodes_tags": 
        {"f": "nodes_tags.csv", "insert_cmd": insert_nodes_tags},
    "ways":
        {"f": "ways.csv", "insert_cmd": insert_ways},
    "ways_tags":
        {"f": "ways_tags.csv", "insert_cmd": insert_ways_tags},
    "ways_nodes":
        {"f": "ways_nodes.csv", "insert_cmd": insert_ways_nodes}
}

############################################################
#       Creates tables in database - osm_vermont.db        #
############################################################


conn = sqlite3.connect("osm_vermont.db")
c = conn.cursor()

# drops tables
for name in table_names:
    c.execute("DROP TABLE IF EXISTS %s" % name)

# creates tables
for key in table_creation_commands:
    c.execute(table_creation_commands[key])

# writes data in csv files into database tables
for key in d:
    with open(d[key]["f"]) as f:
        insert_cmd = d[key]["insert_cmd"]
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for i in reader:
            to_db = tuple(i[headers[j]] for j in range(len(headers)))
            c.execute(insert_cmd, to_db)

conn.commit()
conn.close()

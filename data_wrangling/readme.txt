This data wrangling was done as part of the data analyst nanodegree from Udacity.  I’ve uploaded it to github as a sample project for anyone who wants a demonstration of my competence with data wrangling and cleaning using Python and data-base querying using SQL.  

The project worked with open street map data in Vermont, which is a collection of detailed data on the roads and buildings in Vermont.


Files in order of use during auditing/cleaning/processing OpenStreetMap data.


audit.py
Includes functions for auditing the OSM data. To be run from command line.

data.py
Converts OSM XML data into CSV tables. Cleans data while updating it.

schema.py
Taken from Udacity course and used by the data.py file.

create_sql_db.py
Converts CSV files into sql database.

openstreets.pdf
Summary report of the project, including SQL queries run from command line.

resources.txt
Details two helpful blogposts referenced when working on project.

vermont_sample_100.osm
A subset of the Vermont dataset that was used in the project. The full dataset is too large to upload to github.

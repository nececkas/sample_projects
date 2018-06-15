""" Import this file into python shell then run commands to audit data """


import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint


street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
lower_multiple_colons = re.compile(r'^([a-z]|_|:)*$')
upper_possible_colons = re.compile(r'^([a-z]|[A-Z]|_|:)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

# ================================================== #
#                Investigate Tags                    #
# ================================================== #

# COUNT TAGS
def count_tags(filename):
    d = {}
    for event, element in ET.iterparse(filename):
        if element.tag not in d.keys():
            d[element.tag] = 1
        else:
            d[element.tag] += 1
    return(d)

# COUNT TAG'S KEYS BASED ON THEIR FORMAT (i.e. problematic or not)
# this is a helper function; call function that's further below
def key_type(element, keys):
    if element.tag == "tag":
        if problemchars.search(element.attrib['k']):
            keys['problemchars'] += 1
        elif lower_colon.search(element.attrib['k']):
            keys['lower_colon'] += 1
        elif lower_multiple_colons.search(element.attrib['k']):
            keys['lower_multiple_colons'] += 1
        elif lower.search(element.attrib['k']):    
            keys['lower'] += 1
        elif upper_possible_colons.search(element.attrib['k']):
            keys['upper_possible_colons'] += 1
        else:
            keys['other'] += 1
        pass 

    return(keys)

def count_problem_keys(filename):
    keys = {"lower": 0, "lower_colon": 0, "lower_multiple_colons": 0, "upper_possible_colons": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return(keys)

# PRINT DICTIONARY OF ALL KEYS & THEIR COUNTS
# this is a helper function; call function that's further below
def process_key(element, all_keys):
    if element.tag == 'tag':
        key = element.get('k')
        if key in all_keys:
            all_keys[key] += 1
        else:
            all_keys[key] = 1

    return(all_keys)

def get_all_keys(filename):
    all_keys = {}
    for _, element in ET.iterparse(filename):
        all_keys = process_key(element, all_keys)
    
    pprint.pprint(all_keys)


# PRINT VALUES FOR A TAG WITH A GIVEN 'K' VALUE
# this is a helper function; call function that's further below
def tag_list(element, list_values, tag_key):
    if element.tag == 'tag':
        key = element.get('k')
        if key == tag_key:
            list_values.append(element.attrib['v'])

    return(list_values)

def list_tag_values(filename, tag_key):
    list_values = []
    for _, element in ET.iterparse(filename):
        list_values = tag_list(element, list_values, tag_key)
    
    pprint.pprint(list_values)


# ================================================== #
#                Audit Street Names                  #
# ================================================== #

# I used the audit_streets function to find names to add
# to the below variables: expected, mapping

expected = ["Access", "Avenue", "Center", "Circle", "Court",
            "Drive", "Extension", "Green", "Heights", "Highway",
            "Hill", "Hollow", "Junction", "Lane", "Loop",
            "Parkway", "Place", "Road", "Square", "Street",
            "Terrace"]

mapping = {} 
# Uncomment the below mapping if you want function to update names.
# This will result in shorter list of results when auditing names.
#
# Mapping variable used in the data.py file to update street names
"""mapping = {"Ave": "Avenue",
           "Ave.": "Avenue",
           "Bch": "Beach",
           "Cir": "Circle",
           "Ct": "Court",
           "dr": "Drive"
           "Dr": "Drive",
           "Dr.": "Drive",
           "Hts": "Heights",
           "Hwy": "Highway",
           "Ln": "Lane",
           "Pkwy": "Parkway",
           "Pky": "Parkway",
           "Pl": "Place",
           "Rd": "Road",
           "Rd.": "Road",
           "SQ": "Square",
           "Sq": "Square",
           "St": "Street",
           "St.": "Street",
           "Ter": "Terrace",
           "terrace": "Terrace"}"""

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            #if street_typein mapping:
            #   update_name(street_type, mapping)
            street_types[street_type].add(street_name)
# Remove commented lines in function if you want function to update names.
# This will result in shorter list of results when auditing names.

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit_streets(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    pprint.pprint(street_types)

# ================================================== #
#                Audit Postal Codes                  #
# ================================================== #

POSTCODE = re.compile(r'^\d{5}$')

def check_postcode(postcode, postcodes):
    if not POSTCODE.match(postcode):
        postcodes.add(postcode)


def audit_postcodes(osmfile):
    osm_file = open(osmfile, "r")
    postcodes = set()
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"): 
                if tag.attrib['k'] == 'addr:postcode':
                    postcode = tag.attrib['v']
                    check_postcode(postcode, postcodes)
                    
    osm_file.close()
    pprint.pprint(postcodes)
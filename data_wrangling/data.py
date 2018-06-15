""" Run this file from command line to read map data into csv files """


import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import schema

OSM_PATH = "example.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

# regular expressions for use in below functions
LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
NUMBERS = re.compile(r'\d+')
CANADA_POSTCODE = re.compile(r'^J\S\S ')

TIGER = re.compile(r'tiger')

SCHEMA = schema.schema

mapping = {"Ave": "Avenue",  # words typically at the end of street names
           "Ave.": "Avenue",
           "Bch": "Beach",
           "Cir": "Circle",
           "Ct": "Court",
           "dr": "Drive",
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
           "terrace": "Terrace",
           "rue": "Rue",         # words not typically at end of street names
           "Rt": "Route",
           "Rte.": "Route",
           "route": "Route",
           "VT-": "VT Route",
           "Ste": "Suite",
           "Ste.": "Suite",
           "US-": "US"
           }

# ================================================== #
#     Change Tag Values as Elements are Shaped       #
# ================================================== #

def update_street_name(name, mapping):
    words = re.split('VT-|US-| ', name)
    for w in range(len(words)):
        # substitutes abbreviation for word in mapping (see above dictionary)
        if words[w] in mapping:
            words[w] = mapping[words[w]]
        # if word is 'VT' or 'Vermont' and it's followed directly
        # by a 'word' starting with numbers (i.e. VT 100), then
        # it's changed to 'VT Route'
        if words[w] == "VT":
            if NUMBERS.search(words[w + 1]):
                words[w] = "VT Route"
        # changes 'State' to 'VT' if followed by 'Route'
        # this avoids changing 'State' in 'State Street'
        if words[w] == "State":
            if words[w + 1] == "Route":
                words[w] = "VT"
    name = " ".join(words)
    name = name.strip(' ,')
    return name

def update_postcode(postcode):
    if CANADA_POSTCODE.search(postcode):
        postcode = postcode
    else:
        # remove any character or whitespace that isn't a number
        postcode = re.sub('\D', '', postcode)
        # take only 5 leftmost digits
        postcode = postcode[:5]
    
    return(postcode)


# ================================================== #
#     Shape Element for Insertion into CSV Files     #
# ================================================== #

# Ensures order of fields when they are inserted into CSV files
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_secondary_tags(element):    
    temp_list_children = []
    for child in element.findall('tag'):
        temp_dict_child = {}
        temp_dict_child['id'] = element.attrib['id']
        if PROBLEMCHARS.search(child.attrib['k']):
            break
        elif LOWER_COLON.search(child.attrib['k']):
            t, k = child.attrib['k'].split(':', 1) # maxsplit is 1
            temp_dict_child['key'] = k
            temp_dict_child['type'] = t
        else:
            temp_dict_child['key'] = child.attrib['k']
            temp_dict_child['type'] = 'regular'
        
        if child.attrib['k'] == "addr:street":
            temp_dict_child['value'] = update_street_name(child.attrib['v'], mapping)
        elif child.attrib['k'] == "addr:postcode":
            temp_dict_child['value'] = update_postcode(child.attrib['v'])
        else:
            temp_dict_child['value'] = child.attrib['v']

        temp_list_children.append(temp_dict_child)
    
    return(temp_list_children)


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []

    if element.tag == 'node':
        temp_dict = {}
        for i in NODE_FIELDS:
            temp_dict[i] = element.attrib[i]
        node_attribs.update(temp_dict)

        temp_list_children = shape_secondary_tags(element)
        tags.extend(temp_list_children)
                        
    elif element.tag == 'way':
        temp_dict = {}
        for i in WAY_FIELDS:
            temp_dict[i] = element.attrib[i]
        way_attribs.update(temp_dict)

        temp_list_children = shape_secondary_tags(element)
        tags.extend(temp_list_children)

        position = 0
        for child in element.findall('nd'):
            temp_dict_nds = {}
            temp_dict_nds['id'] = element.attrib['id']
            temp_dict_nds['node_id'] = child.attrib['ref']
            temp_dict_nds['position'] = position
            way_nodes.append(temp_dict_nds)
            position += 1
    
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))

# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = csv.DictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = csv.DictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = csv.DictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = csv.DictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = csv.DictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])

process_map('vermont.osm', 'validate')
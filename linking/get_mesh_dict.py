import sys
import os
from lxml import etree
import xml.etree.ElementTree as et
import pickle as pkl

from utils import dump_dict

_, desc_xml_path, supp_xml_path, dict_dir = sys.argv

# MeSH textual information
mesh_dict = {}

mesh_tree = et.parse(desc_xml_path)
mesh_root = mesh_tree.getroot()
for child in mesh_root:
    mesh_id = child.find('DescriptorUI').text
    mesh_name = child.find('DescriptorName').find('String').text.lower()
    mesh_dict[mesh_name] = mesh_id

    for concept in child.find('ConceptList'):
        concept_name = concept.find('ConceptName').find('String').text.lower()
        mesh_dict[concept_name] = mesh_id
        term_list = concept.find('TermList')
        for term in term_list:
            term_name = term.find('String').text.lower()
            mesh_dict[term_name] = mesh_id

mesh_tree = et.parse(supp_xml_path)
mesh_root = mesh_tree.getroot()
for child in mesh_root:
    recUI = child.find('SupplementalRecordUI')
    if recUI is None: continue
    mesh_id = recUI.text
    mesh_name = child.find('SupplementalRecordName').find('String').text.lower()
    for concept in child.find('ConceptList'):
        concept_name = concept.find('ConceptName').find('String').text.lower()
        mesh_dict[concept_name] = mesh_id
        term_list = concept.find('TermList')
        for term in term_list:
            term_name = term.find('String').text.lower()
            mesh_dict[term_name] = mesh_id

dump_dict(mesh_dict, os.path.join(dict_dir, 'mesh_dict.pkl'))

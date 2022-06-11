import sys
import os
from lxml import etree
import xml.etree.ElementTree as et
import pickle as pkl

from utils import dump_dict

_, uniprot_xml_path, dict_dir = sys.argv

up_dict = {}
# UniProt textual information
uniprot_root = etree.parse(uniprot_xml_path, parser=etree.XMLParser())
for protein in uniprot_root.xpath('./*[local-name()="entry"]'):
    p_synonyms = []
    gene_names = []
    uniprot_id = protein.xpath('./*[local-name()="accession"]')[0].text

    uniprot_name = protein.xpath('./*[local-name()="name"]')[0].text.lower()
    up_dict[uniprot_name] = uniprot_id

    protein_ = protein.xpath('./*[local-name()="protein"]')[0]
    recommended_name = protein_.xpath('./*[local-name()="recommendedName"]')[0]
    recommended_name_full = recommended_name.xpath('./*[local-name()="fullName"]')[0].text.lower()
    up_dict[recommended_name_full] = uniprot_id
    alternative_names = protein_.xpath('./*[local-name()="alternativeName"]')
    for alternative_name in alternative_names:
        alternative_name_full = alternative_name.xpath('./*[local-name()="fullName"]')[0].text.lower()
        up_dict[alternative_name_full] = uniprot_id

    gene = protein.xpath('./*[local-name()="gene"]')
    if len(gene) != 0:
        gene_name = gene[0].xpath('./*[local-name()="name"]')[0].text.lower()
        up_dict[gene_name] = uniprot_id

dump_dict(up_dict, os.path.join(dict_dir, 'up_dict.pkl'))

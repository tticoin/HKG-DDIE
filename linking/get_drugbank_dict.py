import os
import sys
from lxml import etree
import pickle as pkl

from utils import dump_dict

_, xml_path, dict_dir = sys.argv

def get_name_dict(root):
    db_dict = {} # key:name, value:drugbank_id
    atc_dict = {} # key:name, valud:ATC-code

    for drug in root.xpath('./*[local-name()="drug"]'):
        drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text

        # Name
        name_text = drug.xpath('./*[local-name()="name"]')[0].text.lower()
        db_dict[name_text] = drug_id
        # Brand
        for brand in drug.xpath('./*[local-name()="international-brands"]')[0]:
            brand_text = brand.xpath('./*[local-name()="name"]')[0].text.lower()
            db_dict[brand_text] = drug_id
        # Product
        for product in drug.xpath('./*[local-name()="products"]')[0]:
            product_text = product.xpath('./*[local-name()="name"]')[0].text.lower()
            db_dict[product_text] = drug_id
        # Synonyms
        for syn in drug.xpath('./*[local-name()="synonyms"]')[0]:
            syn_text = syn.text.lower()
            db_dict[syn_text] = drug_id

        # ATC-code
        for atcs in drug.xpath('./*[local-name()="atc-codes"]')[0]:
            for atc in atcs:
                atc_id = atc.attrib['code']
                atc_text = atc.text.lower()
                atc_dict[atc_text] = atc_id

    return db_dict, atc_dict

root = etree.parse(xml_path, parser=etree.XMLParser())
db_dict, atc_dict = get_name_dict(root)

dump_dict(db_dict, os.path.join(dict_dir, 'db_dict.pkl'))
dump_dict(atc_dict, os.path.join(dict_dir, 'atc_dict.pkl'))

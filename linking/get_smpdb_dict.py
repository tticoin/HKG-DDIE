import sys
import os
import pickle as pkl
import csv

from utils import dump_dict

_, csv_path, dict_dir = sys.argv

smpdb_dict = {}
with open(csv_path, encoding='utf-8', newline='') as f:
    for cols in csv.reader(f):
        smpdb_id, pw_id, pathway_name, pathway_subject, pathway_description = cols
        smpdb_dict[pathway_name.lower()] = smpdb_id

dump_dict(smpdb_dict, os.path.join(dict_dir, 'smpdb_dict.pkl'))

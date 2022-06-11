import os
import sys
import pickle as pkl
import numpy as np

from utils import dump_dict, load_dict

_, corpus_tsv_in, dict_dir, entities_tsv_in, npy_out = sys.argv


with open(corpus_tsv_in, 'r') as f:
    tsv_lines = f.read().strip().split('\n')

cased_mentions = set()

for line in tsv_lines:
    m_1 = line.split('\t')[2].lower()
    m_2 = line.split('\t')[3].lower()

    cased_mentions.add(m_1)
    cased_mentions.add(m_2)

dict_paths = ('db_dict.pkl', 'atc_dict.pkl', 'up_dict.pkl', 'mesh_dict.pkl') 
d_types = ('DRUGBANK::', 'ATC::', 'UNIPROT::', 'MESH::')
#dict_paths = ('db_dict.pkl',)
#d_types = ('DRUGBANK::',)
d_list = []
for d_path in dict_paths:
    d_list.append(load_dict(os.path.join(dict_dir, d_path)))

all_str_list = []
idx_list = []
for d in d_list:
    keys = list(d.keys())
    prefix = '|||'
    all_str_list.append(prefix.join(keys))

    global_offset = 0
    idx = {}
    for k in keys:
        before_offset = global_offset
        global_offset += len(k)
        for o in range(before_offset, global_offset):
            idx[o] = d[k]
        global_offset += len(prefix)
    idx_list.append(idx)

mention2id = {}
for m in cased_mentions:
    d_idx = 0
    for d, s, i, d_type in zip(d_list, all_str_list, idx_list, d_types):
        if m in d:
            mention2id[m] = d_type + d[m]
            break
        elif m in s:
            mention2id[m] = d_type + i[s.find(m)]
            break
unmatched_mentions = cased_mentions - set(mention2id.keys())
print(len(mention2id) / len(cased_mentions))


threshold = 5
d_idx = d_types.index('MESH::')
for um in unmatched_mentions:
    matched_l = 0
    for x in d_list[d_idx].keys():
        if len(x) < threshold: continue
        if x in um and len(x) > matched_l:
            matched_x = x
            matched_l = len(x)
    if matched_l > 0:
        mention2id[um] = d_types[d_idx] + d_list[d_idx][matched_x]
print(len(mention2id) / len(cased_mentions))


with open(entities_tsv_in, 'r') as f:
    entities_tsv = f.read().strip().split('\n')
    id2entidx = {l.split('\t')[1]: int(l.split('\t')[0]) for l in entities_tsv}
    
UNM_idx = len(entities_tsv)
all_ent_indices = []
unmatched_cnt = 0
for line in tsv_lines:
    ent_indices = []

    unmatched_flg = False
    for idx_ in (2,3):
        m = line.split('\t')[idx_].lower()
        if m in mention2id:
            id_ = mention2id[m]
            if id_ in id2entidx:
                ent_indices.append(id2entidx[id_])
            else:
                ent_indices.append(UNM_idx)
                unmatched_flg = True
        else:
            ent_indices.append(UNM_idx)
            unmatched_flg = True
    all_ent_indices.append(ent_indices)

    if unmatched_flg:
        unmatched_cnt += 1

print('{} {} {:.4f}'.format(unmatched_cnt, len(tsv_lines), 1-unmatched_cnt/len(tsv_lines)))
np.save(npy_out, all_ent_indices)

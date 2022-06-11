import pickle as pkl

def dump_dict(dict_, path):
    with open(path, 'wb') as f:
        pkl.dump(dict_, f)

def load_dict(path):
    with open(path, 'rb') as f:
        d = pkl.load(f)
    return d

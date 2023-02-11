import os, sys
import itertools
import pickle
import numpy as np
import math
from time import strftime, gmtime, time
import json
from pdb import set_trace as bp

def dict_combiner(mydict):
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

class DotDict(dict):
    """dot.notation access to dictionary attributes
    From https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    def __getattr__(*args):
        # Allow nested dicts
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dict_to_file(mydict, fname):
	dumped = json.dumps(mydict, cls=NumpyEncoder)
	with open(fname, 'w') as f:
		json.dump(dumped, f, indent=3)
	return

def file_to_dict(fname):
    with open(fname) as f:
        my_dict = json.load(f)
    return my_dict

def load_data(fname='input.pkl'):
    with open(fname, "rb") as f:
        return pickle.load(f)

def dump_data(out, fname='output.pkl', to_dict=True):
    with open(fname, 'wb') as f:
        pickle.dump(out, f)
    try:
        dict_to_file(out, os.path.splitext(fname)[0]+'.txt')
    except:
        pass
    return

def make_new_dir(x):
    c = 0
    while c >=0 :
        try:
            newdir = x + '_{}'.format(c)
            os.makedirs(newdir)
            c = -1
        except:
            c += 1
    return newdir

def sec2friendly(t):
    return strftime("%H:%M:%S", gmtime(t))

def make_opt_settings(x):
    '''strips _* from dictionary keywords'''
    xnew = {key.split('_')[0]: x[key] for key in x}
    return xnew

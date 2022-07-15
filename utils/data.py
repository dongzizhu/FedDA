import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import os

def load_data(prefix='data', dataset='Amazon', id=0, ifFed=False, path='./'):
    from scripts.data_loader import data_loader
    
    print(os.getcwd())
    dl = data_loader(path + prefix + '/'+ dataset + '_' + str(id))
    
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    return features,\
           adjM, \
            dl

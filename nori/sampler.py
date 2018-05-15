import numpy as np
import os
import cv2
from IPython import embed
import sys
import nori2 as nori
import pickle
import time

info = None
nori_ids = None
nori2label = None
f = nori.Fetcher()

def init(dataset='vggface2', prefix='/unsullied/sharefs/_research_facerec/face-recognition-private-data/training-data-v2/'):
    global nori_ids, info, nori2label
    if dataset == 'vggface2':
        with open(os.path.join(prefix, '{}/latest/info'.format(dataset)), 'rb') as f:
            info = np.array(pickle.load(f))
        with open(os.path.join(prefix, '{}/latest/align5p.nori_id'.format(dataset)), 'rb') as f:
            nori_ids = pickle.load(f)
    else:
        prefix = os.path.join(prefix, 'training-data/align5p')
        if not os.path.exists(prefix):
            if  dataset == 'CASIA':
                prefix = '/unsullied/sharefs/_research_facerec/face-recognition-public-data/CASIA-Webface/'
                dataset = ''
            else:
                raise NotImplementedError

        with open(os.path.join(prefix, dataset, 'align5p.info'), 'rb') as f:
            info = np.array(pickle.load(f))
        with open(os.path.join(prefix, dataset, 'align5p.nori_id'), 'rb') as f:
            nori_ids = pickle.load(f)

    nori2label = np.arange(len(nori_ids))
    for i in range(len(info)):
        for j in range(info[i][0], info[i][1]):
            nori2label[j] = i
    print(info.shape)

def sample_by_img_x1():
    iid = np.random.randint(len(nori_ids))
    nid = nori_ids[iid]
    data = f.get(nid)
    return {'data': data, 'pid': nori2label[iid]}

from occlusion import get as occlusion_get 
def sample_by_img_x1_occlusion():
    iid = np.random.randint(len(nori_ids))
    nid = nori_ids[iid]
    data = occlusion_get(nid)
    return {'data': data, 'pid': nori2label[iid]}


def sample_by_img_x3():
    ref_iid = np.random.randint(len(nori_ids))
    pos_pid = nori2label[ref_iid]
    neg_pid = (np.random.randint(len(info)-1) + pos_pid + 1)%len(info)
    pos_iid = np.random.randint(info[pos_pid][1] - info[pos_pid][0]) + info[pos_pid][0]
    neg_iid = np.random.randint(info[neg_pid][1] - info[neg_pid][0]) + info[neg_pid][0]
    ref_nid = nori_ids[ref_iid]
    pos_nid = nori_ids[pos_iid]
    neg_nid = nori_ids[neg_iid]
    data = [f.get(nid) for nid in [ref_nid, pos_nid, neg_nid]]
    return {'data': data, 'pid': [nori2label[ref_iid], nori2label[pos_iid], nori2label[neg_iid]]}

def sample_x1():
    pid = np.random.randint(len(info))
    iid = np.random.randint(info[pid][1] - info[pid][0]) + info[pid][0]
    nid = nori_ids[iid]
    data = f.get(nid)
    return {'data': data, 'pid': pid}

def sample_xn(n):
    pid = np.random.randint(len(info))
    iid = [np.random.randint(info[pid][1] - info[pid][0]) + info[pid][0] for i in range(n)]
    nid = [nori_ids[i] for i in iid]
    data = [f.get(i) for i in nid]
    return {'data': data, 'pid': pid}

def sample_xn2(n):
    iid = np.random.randint(len(nori_ids))

    #pid = np.random.randint(len(info))
    #iid = np.random.randint(info[pid][1] - info[pid][0]) + info[pid][0]

    iids = [iid]
    pid = nori2label[iid]
    num = info[pid][1] - info[pid][0]
    while True:
        a = np.random.randint(num) + info[pid][0]
        if num == 1 or a!=iid:
            iids.append(a)
            break

    nid = [nori_ids[i] for i in iids]
    data = [f.get(i) for i in nid]
    assert nori2label[iids[0]] == nori2label[iids[1]]
    return {'data': data, 'pid': [pid for i in range(n)]}

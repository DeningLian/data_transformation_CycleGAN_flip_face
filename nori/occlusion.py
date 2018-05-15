import os
import numpy as np
import cv2
import pickle
from meghair.utils.imgproc import imdecode
import nori2 as nori
glassesmap = {}
maskmap = {}

maps = {
    'glasses': glassesmap,
    'mask': maskmap,
}

fs = {
    'origin': None,
    'glasses': None,
    'mask': None,
}

def init():
    origin_ids = list(map(lambda x:x.strip(), open('/home/hza/models/webface_5p_glasses.nori/origin_ids')))
    for i in ['glasses', 'mask']:
        path = '/home/hza/models/webface_5p_{}.nori'.format(i)
        ids = list(map(lambda x:x.strip(), open(os.path.join(path, 'ids'))))
        for a, b in zip(origin_ids, ids):
            maps[i][a] = b
        fs[i] = nori.open(path, 'r')
    fs['origin'] = nori.Fetcher()
    return origin_ids

def get(nid):
    if np.random.randint(2) == 0 or nid not in maps['mask']:
        return fs['origin'].get(nid)
    else:
        return fs['glasses'].get(maps['glasses'][nid])
        if np.random.randint(2) == 0:
            return fs['mask'].get(maps['mask'][nid])
        else:
            return fs['glasses'].get(maps['glasses'][nid])

def load(f, id):
    data = pickle.loads(f.get(id))
    if type(data['img']) is bytes:
        img = imdecode(data['img'])
        data['img'] = img
    if data['img'].shape[0] == 128:
        data['img'] = cv2.resize(data['img'], (256, 256))
    #print(data['img'].min(), data['img'].max())
    data['img'] = np.maximum(0, data['img'])
    data['img'] = np.minimum(255, data['img']).astype('uint8')
    return data

if __name__ == '__main__':
    ids = init()
    for i in ids[-10::-1]:
        img2 = load(fs['glasses'], maps['glasses'][i])['img']
        img1 = load(fs['origin'], i)['img']
        print(img1.shape, img2.shape)
        print(img1.shape, img2.shape)
        cv2.imshow('x', np.concatenate((img1, img2), axis=1))
        cv2.waitKey(0)
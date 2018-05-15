from dpflow import control, OutputPipe
import numpy.random as nr
from meghair.utils.misc import add_rand_seed_entropy, stable_rand_seed
import collections
import random
from dpflow.pipe import MuxPipe
import numpy as np
import nori2 as nori
import pickle
import time
import cv2
import sys,os
import datetime
import argparse
import threading
import traceback
#from sampler import sample_xn as data_sampler
from sampler import init
from IPython import embed
import setproctitle
from meghair.utils.imgproc import imdecode

import torchvision.transforms as transforms

from flip_face import create_CycleGAN, flip_face

setproctitle.setproctitle('data_provider')
parser = argparse.ArgumentParser(
        description='data provider for train and validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--out_pipe', help='output pipe name', default='xxx')
parser.add_argument('--show', action='store_true', help='show images?', default=False)
parser.add_argument('--num-worker', default=1, type=int)
parser.add_argument('--dataset', default='vggface2', type=str, choices=['vggface2', 'CASIA'])
parser.add_argument('--sampler', default='sample_by_img_x1', type=str)
parser.add_argument('--n', default=2, type=int)
args = parser.parse_args()

if args.sampler == 'sample_by_img_x1': 
    from sampler import sample_by_img_x1 as data_sampler
elif args.sampler == 'sample_xn2':
    from sampler import sample_xn2 as data_sampler_
    data_sampler = lambda:data_sampler_(args.n)
elif args.sampler == 'sample_by_img_x3': 
    from sampler import sample_by_img_x3 as data_sampler
elif args.sampler == 'sample_by_img_x1_occlusion': 
    from sampler import sample_by_img_x1_occlusion as data_sampler
else:
    raise NotImplementedError

init(args.dataset)
print('PIPE NAME:', '{}.train'.format(args.out_pipe))

gpu_ids = [0]
G = create_CycleGAN(gpu_ids=gpu_ids)

def main():
    BATCH_SIZE = 256

    if 'occlusion' in args.sampler:
        import occlusion
        occlusion.init()

    if args.show:
        BATCH_SIZE = 16
    #rng = stable_rng(stable_rng)
    #np.random.seed(stable_rand_seed(rng))

    def show(imgs, labels):
        st = np.random.randint(len(imgs) - 10)
        if st % 2 == 1: st = st -1
        img = np.concatenate(imgs[st:st+10].transpose((0,2,3,1)), axis=1)
        print(labels[st:st+10])
        cv2.imwrite('xxx.jpg', img)
        #cv2.waitKey(0)
                

    def preprocess(data):
        raw_imgs = data['data']
        pids = data['pid']
        if type(raw_imgs) is not list:
            raw_imgs = [raw_imgs]
            pids = [pids]

        raw_img = [pickle.loads(i) for i in raw_imgs]
        #raw_img = pickle.loads(raw_imgs)
        imgs = []
        for i in range(len(raw_img)):
            label = pids[i]
            if type(raw_img[i]['img']) is bytes:
                img = imdecode(raw_img[i]['img'])
            else:
                img = raw_img[i]['img']
                img = np.maximum(0, img)
                img = np.minimum(255, img).astype('uint8')
                if img.shape[0] == 128:
                    img = cv2.resize(img, (128, 128))
            # random crop
            base = np.zeros((256+10, 256+10, 3), dtype='uint8')
            if np.random.randint(2):
                base[5:-5, 5:-5, :] = img
            else:
                base[5:-5, 5:-5, :] = img[:,::-1,:]
            dx = np.random.randint(11)
            dy = np.random.randint(11)
            img = cv2.resize(img[dx:dx+256, dy:dy+256, :], (256, 256))
            #img = flip_face(G, img, gpu_ids) 
            img = cv2.resize(img, (224, 224))
            imgs.append(img.transpose((2, 0, 1)))
        return {'img':np.array(imgs), 'label':np.array(pids)}

    pipe = OutputPipe('{}.train'.format(args.out_pipe), buffer_size=4)    
    with control(io=[pipe]):
        while True:
            imgs = []
            labels = []
            for batch_id in range(BATCH_SIZE):
                try:
                    data = data_sampler()
                    data = preprocess(data)
                    imgs.append(data['img'])
                    labels.append(data['label'])
                except Exception:
                    #print('###', data)
                    tb = traceback.format_exc()
                    print(tb)
                    exit(0)
            
            imgs = np.array(imgs)
            labels = np.array(labels)
            imgs = np.concatenate(imgs, axis=0)
            labels = np.concatenate(labels, axis=0)
            if args.show:
                show(imgs, labels)
            if not pipe.full():
                pipe.put_pyobj({'img':imgs, 'label':labels})
            
if args.num_worker == 1:
    main()
else:
    pids = []
    for i in range(args.num_worker):
        add_rand_seed_entropy(random.random())
        #random.random()

        pid = os.fork()
        if pid == 0: # chile
            main()
        else: # master
            pids.append(pid)

    for pid in pids:
        os.waitpid(pid, 0)


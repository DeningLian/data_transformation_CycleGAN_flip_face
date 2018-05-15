import numpy as np
import pickle
from meghair.train.base import DatasetProviderBase
from dpflow import Controller, InputPipe, control, Combiner
import time
import numpy as np
from IPython import embed
from .utils import MI
import cv2

SHAPE = (3, 224, 224)

class DatasetProvider(DatasetProviderBase):
    def __init__(self, pipe_prefix, batch_size):
        self.batch_size = batch_size
        self._input_pipe = InputPipe(pipe_prefix, buffer_size=32, policy={'MAX_PEER': 64})
        def receiver(pipe):
            with control(io=[pipe]):
                yield from pipe

        self._receiver = receiver(self._input_pipe)

        def filler(buf_dict, idx, data_dict):
            buf_dict['img'][:] = data_dict['img']#.transpose((0,3,1,2))
            buf_dict['label'][:] = data_dict['label']

        self.descriptor = {'label':{'shape': (batch_size), 'dtype':'int32'},
                          'img': {'shape': (batch_size,) + SHAPE,'dtype': 'uint8'}}
                            #'img': {'shape': (batch_size, 3, 112,96),'dtype': 'uint8'}}
        self._combiner = Combiner(self._receiver, 1, self.descriptor, filler)
        self._diter = iter(self._combiner)

    def get_epoch_minibatch_iter(self):
        return next(self._diter)

    def __del__(self):
        del self._diter
        del self._receiver
        del self._combiner

NUM_CLASS = {
    'vggface2': 9131,
    'pairwise2': 10575,
    'pairwise3': 10575,
    'occlusion': 10575
}

def make_dp(dataset):
    if dataset == 'vggface2':
        dp = DatasetProvider('{}.hza.x3.train'.format(dataset), 256)
    elif dataset == 'pairwise2':
        import os
        if os.path.exists('/unsullied/sharefs/_research_facerec/face-recognition-private-data/training-data/align5p/CASIA/') and False:
            dp = DatasetProvider('hza.pairwise.sample.train', 256 * 2)
        else:
            dp = DatasetProvider('hza.pairwise.nanjing.train', 256 * 2)
    elif dataset == 'pairwise3':
        import os
        dp = DatasetProvider('hza.pairwise3.nanjing.train', 256 * 3)
    elif dataset == 'occlusion':
        import os
        dp = DatasetProvider('hza.occlusion.nanjing.train', 256)
    else:
        raise NotImplementedError
    while True:

        data = dp.get_epoch_minibatch_iter()
        img = MI(data['img'].astype('float32'))
        img = img.view((256, -1,) + SHAPE).transpose(1, 0).contiguous().view((-1,)+SHAPE)
        
        label = MI(data['label']).long()
        label = label.view((256, -1)).transpose(1, 0).contiguous().view((-1,))
        yield {
            'img': img,
            'label': label
        }
import numpy as np
import torch
import torchvision 
import torch.nn as nn
from torch.autograd import Variable

from data_provider import create_data_loader
from model import create_model 
from visualizer import Visualizer

import time
from tqdm import tqdm
from IPython import embed

#dataset = opt.data

EPOCH_COUNT = 45
EPOCH = 200
BATCH = 200

BATCH_SIZE = 256

LR = 0.0005
BETA1 = 0.5

SAVE_EPOCH_FREQ = 5 

name = 'origin'
dataset = 'vggface2'

gpu_ids = [i for i in range(4)]
data_loader = create_data_loader(dataset)
model = create_model(name, gpu_ids, LR, BETA1, BATCH_SIZE, BATCH, EPOCH_COUNT)
visualizer = Visualizer(name)

model.save('latest')

for epoch in range(EPOCH_COUNT+1, EPOCH+1):
    epoch_start_time = time.time()
    t_data = 0
    model.reset_errors()

    with tqdm(total=BATCH) as pbar:
        for i in range(BATCH):
            iter_start_time = time.time()
            data = next(data_loader)
            t_data += time.time() - iter_start_time

            output = model.set_input(data)
            model.backward()

            pbar.set_description("training score = %.3f"% model.train_score)
            pbar.update()
    
    t = time.time() - epoch_start_time
    print("End of epoch %d / %d \t Time Taken: %d sec" %
          (epoch, EPOCH, time.time() - epoch_start_time))
    errors = model.get_current_errors()
    visualizer.print_current_errors(epoch, t, t_data, errors)
    visualizer.plot_current_errors(epoch, errors)

    if epoch % SAVE_EPOCH_FREQ == 0:
        print("saving the model at the end of epoch %d" % epoch)
        model.save('latest')
        model.save(epoch)

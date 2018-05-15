import torch 
import torchvision  
import torch.nn as nn
import torchvision.transforms as transforms 
from torch.autograd import Variable 
import numpy as np
import cv2 
import os

from IPython import embed

class ResNetModel(nn.Module):
    def __init__(self, name, gpu_ids, LR, BETA1, BATCH_SIZE, BATCH, EPOCH_COUNT):
        super(ResNetModel, self).__init__()
        self.name = name
        self.save_dir = os.path.join('checkpoints/', self.name)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        net = torchvision.models.resnet34(pretrained=True)
        del net.fc
        net.fc = nn.Linear(512, 10000)

        self.net = nn.DataParallel(net, gpu_ids)
        if EPOCH_COUNT > 0:
            self.load(EPOCH_COUNT)
        self.net.cuda()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=LR, betas=(BETA1, 0.999))
        self.Loss = nn.CrossEntropyLoss()
        print(self.net)

        self.gpu_ids = gpu_ids
        self.BATCH_SIZE = BATCH_SIZE
        self.BATCH = BATCH


    def set_input(self, data):
        self.x = data['img']
        self.target = data['label']

    def backward(self):
        output = self.net(self.x)

        predict = torch.max(output, 1)[1].data.cpu().numpy()
        target = self.target.data.cpu().numpy()

        correct = sum(predict == target)
        self.train_score = correct / self.BATCH_SIZE
        self.epoch_score += self.train_score

        self.loss = self.Loss(output, self.target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_current_errors(self):
        self.epoch_score /= self.BATCH
        return {'loss': self.loss, 'acc': self.epoch_score}
        #visualizer.loss = self.loss
    
    def get_current_accuracy(self, TEST_SIZE=10000, sampler='sample_by_img_x1'):
        from nori.sampler import sample_by_img_x1 as data_sampler
        from nori.sender import preprocess
        imgs = []
        labels = []
        for batch_id in range(TEST_SIZE):
            try:
                data = data_sampler()
                data = *preprocess(data)
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
            
            output = self.net(imgs)
            predict = torch.max(output, 1)[1].data.cpu().numpy()
            target = self.target.data.cpu().numpy()

            self.accuracy = sum(predict == target)

    def reset_errors(self):
        self.epoch_score = 0

    def save(self, label):
        save_filename = '%s.pth' % label
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.net.cpu().state_dict(), save_path)
        if len(self.gpu_ids) and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def load(self, label):
        load_filename = '%s.pth' % label
        load_path = os.path.join(self.save_dir, load_filename)
        self.net.load_state_dict(torch.load(load_path))


def create_model(name, gpu_ids, LR, BETA1, BATCH_SIZE, BATCH, EPOCH_COUNT):
    model = ResNetModel(name, gpu_ids, LR, BETA1,
                        BATCH_SIZE, BATCH, EPOCH_COUNT)
    return model

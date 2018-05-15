import os 
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
import cv2
from PIL import Image

from CycleGAN.models.cycle_gan_model import CycleGANModel
from CycleGAN.models import networks
import CycleGAN.util.util as util

import time

from IPython import embed

def create_CycleGAN(name='face_up2down_cyclegan_more_data', gpu_ids=[]):
    model = CycleGANModel()
    model.gpu_ids = gpu_ids
    if len(model.gpu_ids) > 0:
        torch.cuda.set_device(model.gpu_ids[0])

    model.netG_A = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', gpu_ids)
    #print(model.netG_A)
    
    checkpoints_dir = os.path.join("CycleGAN/checkpoints/", name)
    which_epoch = 'latest'
    model.netG_A.load_state_dict(torch.load(os.path.join(checkpoints_dir, "latest_net_G_A.pth")))

    return model.netG_A

def flip_face(G, img, gpu_ids=[]):
    """
    input: cv2
    processing: PIL 256x256
    output: cv2 256x256
    """
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))    

    transform_list = [transforms.Resize((256, 256)), 
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    real_A = torch.unsqueeze(transform(img), 0)
    real_A = Variable(real_A)
    if len(gpu_ids) > 0:
        real_A = real_A.cuda()
    fake_B = G(real_A).data
    #print(real_A.shape)
    #print(fake_B.shape)
    result = Image.fromarray(util.tensor2im(fake_B))
    result = cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2BGR)  
    #cv2.imwrite("fake_B.png", result)
    return result
    
    
if __name__ == "__main__":
    gpu_ids = [0,1]
    print("gpu_ids = ", gpu_ids)
    beginning_time = time.time()
    G = create_CycleGAN(gpu_ids=gpu_ids)
    print("intial time = %.3f sec" % (time.time() - beginning_time))

    import numpy as np
    #cv2.imwrite("real_A.png", img)
    for i in range(1000):
        img = cv2.imread("CycleGAN/checkpoints/face_up2down_cyclegan_more_data/web/images/epoch%03d_real_A.png"
                         % (np.random.randint(500)+1))
        start_time = time.time()
        flip_face(G, img, gpu_ids=gpu_ids)
        print("running time = %.3f sec" % (time.time() - start_time))
    

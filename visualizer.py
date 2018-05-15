from tensorboardX import SummaryWriter
import numpy as np
import os, time

class Visualizer():
    def __init__(self, name): 
        self.vis = SummaryWriter()

        self.name = name
        self.save_dir = os.path.join('checkpoints/', self.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==================== Train Loss (%s) ======================\n' % now)

    def plot_current_errors(self, epoch, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'epoch': [], 'loss': [], 'acc': []}

        loss = errors['loss']
        acc = errors['acc']
    
        self.plot_data['epoch'].append(epoch)
        self.plot_data['loss'].append(loss)
        self.plot_data['acc'].append(acc)

        self.vis.add_scalars(
            'loss',
            {'loss': loss},
            epoch
        )
        self.vis.add_scalars(
            'accuracy',
            {'acc': acc}, 
            epoch
        )
        
    def print_current_errors(self, epoch, t, t_data, errors):
        loss = errors['loss']
        acc = errors['acc']

        message = '(epoch: %d, time: %.3f, t_data: %.3f, loss: %.6f, accuracy: %.2f)' % (epoch, t, t_data, loss, acc)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)

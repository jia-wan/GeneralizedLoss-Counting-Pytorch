import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import StepLR
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vgg import vgg19
from datasets.crowd import Crowd, train_val, get_im_list
from geomloss import SamplesLoss
import inspect

print(inspect.getfile(SamplesLoss))

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def grid(H, W, stride):
    coodx = torch.arange(0, W, step=stride) + stride / 2
    coody = torch.arange(0, H, step=stride) + stride / 2
    y, x = torch.meshgrid( [  coody.type(dtype) / 1, coodx.type(dtype) / 1 ] )
    return torch.stack( (x,y), dim=2 ).view(-1,2)

def per_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    s = (x_col[:,:,:,-1] + y_lin[:,:,:,-1]) / 2
    s = s * 0.2 + 0.5
    return (torch.exp(C/s) - 1)

def exp_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    return (torch.exp(C/scale) - 1.)

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class EMDTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        global scale
        scale = args.scale
        if args.cost == 'exp':
            self.cost = exp_cost
        elif args.cost == 'per':
            self.cost = per_cost

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x
                                  ) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(self.args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False), drop_last=True)
                            for x in ['train', 'val']}

        self.model = vgg19()

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.blur = args.blur
        self.criterion = SamplesLoss(blur=args.blur, scaling=args.scaling, debias=False, backend='tensorized', cost=self.cost, reach=args.reach, p=args.p)

        self.log_dir = os.path.join('./runs', args.save_dir.split('/')[-1]) 
        self.writer = SummaryWriter(self.log_dir)

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae_list = []
        self.best_mae = {}
        self.best_mse = {}
        self.best_epoch = {}
        for stage in ['train', 'val']:
            self.best_mae[stage] = np.inf
            self.best_mse[stage] = np.inf
            self.best_epoch[stage] = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch(epoch)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch(stage='val')

    def train_eopch(self, epoch=0):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        shape = (1,int(512/self.args.downsample_ratio),int(512/self.args.downsample_ratio))

        if epoch < 10:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] >= 0.1*self.args.lr:
                    param_group['lr'] = self.args.lr * (epoch + 1) / 10
        print('learning rate: {}, batch size: {}'.format(self.optimizer.param_groups[0]['lr'], self.args.batch_size))
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]
            shape = (inputs.shape[0],int(inputs.shape[2]/self.args.downsample_ratio),int(inputs.shape[3]/self.args.downsample_ratio))

            with torch.autograd.set_grad_enabled(True):

                outputs = self.model(inputs)
                
                cood_grid = grid(outputs.shape[2], outputs.shape[3], 1).unsqueeze(0) * self.args.downsample_ratio + (self.args.downsample_ratio / 2)
                cood_grid = cood_grid.type(torch.cuda.FloatTensor) / float(self.args.crop_size)
                i = 0
                emd_loss = 0
                point_loss = 0
                pixel_loss = 0
                entropy = 0
                for p in points:
                    if len(p) < 1:
                        gt = torch.zeros((1, shape[1], shape[2])).cuda()
                        point_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                        pixel_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                        emd_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                    else:
                        gt = torch.ones((1, len(p), 1)).cuda()
                        cood_points = p.reshape(1, -1, 2) / float(self.args.crop_size) 
                        A = outputs[i].reshape(1, -1, 1)
                        l, F, G = self.criterion(A, cood_grid, gt, cood_points)

                        C = self.cost(cood_grid, cood_points)
                        PI = torch.exp((F.repeat(1,1,C.shape[2])+G.permute(0,2,1).repeat(1,C.shape[1],1)-C).detach()/self.args.blur**self.args.p)*A*gt.permute(0,2,1)
                        entropy += torch.mean((1e-20+PI) * torch.log(1e-20+PI))
                        AE = PI
                        AE = AE.sum(1).reshape(1,-1,1)
                        emd_loss += (torch.mean(l) / shape[0])
                        if self.args.d_point == 'l1':
                            point_loss += torch.sum(torch.abs(PI.sum(1).reshape(1,-1,1)-gt)) / shape[0] 
                        else:
                            point_loss += torch.sum((PI.sum(1).reshape(1,-1,1)-gt)**2) / shape[0] 
                        if self.args.d_pixel == 'l1':
                            pixel_loss += torch.sum(torch.abs(PI.sum(2).reshape(1,-1,1).detach()-A)) / shape[0] 
                        else:
                            pixel_loss += torch.sum((PI.sum(2).reshape(1,-1,1).detach()-A)**2) / shape[0] 
                    i += 1

                loss = emd_loss + self.args.tau*(pixel_loss + point_loss) + self.blur*entropy


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                outputs = torch.mean(outputs, dim=1)
                pre_count = torch.sum(outputs[-1]).detach().cpu().numpy()
                res = (pre_count - gd_count[-1]) #gd_count
                if step % 200 == 0:
                    print(res, pre_count, gd_count[-1], point_loss.item(), pixel_loss.item(), loss.item())
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                    time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models


    def val_epoch(self, stage='val'):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        epoch_fore = []
        epoch_back = []
        # Iterate over data.
        if stage == 'val':
            dataloader = self.dataloaders['val']
        for inputs, points, name in dataloader:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                points = points[0].type(torch.LongTensor)
                res = len(points) - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.writer.add_scalar(stage+'/mae', mae, self.epoch)
        self.writer.add_scalar(stage+'/mse', mse, self.epoch)
        logging.info('{} Epoch {}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(stage, self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()

        if ( mae) < ( self.best_mae[stage]):
            self.best_mse[stage] = mse
            self.best_mae[stage] = mae
            self.best_epoch[stage] = self.epoch 
            logging.info("{} save best mse {:.2f} mae {:.2f} model epoch {}".format(stage,
                                                                            self.best_mse[stage],
                                                                            self.best_mae[stage],
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_{}.pth').format(stage))
        # print log info
        logging.info('Val: Best Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.best_epoch['val'], self.best_mse['val'], self.best_mae['val'], time.time()-epoch_start))




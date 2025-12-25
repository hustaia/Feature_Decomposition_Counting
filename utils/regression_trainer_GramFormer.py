from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vgg_graph import GramFormer, FD_GramFormer
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from losses.kl_loss import compute_kl_loss
from math import ceil
import random


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    images_low = torch.stack(transposed_batch[1], 0)
    points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[3]
    st_sizes = torch.FloatTensor(transposed_batch[4])
    return images, images_low, points, targets, st_sizes


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        setup_seed(args.seed)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x.replace('val', 'val')),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(False))
                                        #   pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = GramFormer(args.topk, args.usenum)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        import ptflops
        macs, params = ptflops.get_model_complexity_info(self.model, (3, 224, 224), as_strings=True,
                                                         print_per_layer_stat=True, verbose=True)
        
        print(f'Computational complexity: {macs}')
        print(f'Number of parameters: {params}')

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device), strict=False)

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = args.save_all
        self.best_count = 0
        

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            # self.val_epoch()
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        if False:
            for step, (inputs, _, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
                inputs = inputs.to(self.device)
                st_sizes = st_sizes.to(self.device)
                gd_count = np.array([len(p) for p in points], dtype=np.float32)
                points = [p.to(self.device) for p in points]
                targets = [t.to(self.device) for t in targets]
    
                with torch.set_grad_enabled(True):
                    outputs, pe_list = self.model(inputs)
                    prob_list = self.post_prob(points, st_sizes)
                    loss = self.criterion(prob_list, targets, outputs)
                    for pe in pe_list:
                        loss_pe = torch.var(pe, dim=-1)
                        # loss_pe = torch.sum(loss_pe[loss_pe>0.1])
                        loss_pe = torch.sum(loss_pe)
                        loss += 0.1*loss_pe
                    loss.backward()
      
                self.optimizer.step()
                self.optimizer.zero_grad()
    
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
                    
                if step % 100 == 0:
                    print(('Epoch: [{0}][{1}/{2}]\tLoss {3}\t')
                        .format(self.epoch, step, len(self.dataloaders['train']), loss.item()))
    
            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time()-epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models
        else:
            for step, (inputs, inputs1, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
                inputs = inputs.to(self.device)
                inputs1 = inputs1.to(self.device)
                st_sizes = st_sizes.to(self.device)
                gd_count = np.array([len(p) for p in points], dtype=np.float32)
                points = [p.to(self.device) for p in points]
                targets = [t.to(self.device) for t in targets]
    
                with torch.set_grad_enabled(True):
                    outputs, pe_list = self.model(inputs)
                    outputs1, pe_list1 = self.model(inputs1)
                    prob_list = self.post_prob(points, st_sizes)
                    loss = self.criterion(prob_list, targets, outputs) + self.criterion(prob_list, targets, outputs1)
                    pe_list.extend(pe_list1)
                    for pe in pe_list:
                        loss_pe = torch.var(pe, dim=-1)
                        # loss_pe = torch.sum(loss_pe[loss_pe>0.1])
                        loss_pe = torch.sum(loss_pe)
                        loss += 0.1*loss_pe
                    loss.backward()
      
                self.optimizer.step()
                self.optimizer.zero_grad()
    
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
                    
                if step % 100 == 0:
                    print(('Epoch: [{0}][{1}/{2}]\tLoss {3}\t')
                        .format(self.epoch, step, len(self.dataloaders['train']), loss.item()))
    
            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time()-epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            c_size = 1024
            if h >= c_size or w >= c_size:
                h_stride = int(ceil(1.0 * h / c_size))
                w_stride = int(ceil(1.0 * w / c_size))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse, self.best_mae, self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))


class RegTrainer_FD(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        setup_seed(args.seed)
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
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(False))
                                        #   pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = FD_GramFormer(args.topk, args.usenum)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        import ptflops
        macs, params = ptflops.get_model_complexity_info(self.model, (3, 224, 224), as_strings=True,
                                                         print_per_layer_stat=True, verbose=True)
        
        print(f'Computational complexity: {macs}')
        print(f'Number of parameters: {params}')

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device), strict=False)

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = args.save_all
        self.best_count = 0
        

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            # self.val_epoch()
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs_high, inputs_low, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs_high = inputs_high.to(self.device)
            inputs_low = inputs_low.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                feat_high, feat_high_common, feat_high_unique, outputs_high, pe_list_high, \
                    feat_low, feat_low_common, feat_low_unique, outputs_low, pe_list_low = self.model(inputs_high, inputs_low)
                prob_list = self.post_prob(points, st_sizes)
                density_loss = self.criterion(prob_list, targets, outputs_high) + self.criterion(prob_list, targets, outputs_low)
                loss_pes = 0
                pe_list_high.extend(pe_list_low)
                for pe in pe_list_high:
                    loss_pe = torch.var(pe, dim=-1)
                    # loss_pe = torch.sum(loss_pe[loss_pe>0.1])
                    loss_pe = torch.sum(loss_pe)
                    loss_pes += loss_pe
                
                consistency_loss = compute_kl_loss(feat_high_common, feat_low_common)
                #consistency_loss = compute_kl_loss(feat_high_common, feat_low_common) - compute_kl_loss(feat_high_unique, feat_low_unique)
                recon_loss = compute_kl_loss(feat_high, torch.mul(feat_high_common, feat_high_unique)) + \
                            compute_kl_loss(feat_low, torch.mul(feat_low_common, feat_low_unique))
                loss = density_loss + 0.1 * loss_pes + self.args.lamda * (consistency_loss + recon_loss)
                loss.backward()
  
            self.optimizer.step()
            self.optimizer.zero_grad()

            N = inputs_high.size(0)
            pre_count = torch.sum(outputs_high.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            epoch_loss.update(loss.item(), N)
            epoch_mse.update(np.mean(res * res), N)
            epoch_mae.update(np.mean(abs(res)), N)
                
            if step % 100 == 0:
                print(('Epoch: [{0}][{1}/{2}]\tLoss {3}\t')
                    .format(self.epoch, step, len(self.dataloaders['train']), loss.item()))

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            c_size = 1024
            if h >= c_size or w >= c_size:
                h_stride = int(ceil(1.0 * h / c_size))
                w_stride = int(ceil(1.0 * w / c_size))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse, self.best_mae, self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
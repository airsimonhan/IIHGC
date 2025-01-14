import yaml
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import sys
import traceback
import time
import shutil
import inspect
from collections import OrderedDict
import pickle
import glob
import utils
from tqdm import tqdm
from loss import * #coxph_loss, mse_loss
import json


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='multi-modal survival prediction')
    parser.add_argument('--config', default='config/gsz/wsi_with_ct.yaml', help='path to the configuration file')
    parser.add_argument('--work_dir',default='./work_dir/',help='the work folder for storing results')
    parser.add_argument('--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--print_log',default=True, help='print logging or not')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=1, help='the interval for eval')
    # data_loader
    parser.add_argument('--n_fold', default=5, help='the num of fold for cross validation')
    parser.add_argument('--dataset', default='dataset.WSI_Dataset.SlidePatch', help='data set will be used')
    parser.add_argument('--data_seed',type=int, default=1, help='random seed for n_fold dataset')
    parser.add_argument('--WSI_info_list_file', help='path to the information list of WSI sample')
    parser.add_argument('--WSI_patch_ft_dir', help='path to the feature of WSI patch')
    parser.add_argument('--WSI_patch_coor_dir', help='path to the coordinate of WSI patch')
    parser.add_argument('--center', default=['TCGA-LUSC'],nargs='+', help='the center of data')
    parser.add_argument('--clinical', default=False, type=bool, help='if use clinical data')

    parser.add_argument('--num_worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')

    # model
    parser.add_argument('--H_coors', default=False, help='if use the coors of patches to create H')
    parser.add_argument('--intra_model', default=None, help='the intra_model will be used')
    parser.add_argument('--intra_model_args', default=dict(), help='the arguments of intra_model')
    parser.add_argument('--intra_weights', default=None, help='the intra_weights for network initialization')
    parser.add_argument('--ignore_intra_weights', type=str, default=[], nargs='+',help='the name of intra_weights which will be ignored in the initialization')
    parser.add_argument('--inter_model', default=None, help='the inter_model will be used')
    parser.add_argument('--inter_model_args', default=dict(), help='the arguments of inter_model')
    parser.add_argument('--inter_weights', default=None, help='the inter_weights for network initialization')
    parser.add_argument('--ignore_inter_weights', type=str, default=[], nargs='+',help='the name of inter_weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--step', type=int, default=100, nargs='+',help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=300, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss', type=str, default='loss.mse_loss', help='the type of loss function')

    return parser

class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        self.load_data()
        self.load_model()
        self.load_optimizer()

        self.intra_lr = self.arg.base_lr
        self.inter_lr = self.arg.base_lr
        self.inter_best_i_fold_c_index_s2 = 0
        self.inter_best_i_fold_c_index_s2_epoch = 0
        self.inter_best_i_fold_c_index = 0
        self.inter_best_i_fold_c_index_epoch = 0
        self.intra_best_i_fold_c_index = 0
        self.intra_best_i_fold_c_index_epoch = 0
        self.inter_best_c_index = 0
        self.inter_best_i_fold = 0
        self.inter_best_epoch = 0
        self.intra_best_c_index = 0
        self.intra_best_i_fold = 0
        self.intra_best_epoch = 0

        self.inter_model = self.inter_model.cuda(self.output_device)
        self.intra_model = self.intra_model.cuda(self.output_device)
        self.loss = import_class(self.arg.loss)()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.inter_model = nn.DataParallel(
                    self.inter_model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
                self.intra_model = nn.DataParallel(
                    self.intra_model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_data(self):
        dataset = import_class(self.arg.dataset)
        self.data_loader = dict()
        WSI_info_list, self.survival_time_max, self.survival_time_min = utils.get_WSI_sample_list(self.arg.WSI_info_list_file, self.arg.center,self.arg.WSI_patch_ft_dir,self.arg.WSI_patch_coor_dir,clinical=self.arg.clinical)
        n_fold_train_list, n_fold_val_list = utils.get_n_fold_data_list(WSI_info_list,self.arg.n_fold,self.arg.data_seed,clinical=self.arg.clinical)

        self.data_loader['train'] = []
        self.data_loader['val'] = []
        for i in range(len(n_fold_train_list)):
            # if self.arg.phase == 'train':
            self.data_loader['train'].append(torch.utils.data.DataLoader(
                dataset=dataset(n_fold_train_list[i], self.survival_time_max, self.survival_time_min,clinical=self.arg.clinical),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed))
            self.data_loader['val'].append(torch.utils.data.DataLoader(
                dataset=dataset(n_fold_val_list[i], self.survival_time_max, self.survival_time_min,clinical=self.arg.clinical),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed))
    def load_model(self, i=0):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.intra_model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.intra_model = Model(**self.arg.intra_model_args)
        print(self.intra_model)

        Model = import_class(self.arg.inter_model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        if isinstance(self.arg.inter_model_args, str):
            self.arg.inter_model_args = json.loads(self.arg.inter_model_args)
        self.inter_model = Model(**self.arg.inter_model_args)
        print(self.inter_model)


        if self.arg.inter_weights:
            self.print_log('Load weights from {}.'.format(self.arg.inter_weights))
            if '.pkl' in self.arg.inter_weights:
                with open(self.arg.inter_weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                inter_weights = os.path.join(self.arg.inter_weights, 'inter_'+str(i)+'_fold_best_model.pt')
                weights = torch.load(inter_weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_inter_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.inter_model.load_state_dict(weights)
            except:
                state = self.inter_model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.inter_model.load_state_dict(state)

        if self.arg.intra_weights:
            intra_weights = os.path.join(self.arg.intra_weights, str(i)+'_fold_best_model.pt')
            self.print_log('Load weights from {}.'.format(intra_weights))
            if '.pkl' in intra_weights:
                with open(intra_weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(intra_weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_intra_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.intra_model.load_state_dict(weights)
            except:
                state = self.intra_model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.intra_model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.inter_optimizer = optim.SGD(
                self.inter_model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.inter_optimizer = optim.Adam(
                self.inter_model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.inter_scheduler = lr_scheduler.StepLR(self.inter_optimizer, step_size=self.arg.step, gamma=self.arg.lr_decay_rate)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    def concat(self,a,b):
        if a is None:
            return b
        else:
            a = torch.concat((a,b),dim=0)
            return a
    def compute_loss(self, sorted_output, sorted_gt, sorted_status, model, features):
        loss = (self.loss(sorted_output, sorted_status)).sum()
        return loss
    def train(self, epoch, i_fold, save_model=False, train=True):
        self.inter_model.train()
        self.intra_model.eval()
        self.print_log('Training epoch: {} , n_fold: {}'.format(epoch + 1, i_fold))
        loader = self.data_loader['train'][i_fold]

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)
        
        if self.train_gt_value is None:
            for batch_idx, (features, survival_time, status, coors, attr, id) in enumerate(process):
                # if status.sum() == 0:
                #     continue
                with torch.no_grad():
                    features = features.float().cuda(self.output_device)
                    survival_time = survival_time.float().cuda(self.output_device)
                    coors = coors.float().cuda(self.output_device)
                    status = status.long().cuda(self.output_device)
                    attr = attr.long().cuda(self.output_device)
                timer['dataloader'] += self.split_time()

                # forward
                if self.arg.H_coors:
                    output, output_fts = self.intra_model(features, coors)
                else:
                    output, output_fts = self.intra_model(features)  # ,wloss ,label
                if self.train_wsi_fts is None:
                    self.train_wsi_fts = output_fts.detach()
                    self.train_wsi_risk = output.detach()
                    self.train_wsi_st = survival_time
                    self.train_wsi_status = status
                    self.train_attr = attr
                else:
                    self.train_wsi_fts = torch.concat((self.train_wsi_fts, output_fts.detach()), dim=0)
                    self.train_wsi_risk = torch.concat((self.train_wsi_risk, output.detach()), dim=0)
                    self.train_wsi_st = torch.concat((self.train_wsi_st, survival_time), dim=0)
                    self.train_wsi_status = torch.concat((self.train_wsi_status, status), dim=0)
                    self.train_attr = torch.concat((self.train_attr, attr), dim=0)
                sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(survival_time,
                                                                                                      output, status,
                                                                                                      output_fts)
                intra_loss = self.compute_loss(sorted_output, sorted_gt, sorted_status, self.intra_model,
                                               sorted_output_fts)

                torch.autograd.grad(outputs=intra_loss, inputs=self.intra_model.parameters(), retain_graph=False, allow_unused=True)#

                self.train_intra_loss_value.append(intra_loss.data.item())
                timer['model'] += self.split_time()

                self.train_output_value = self.concat(self.train_output_value, sorted_output)
                self.train_status_value = self.concat(self.train_status_value, sorted_status)
                self.train_gt_value = self.concat(self.train_gt_value, sorted_gt)

                # statistics
                timer['statistics'] += self.split_time()

        if train: 
            inter_output_s2, inter_output_s1s2, inter_output_fts = self.inter_model(self.train_wsi_fts,self.train_wsi_risk,attr=self.train_attr)#
            sorted_wsi_st, sorted_inter_output, sorted_wsi_status, sorted_inter_output_fts = utils.sort_survival_time(self.train_wsi_st, inter_output_s2, self.train_wsi_status, inter_output_fts)
            inter_loss = self.compute_loss(sorted_inter_output, sorted_wsi_st, sorted_wsi_status, self.inter_model, sorted_inter_output_fts)

            self.inter_optimizer.zero_grad()
            inter_loss.backward()
            self.inter_optimizer.step()
            self.inter_lr = self.inter_optimizer.param_groups[0]['lr']

            # statistics of time consumption and loss
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }

            intra_c_index = utils.accuracytest(self.train_gt_value,-self.train_output_value,self.train_status_value)
            inter_c_index_s2 = utils.accuracytest(self.train_wsi_st,-inter_output_s2,self.train_wsi_status)
            inter_c_index_s1s2 = utils.accuracytest(self.train_wsi_st,-inter_output_s1s2,self.train_wsi_status)
            self.print_log(
                '\tMean training intra_loss: {:.4f}.  Mean intra_c-index: {:.2f}%.'.format(np.mean(self.train_intra_loss_value), np.mean(intra_c_index) * 100))
            self.print_log(
                '\tMean training inter_loss: {:.4f}.  Mean stage 2 inter_c-index: {:.2f}%. Mean stage 1-2 c-index: {:.2f}%. inter_lr: {:.8f}'.format(inter_loss, np.mean(inter_c_index_s2) * 100,np.mean(inter_c_index_s1s2) * 100,self.inter_lr))
            self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.inter_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, 'inter-'+str(i_fold)+'-runs-' + str(epoch+1) + '.pt'))
        
    def eval(self, epoch, i_fold, save_model=False):
        self.inter_model.eval()
        self.intra_model.eval()
        self.print_log('Eval epoch: {},  n_fold: {}'.format(epoch + 1, i_fold))
        process = tqdm(self.data_loader['val'][i_fold], ncols=40)

        if self.eval_gt_value is None:
            for batch_idx, (features, survival_time, status, coors, attr, id) in enumerate(process):
                with torch.no_grad():
                    features = features.float().cuda(self.output_device)
                    survival_time = survival_time.float().cuda(self.output_device)
                    coors = coors.float().cuda(self.output_device)
                    status = status.long().cuda(self.output_device)
                    attr = attr.long().cuda(self.output_device)
                    if isinstance(self.intra_model, torch.nn.DataParallel):
                        if self.arg.H_coors:
                            output, output_fts = self.intra_model.module.forward(features, coors)
                        else:
                            output, output_fts = self.intra_model.module.forward(features)
                    else:
                        if self.arg.H_coors:
                            output, output_fts = self.intra_model.forward(features, coors)
                        else:
                            output, output_fts = self.intra_model.forward(features)
                    if self.eval_wsi_fts is None:
                        self.eval_wsi_fts = output_fts.detach()
                        self.eval_wsi_risk = output.detach()
                        self.eval_wsi_st = survival_time
                        self.eval_wsi_status = status
                        self.eval_wsi_id = id
                        self.eval_attr = attr
                    else:
                        self.eval_wsi_fts = torch.concat((self.eval_wsi_fts, output_fts.detach()), dim=0)
                        self.eval_wsi_risk = torch.concat((self.eval_wsi_risk, output.detach()), dim=0)
                        self.eval_wsi_st = torch.concat((self.eval_wsi_st, survival_time), dim=0)
                        self.eval_wsi_status = torch.concat((self.eval_wsi_status, status), dim=0)
                        self.eval_wsi_id = self.eval_wsi_id + id
                        self.eval_attr = torch.concat((self.eval_attr, attr), dim=0)

                    sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(survival_time,
                                                                                                          output,
                                                                                                          status,
                                                                                                          output_fts)

                    intra_loss = self.compute_loss(sorted_output, sorted_gt, sorted_status, self.intra_model,
                                                   sorted_output_fts)

                    if status.sum() == 0:
                        intra_loss[intra_loss != intra_loss] = 0  # turn nan to 0
                    self.eval_intra_loss_value.append(intra_loss.data.item())
                    self.eval_output_value = self.concat(self.eval_output_value, sorted_output)
                    self.eval_status_value = self.concat(self.eval_status_value, sorted_status)
                    self.eval_gt_value = self.concat(self.eval_gt_value, sorted_gt)

        
        with torch.no_grad():
            inter_output_s2,inter_output_s1s2, inter_output_fts = self.inter_model(self.eval_wsi_fts,self.eval_wsi_risk,self.train_wsi_fts,self.train_wsi_risk,attr=self.eval_attr,train_attr=self.train_attr)
            sorted_wsi_st, sorted_inter_output, sorted_wsi_status, sorted_inter_output_fts = utils.sort_survival_time(self.eval_wsi_st, inter_output_s2, self.eval_wsi_status, inter_output_fts)
            inter_loss = self.compute_loss(sorted_inter_output, sorted_wsi_st, sorted_wsi_status, self.inter_model, sorted_inter_output_fts)

            intra_loss = np.mean(self.eval_intra_loss_value)
            intra_c_index = utils.accuracytest(self.eval_gt_value, -self.eval_output_value, self.eval_status_value)
            inter_c_index_s2 = utils.accuracytest(self.eval_wsi_st,-inter_output_s2,self.eval_wsi_status)
            inter_c_index_s1s2 = utils.accuracytest(self.eval_wsi_st,-inter_output_s1s2,self.eval_wsi_status)

        if intra_c_index > self.intra_best_i_fold_c_index:
            self.intra_best_i_fold_c_index = intra_c_index
            self.intra_best_i_fold_c_index_epoch = epoch + 1
        if intra_c_index > self.intra_best_c_index:
            self.intra_best_c_index = intra_c_index
            self.intra_best_epoch = epoch+1
            self.intra_best_i_fold = i_fold
        if inter_c_index_s2 > self.inter_best_i_fold_c_index_s2:
            self.inter_best_i_fold_c_index_s2 = inter_c_index_s2
            self.inter_best_i_fold_c_index_s2_epoch = epoch + 1
        if inter_c_index_s1s2 > self.inter_best_i_fold_c_index:
            self.inter_best_i_fold_c_index = inter_c_index_s1s2
            self.inter_best_i_fold_c_index_epoch = epoch + 1
            save_model=True
        if inter_c_index_s1s2 > self.inter_best_c_index:
            self.inter_best_c_index = inter_c_index_s1s2
            self.inter_best_epoch = epoch+1
            self.inter_best_i_fold = i_fold
        if save_model:
            state_dict = self.inter_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, 'inter_'+str(i_fold)+'_fold_best_model.pt')) #inter_output_s1s2.cpu().numpy()
            result_dict = {'id': self.eval_wsi_id,'eval_risk': inter_output_s1s2.cpu().detach().numpy(),'eval_survival_time': self.eval_wsi_st.cpu().numpy(),'eval_status':self.eval_wsi_status.cpu().numpy()}
            if i_fold==1:
                print()
            with open(os.path.join(self.arg.work_dir, 'inter_'+str(i_fold)+'_fold_best_model.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)


        self.print_log('\tMean intra_val loss: {:.4f}. current epoch intra_c-index: {:.2f}%. best intra_c-index: {:.2f}%.'.format(intra_loss, np.mean(intra_c_index) * 100, np.mean(self.intra_best_i_fold_c_index) * 100))
        self.print_log('\tMean inter_val loss: {:.4f}. current epoch stage 2 inter_c-index: {:.2f}%. current epoch stage 1-2 inter_c-index: {:.2f}%. best s2_c-index: {:.2f}% best c-index: {:.2f}%.'.format(inter_loss, np.mean(inter_c_index_s2) * 100, np.mean(inter_c_index_s1s2) * 100,np.mean(self.inter_best_i_fold_c_index_s2) * 100, np.mean(self.inter_best_i_fold_c_index) * 100))
        return np.mean(intra_c_index) * 100, np.mean(inter_c_index_s1s2) * 100
    
    def test_best_model(self,i_fold,epoch,save_model=False):
        inter_weights_path = os.path.join(self.arg.work_dir, 'inter_'+str(i_fold)+'_fold_best_model.pt')
        inter_weights = torch.load(inter_weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                inter_weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in inter_weights.items()])
        self.inter_model.load_state_dict(inter_weights)
        self.arg.print_log = False
        c_index = self.eval(epoch=0, i_fold=i_fold)
        self.arg.print_log = True
        return c_index

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.inter_model)}')
            self.print_log(f'# Parameters: {count_parameters(self.intra_model)}')

            n_fold_val_best_c_index=[]
            for i in range(len(self.data_loader['train'])):
                if i > -1:
                    self.load_model(i)
                    self.load_optimizer()
                    self.inter_model = self.inter_model.cuda(self.output_device)
                    self.intra_model = self.intra_model.cuda(self.output_device)
                    self.inter_best_i_fold_c_index = 0
                    self.inter_best_i_fold_c_index_epoch = 0
                    self.intra_best_i_fold_c_index = 0
                    self.intra_best_i_fold_c_index_epoch = 0
                    self.inter_best_i_fold_c_index_s2 = 0
                    self.inter_best_i_fold_c_index_s2_epoch = 0

                    self.train_intra_loss_value = []
                    self.train_output_value = None
                    self.train_gt_value = None
                    self.train_status_value = None
                    self.train_wsi_fts = None
                    self.train_wsi_risk = None
                    self.train_wsi_st = None
                    self.train_wsi_status = None
                    self.train_attr = None

                    self.eval_intra_loss_value = []
                    self.eval_output_value = None
                    self.eval_gt_value = None
                    self.eval_status_value = None
                    self.eval_wsi_fts = None
                    self.eval_wsi_risk = None
                    self.eval_wsi_st = None
                    self.eval_wsi_status = None
                    self.eval_wsi_id = None
                    self.eval_attr = None
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                            epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                    self.train(epoch, i_fold=i)
                    self.inter_scheduler.step()
                    if  (epoch + 1) % self.arg.eval_interval == 0:
                        self.eval(epoch, i_fold=i)
                
                # test the best model
                self.print_log(f'Best model with best intra_c-index')
                intra_best_c_index = self.test_best_model(i,self.intra_best_i_fold_c_index_epoch)
                self.print_log(f'Best model with best inter_c-index')
                inter_best_c_index = self.test_best_model(i,self.inter_best_i_fold_c_index_epoch)
                n_fold_val_best_c_index.append([intra_best_c_index,inter_best_c_index])

            # for i in range(len(n_fold_val_best_c_index)):
            #     self.print_log('n_fold: {}, intra get best. best intra_c-index: {}, best inter_c-index: {}'.format(i,n_fold_val_best_c_index[i][0][0],n_fold_val_best_c_index[i][0][1]))
            # self.print_log('{}_fold, intra get best best mean intra_c-index: {}, best mean inter_c-index: {}'.format(self.arg.n_fold, np.mean([n_fold_val_best_c_index[i][0][0] for i in range(len(n_fold_val_best_c_index))]),
            #                                                                                            np.mean([n_fold_val_best_c_index[i][0][1] for i in range(len(n_fold_val_best_c_index))])))
            for i in range(len(n_fold_val_best_c_index)):
                self.print_log('n_fold: {}, inter get best. best intra_c-index: {}, best inter_c-index: {}'.format(i,n_fold_val_best_c_index[i][1][0],n_fold_val_best_c_index[i][1][1]))
            self.print_log('{}_fold, inter get best best mean intra_c-index: {}, best mean inter_c-index: {}, best std inter_c-index: {}'.format(self.arg.n_fold, np.mean([n_fold_val_best_c_index[i][1][0] for i in range(len(n_fold_val_best_c_index))]),
                                                                                                       np.mean([n_fold_val_best_c_index[i][1][1] / 100 for i in range(len(n_fold_val_best_c_index))]),
                                                                                                       np.std([n_fold_val_best_c_index[i][1][1] / 100 for i in range(len(n_fold_val_best_c_index))])))

            self.print_log(f'Best inter_c-index: {self.inter_best_c_index}')
            self.print_log(f'Best inter_i_fold: {self.inter_best_i_fold}')
            self.print_log(f'Epoch number of inter: {self.inter_best_epoch}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.inter_weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            n_fold_val_best_c_index = []
            for i in range(len(self.data_loader['val'])):
                self.load_model(i=i)
                self.intra_model.to(self.arg.device[0])
                self.inter_model.to(self.arg.device[0])

                self.train_intra_loss_value = []
                self.train_output_value = None
                self.train_gt_value = None
                self.train_status_value = None
                self.train_wsi_fts = None
                self.train_wsi_risk = None
                self.train_wsi_st = None
                self.train_wsi_status = None
                self.train_attr = None

                self.eval_intra_loss_value = []
                self.eval_output_value = None
                self.eval_gt_value = None
                self.eval_status_value = None
                self.eval_wsi_fts = None
                self.eval_wsi_risk = None
                self.eval_wsi_st = None
                self.eval_wsi_status = None
                self.eval_wsi_id = None
                self.eval_attr = None

                self.train(epoch=0, i_fold=i)
                intra_cindex, inter_cindex = self.eval(epoch=0, i_fold=i)
                n_fold_val_best_c_index.append([intra_cindex, inter_cindex])
            for i in range(len(n_fold_val_best_c_index)):
                self.print_log('n_fold: {}, best intra_c-index: {}, best inter_c-index: {}'.format(i,n_fold_val_best_c_index[i][0], n_fold_val_best_c_index[i][1],))
            self.print_log('{}_fold, inter get best best mean intra_c-index: {}, best mean inter_c-index: {}, best std inter_c-index: {}'.format(self.arg.n_fold, np.mean([n_fold_val_best_c_index[i][0] for i in range(len(n_fold_val_best_c_index))]),
                                                                                            np.mean([n_fold_val_best_c_index[i][1] / 100 for i in range(len(n_fold_val_best_c_index))]),
                                                                                            np.std([n_fold_val_best_c_index[i][1] / 100 for i in range(len(n_fold_val_best_c_index))])))

            self.print_log('Done.\n')
        
if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

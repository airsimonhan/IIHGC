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
from loss import * 
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
    parser.add_argument('--config', default='config/gsz/wsi.yaml', help='path to the configuration file')
    parser.add_argument('--work_dir',default='./work_dir/',help='the work folder for storing results')
    parser.add_argument('--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--print_log',default=True, help='print logging or not')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')

    # data_loader
    parser.add_argument('--n_fold', type=int,default=5, help='the num of fold for cross validation')
    parser.add_argument('--start_fold', type=int,default=0, help='the start fold for cross validation')
    parser.add_argument('--dataset', default='dataset.WSI_Dataset.SlidePatch', help='data set will be used')
    parser.add_argument('--data_seed',type=int, default=1, help='random seed for n_fold dataset')
    parser.add_argument('--WSI_info_list_file', help='path to the information list of WSI sample')
    parser.add_argument('--WSI_patch_ft_dir', help='path to the feature of WSI patch')
    parser.add_argument('--WSI_patch_coor_dir', help='path to the coordinate of WSI patch')
    parser.add_argument('--center',type=str, default=['GY'],nargs='+', help='the center of data')

    parser.add_argument('--num_worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')

    # model
    parser.add_argument('--H_coors', default=False, help='if use the coors of patches to create H')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--step', type=int, default=100,help='the epoch where optimizer reduce the learning rate') #, nargs='+'
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

        self.lr = self.arg.base_lr
        self.best_i_fold_c_index = 0
        self.best_i_fold_c_index_epoch = 0
        self.best_c_index = 0
        self.best_i_fold = 0
        self.best_epoch = 0

        self.model = self.model.cuda(self.output_device)
        self.loss = import_class(self.arg.loss)()#coxph_loss()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
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
        WSI_info_list, self.survival_time_max, self.survival_time_min = utils.get_WSI_sample_list(self.arg.WSI_info_list_file, self.arg.center,self.arg.WSI_patch_ft_dir,self.arg.WSI_patch_coor_dir)# , multi_label=True
        n_fold_train_list, n_fold_val_list = utils.get_n_fold_data_list(WSI_info_list,self.arg.n_fold,self.arg.data_seed)

        self.data_loader['train'] = []
        self.data_loader['val'] = []
        for i in range(len(n_fold_train_list)):
            self.data_loader['train'].append(torch.utils.data.DataLoader(
                dataset=dataset(n_fold_train_list[i], self.survival_time_max, self.survival_time_min),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed))
            
            self.data_loader['val'].append(torch.utils.data.DataLoader(
                dataset=dataset(n_fold_val_list[i], self.survival_time_max, self.survival_time_min),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed))

    def load_model(self,i=0):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        if isinstance(self.arg.model_args, str):
            self.arg.model_args = json.loads(self.arg.model_args)
        self.model = Model(**self.arg.model_args)
        print(self.model)

        if self.arg.weights:
            weights = os.path.join(self.arg.weights, str(i)+'_fold_best_model.pt')
            self.print_log('Load weights from {}.'.format(weights))
            if '.pkl' in weights:
                with open(weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.arg.step, gamma=self.arg.lr_decay_rate)


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
    def compute_loss(self, sorted_output, sorted_status):
        loss = (self.loss(sorted_output, sorted_status)).sum() #coxph_loss
        return loss
    def train(self, epoch, i_fold, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {} , n_fold: {}'.format(epoch + 1, i_fold))
        loader = self.data_loader['train'][i_fold]

        loss_value = []
        output_value = None
        gt_value = None
        status_value = None
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (features, survival_time, status, coors, id) in enumerate(process):
            if status.sum()==0:
                continue
            with torch.no_grad():
                features = features.float().cuda(self.output_device)
                survival_time = survival_time.float().cuda(self.output_device)
                coors = coors.float().cuda(self.output_device)
                status = status.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            if self.arg.H_coors:
                output, output_fts = self.model(features,coors,train=True)
            else:
                output, output_fts = self.model(features,train=True)  # ,wloss ,label
            sorted_gt,sorted_output,sorted_status,sorted_output_fts = utils.sort_survival_time(survival_time,output,status,output_fts)
            loss = self.compute_loss(sorted_output, sorted_status) 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()
            output_value = self.concat(output_value,sorted_output)
            status_value = self.concat(status_value,sorted_status)
            gt_value = self.concat(gt_value,sorted_gt)
            
            

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        c_index = utils.accuracytest(gt_value,-output_value,status_value)
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean c-index: {:.2f}%. lr: {:.8f}'.format(np.mean(loss_value), np.mean(c_index) * 100,self.lr))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold)+'-runs-' + str(epoch+1) + '.pt'))
        return gt_value,output_value,status_value

    def eval(self, epoch, i_fold,train_gt_value=None,train_output_value=None,train_status_value=None,save_model=False,save_score=False):
        self.model.eval()
        self.print_log('Eval epoch: {},  n_fold: {}'.format(epoch + 1, i_fold))
        loss_value = []
        output_value = None
        output_feature = None
        gt_value = None
        status_value = None
        all_id = None
        step = 0
        process = tqdm(self.data_loader['val'][i_fold], ncols=40)
        for batch_idx, (features, survival_time, status, coors, id) in enumerate(process):
            with torch.no_grad():
                features = features.float().cuda(self.output_device)
                survival_time = survival_time.float().cuda(self.output_device)
                coors = coors.float().cuda(self.output_device)
                status = status.long().cuda(self.output_device)
                if all_id is None:
                    all_id =id
                else:
                    all_id = all_id + id
                if isinstance(self.model, torch.nn.DataParallel):
                    if self.arg.H_coors:
                        output, output_fts = self.model.module.forward(features,coors)  
                    else:
                        output, output_fts = self.model.module.forward(features)
                else:
                    if self.arg.H_coors:
                        output, output_fts = self.model.forward(features,coors)
                    else:
                        output, output_fts = self.model.forward(features)  

                sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(survival_time, output, status, output_fts)

                loss = self.compute_loss(sorted_output, sorted_status) 

                if status.sum() == 0:
                    loss[loss != loss] = 0  # turn nan to 0
                loss_value.append(loss.data.item())
                output_value = self.concat(output_value, output)
                output_feature = self.concat(output_feature, output_fts)
                status_value = self.concat(status_value, status)
                gt_value = self.concat(gt_value, survival_time)
                step += 1
        with torch.no_grad():
            loss = np.mean(loss_value)
            c_index = utils.accuracytest(gt_value, -output_value, status_value)
        if c_index > self.best_i_fold_c_index:
            self.best_i_fold_c_index = c_index
            self.best_i_fold_c_index_epoch = epoch + 1
            save_model=True
            save_score=True
        if c_index > self.best_c_index:
            self.best_c_index = c_index
            self.best_epoch = epoch+1
            self.best_i_fold = i_fold
        
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold)+'_fold_best_model.pt'))
        result_dict = None
        if save_score:
            result_dict = {'id':all_id,'eval_risk': output_value.cpu().numpy(),'eval_feature': output_feature.cpu().numpy(),'eval_survival_time': gt_value.cpu().numpy(),'eval_status':status_value.cpu().numpy()}
            with open(os.path.join(self.arg.work_dir, str(i_fold)+'_fold_best_model.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)
        self.print_log('\tMean val loss: {:.4f}. current epoch c-index: {:.2f}%. best c-index: {:.2f}%.'.format(loss, np.mean(c_index) * 100, np.mean(self.best_i_fold_c_index) * 100))
        return np.mean(c_index) * 100, result_dict

    def test_best_model(self,i_fold,epoch,save_model=False):
        weights_path = os.path.join(self.arg.work_dir, str(i_fold)+'_fold_best_model.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model.load_state_dict(weights)
        self.arg.print_log = False
        c_index,result_dict = self.eval(epoch=0, i_fold=i_fold,save_score=True)
        self.arg.print_log = True
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold)+'_fold_best_model.pt'))
        if result_dict is not None:
            with open(os.path.join(self.arg.work_dir, str(i_fold)+'_fold_best_model.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)
        return c_index

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            n_fold_val_best_c_index=[]
            for i in range(len(self.data_loader['train'])):
                if i<self.arg.start_fold:
                    continue

                if i > 0:
                    self.load_model()
                    self.load_optimizer()
                    self.model = self.model.cuda(self.output_device)
                    self.best_i_fold_c_index = 0
                    self.best_i_fold_c_index_epoch = 0
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                            epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                    train_gt_value,train_output_value,train_status_value = self.train(epoch, i_fold=i, save_model=False)
                    self.scheduler.step()

                    self.eval(epoch,i,train_gt_value,train_output_value,train_status_value)

                # test the best model
                c_index = self.test_best_model(i,self.best_i_fold_c_index_epoch,save_model=True)
                n_fold_val_best_c_index.append(c_index/100)

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            for i in range(len(n_fold_val_best_c_index)):
                self.print_log('n_fold: {}, best c-index: {}'.format(i,n_fold_val_best_c_index[i]))
            self.print_log('{}_fold, best mean c-index: {}. std c-index: {}.'.format(self.arg.n_fold, np.mean(n_fold_val_best_c_index), np.std(n_fold_val_best_c_index)))
            self.print_log(f'Best c-index: {self.best_c_index}')
            self.print_log(f'Best i_fold: {self.best_i_fold}')
            self.print_log(f'Epoch number: {self.best_epoch}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            n_fold_val_best_c_index = []
            for i in range(len(self.data_loader['val'])):
                self.load_model(i=i)
                self.model.to(self.arg.device[0])
                cindex, result_dict = self.eval(epoch=0, i_fold=i,save_score=True)
                n_fold_val_best_c_index.append(cindex)
                if result_dict is not None:
                    with open(os.path.join(self.arg.work_dir, str(i)+'_fold_best_model.pkl'), 'wb') as f:
                        pickle.dump(result_dict, f)

            for i in range(len(n_fold_val_best_c_index)):
                self.print_log('n_fold: {}, best c-index: {}'.format(i,n_fold_val_best_c_index[i]))
            self.print_log('{}_fold, best mean c-index: {}'.format(self.arg.n_fold, np.mean(n_fold_val_best_c_index)))
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

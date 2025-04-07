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
from loss import coxph_loss
import json
from dataset.WSI_Dataset import SlidePatch
from model.intra_stage import intraModel


class Processor:
    """
    Processor for Skeleton-based Action Recognition
    """

    def __init__(self, arg):
        self.arg = arg
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        self.data_loader = {'train': [], 'val': []}

        self._initialize_variables()
        self._setup_work_dir()
        self._save_configuration()
        self._load_data()
        self._initialize_model(i=0)
        self._initialize_optimizer()
        self.best_i_fold_c_index = 0
        self.best_i_fold_c_index_epoch = 0
        self.best_c_index = 0
        self.best_i_fold = 0
        self.best_epoch = 0

    def _initialize_variables(self):
        """Initialize internal variables."""
        self.lr = self.arg.base_lr
        self.best_metrics = {
            'fold_c_index': 0,
            'fold_c_index_epoch': 0,
            'overall_c_index': 0,
            'best_fold': 0,
            'best_epoch': 0
        }

    def _setup_work_dir(self):
        """Create the working directory if it doesn't exist."""
        os.makedirs(self.arg.work_dir, exist_ok=True)

    def _save_configuration(self):
        """Save the configuration to a YAML file."""
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            f.write(f"# Command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(vars(self.arg), f)

    def _load_data(self):
        """Load training and validation data."""
        WSI_info_list, max_survival_time, min_survival_time = utils.get_WSI_sample_list(
            self.arg.WSI_info_list_file, self.arg.center, self.arg.WSI_patch_ft_dir, self.arg.WSI_patch_coor_dir
        )
        n_fold_train_list, n_fold_val_list = utils.get_n_fold_data_list(
            WSI_info_list, self.arg.n_fold, self.arg.data_seed
        )

        for train_list, val_list in zip(n_fold_train_list, n_fold_val_list):
            self.data_loader['train'].append(torch.utils.data.DataLoader(
                dataset=SlidePatch(train_list, max_survival_time, min_survival_time),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed
            ))

            self.data_loader['val'].append(torch.utils.data.DataLoader(
                dataset=SlidePatch(val_list, max_survival_time, min_survival_time),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed
            ))

    def _initialize_model(self, i):
        """Load and initialize the model."""
        self.model = intraModel(
            in_channels=self.arg.in_channels,
            n_target=self.arg.n_target,
            k_nearest=self.arg.k_nearest,
            k_threshold=self.arg.k_threshold,
            hiddens=self.arg.hiddens,
            dropout=self.arg.dropout,
            drop_max_ratio=self.arg.drop_max_ratio,)

        shutil.copy2(inspect.getfile(self.model.__class__), self.arg.work_dir)

        if type(self.arg.device) is list:
            self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

        self._load_weights(i=i)
        self.loss = coxph_loss()

    def _load_weights(self, i=0):
        """Load pre-trained weights if specified."""
        if self.arg.weights:
            weights_path = os.path.join(self.arg.weights, str(i) + '_fold_best_model.pt')
            print(f"\nLoading weights from {weights_path}")

            weights = torch.load(weights_path, weights_only=True)
            weights = OrderedDict(
                [['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()]
            )

            self.model.load_state_dict(weights, strict=False)

    def _initialize_optimizer(self):
        """Initialize the optimizer and learning rate scheduler."""
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError("Unsupported optimizer type.")

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.arg.step, gamma=self.arg.lr_decay_rate
        )

    def train(self, epoch, i_fold, save_model=False):
        """
        Training loop for a single epoch.

        Parameters:
            epoch: Current training epoch.
            i_fold: Index of the fold (for cross-validation).
            save_model: Whether to save the model after training.

        Returns:
            gt_value, output_value, status_value: Ground truth, outputs, and status values for the epoch.
        """
        print('\n\033[92mFold {} - Epoch {:04d}\033[0m'.format(i_fold, epoch+1))

        self.model.train()

        loss_values = []
        output_values = None
        gt_values = None
        status_values = None
        self._record_time()

        timer = {"dataloader": 0.001, "model": 0.001, "statistics": 0.001}
        process = tqdm(self.data_loader['train'][i_fold], desc='[Train]')

        for batch_idx, (features, survival_time, status, coors, _) in enumerate(process):
            if status.sum() == 0:
                continue  # Skip batches with no valid status

            # Move data to the specified device
            features = features.float().to(self.output_device)
            survival_time = survival_time.float().to(self.output_device)
            coors = coors.float().to(self.output_device)
            status = status.long().to(self.output_device)
            timer['dataloader'] += self._split_time()

            # Forward pass
            if self.arg.H_coors:
                output, output_fts = self.model(features, coors, train=True)
            else:
                output, output_fts = self.model(features, train=True)

            sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(
                survival_time, output, status, output_fts
            )

            # Compute loss
            loss = self.loss(sorted_output, sorted_status).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            loss_values.append(loss.item())
            timer['model'] += self._split_time()
            output_values = torch.cat((output_values, sorted_output),
                                      dim=0) if output_values is not None else sorted_output
            gt_values = torch.cat((gt_values, sorted_gt), dim=0) if gt_values is not None else sorted_gt
            status_values = torch.cat((status_values, sorted_status),
                                      dim=0) if status_values is not None else sorted_status

            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self._split_time()

        # Calculate and log statistics
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        c_index = utils.accuracytest(gt_values, -output_values, status_values)
        self.print_log(
            'Mean training loss: {:.4f}.  Mean c-index: {:.2f}%. lr: {:.8f}'.format(np.mean(loss_values),
                                                                                    np.mean(c_index) * 100, self.lr))
        self.print_log(
            'Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # Save the model if required
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict({k: v.cpu() for k, v in state_dict.items()})
            model_path = f"{self.work_dir}/{i_fold}-epoch-{epoch + 1}.pt"
            torch.save(weights, model_path)
            self.print_log(f"Model saved to {model_path}")

        return gt_values, output_values, status_values

    def print_log(self, message, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            message = f"[ {localtime} ] {message}"
        print(message)

    def _move_to_device(self, *args):
        """Move data to the specified device."""
        return [x.cuda(self.output_device) for x in args]

    def _record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def _split_time(self):
        split_time = time.time() - self.cur_time
        self._record_time()
        return split_time

    def eval(self, epoch, i_fold, save_model=False, save_score=False):
        """
        Evaluation loop for the specified fold.

        Parameters:
            epoch: Current evaluation epoch.
            i_fold: Index of the fold being evaluated.
            train_gt_value, train_output_value, train_status_value: Training metrics for comparison.
            save_model: Whether to save the model if it achieves the best score.
            save_score: Whether to save evaluation scores.

        Returns:
            c_index: C-index of the evaluation.
            result_dict: Dictionary of evaluation results if save_score is True.
        """
        if self.arg.phase == 'test':
            print('\033[92mFold {} - Testing Phase'.format(i_fold))

        self.model.eval()

        loss_values = []
        output_values = None
        output_features = None
        gt_values = None
        status_values = None
        all_ids = None

        process = tqdm(self.data_loader['val'][i_fold], desc='[Valid]')
        for batch_idx, (features, survival_time, status, coors, ids) in enumerate(process):
            with torch.no_grad():
                # Move data to the device
                features = features.float().to(self.output_device)
                survival_time = survival_time.float().to(self.output_device)
                coors = coors.float().to(self.output_device)
                status = status.long().to(self.output_device)

                all_ids = ids if all_ids is None else all_ids + ids

                # Forward pass
                if isinstance(self.model, torch.nn.DataParallel):
                    output, output_fts = self.model(features, coors)
                else:
                    output, output_fts = self.model(features)

                sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(
                    survival_time, output, status, output_fts
                )

                loss = self.loss(sorted_output, sorted_status).sum()

                # Handle invalid loss
                if status.sum() == 0:
                    loss[loss != loss] = 0

                # Track metrics
                loss_values.append(loss.item())
                output_values = torch.cat((output_values, output), dim=0) if output_values is not None else output
                output_features = torch.cat((output_features, output_fts),
                                            dim=0) if output_features is not None else output_fts
                gt_values = torch.cat((gt_values, survival_time), dim=0) if gt_values is not None else survival_time
                status_values = torch.cat((status_values, status), dim=0) if status_values is not None else status

        with torch.no_grad():
            mean_loss = np.mean(loss_values)
            c_index = utils.accuracytest(gt_values, -output_values, status_values)

        # Save the best model
        if c_index > self.best_i_fold_c_index:
            self.best_i_fold_c_index = c_index
            self.best_i_fold_c_index_epoch = epoch + 1
            save_model = True
            save_score = True

        if c_index > self.best_c_index:
            self.best_c_index = c_index
            self.best_epoch = epoch + 1
            self.best_i_fold = i_fold

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict({k: v.cpu() for k, v in state_dict.items()})
            model_path = os.path.join(self.arg.work_dir, f'{i_fold}_fold_best_model.pt')
            torch.save(weights, model_path)
            self.print_log(f"Model saved to {model_path}")

        result_dict = None
        if save_score:
            result_dict = {
                'id': all_ids,
                'eval_risk': output_values.cpu().numpy(),
                'eval_feature': output_features.cpu().numpy(),
                'eval_survival_time': gt_values.cpu().numpy(),
                'eval_status': status_values.cpu().numpy()
            }
            result_path = os.path.join(self.arg.work_dir, f'{i_fold}_fold_best_model.pkl')
            with open(result_path, 'wb') as f:
                pickle.dump(result_dict, f)
            self.print_log(f"Results saved to {result_path}")

        self.print_log(
            f"Mean val loss: {mean_loss:.4f}. Current epoch c-index: {c_index * 100:.2f}%. Best c-index: {self.best_i_fold_c_index * 100:.2f}%.")

        return c_index * 100, result_dict

    def start(self):
        if self.arg.phase == 'train':
            self._log_parameters()
            n_fold_val_best_c_index = self._train_all_folds()

            self._log_final_results(n_fold_val_best_c_index)

        elif self.arg.phase == 'test':
            self._test_all_folds()

    def _log_parameters(self):
        self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.print_log(f'# Parameters: {count_parameters(self.model)}')

    def _train_all_folds(self):
        n_fold_val_best_c_index = []

        for i in range(len(self.data_loader['train'])):
            if i < self.arg.start_fold:
                continue

            if i > 0:
                self._reset_model_and_optimizer()
                self.best_i_fold_c_index = 0
                self.best_i_fold_c_index_epoch = 0

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = self._should_save_model(epoch)

                train_gt_value, train_output_value, train_status_value = self.train(epoch, i_fold=i, save_model=False)
                self.scheduler.step()

                self.eval(epoch, i)

            # Test the best model
            c_index = self._test_best_model(i, self.best_epoch, save_model=True)
            n_fold_val_best_c_index.append(c_index / 100)

        return n_fold_val_best_c_index

    def _test_best_model(self, i_fold, epoch, save_model=False):
        weights_path = os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pt')
        weights = torch.load(weights_path)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
        self.model.load_state_dict(weights)
        self.arg.print_log = False
        c_index, result_dict = self.eval(epoch=0, i_fold=i_fold, save_score=True)
        self.arg.print_log = True
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pt'))
        if result_dict is not None:
            with open(os.path.join(self.arg.work_dir, str(i_fold) + '_fold_best_model.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)
        return c_index

    def _log_final_results(self, n_fold_val_best_c_index):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        for i, c_index in enumerate(n_fold_val_best_c_index):
            self.print_log(f'n_fold: {i}, best c-index: {c_index}')

        mean_c_index = np.mean(n_fold_val_best_c_index)
        std_c_index = np.std(n_fold_val_best_c_index)

        self.print_log(f'{self.arg.n_fold}_fold, best mean c-index: {mean_c_index}. std c-index: {std_c_index}.')
        self.print_log(f'Best c-index: {self.best_c_index}')
        self.print_log(f'Best i_fold: {self.best_i_fold}')
        self.print_log(f'Epoch number: {self.best_epoch}')
        self.print_log(f'Model total number of params: {num_params}')
        self.print_log(f'Weight decay: {self.arg.weight_decay}')
        self.print_log(f'Base LR: {self.arg.base_lr}')
        self.print_log(f'Batch Size: {self.arg.batch_size}')
        self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
        self.print_log(f'seed: {self.arg.seed}')

    def _test_all_folds(self):
        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')

        self.print_log(f'Model: {self.arg.model}.')
        self.print_log(f'Weights: {self.arg.weights}.')

        n_fold_val_best_c_index = []

        for i in range(len(self.data_loader['val'])):
            self._initialize_model(i=i)
            self.model.to(self.output_device)

            c_index, result_dict = self.eval(epoch=0, i_fold=i, save_score=True)
            n_fold_val_best_c_index.append(c_index)

            if result_dict is not None:
                with open(os.path.join(self.arg.work_dir, f'{i}_fold_best_model.pkl'), 'wb') as f:
                    pickle.dump(result_dict, f)

        print("\n")
        for i, c_index in enumerate(n_fold_val_best_c_index):
            self.print_log(f'n_fold: {i}, best c-index: {c_index}')

        mean_c_index = np.mean(n_fold_val_best_c_index)
        self.print_log(f'{self.arg.n_fold}_fold, best mean c-index: {mean_c_index}.')
        self.print_log('Done.\n')

    def _reset_model_and_optimizer(self):
       self._initialize_model(i=0)
        self._initialize_optimizer()
        self.model = self.model.cuda(self.output_device)
        self.best_c_index = 0
        self.best_epoch = 0

    def _should_save_model(self, epoch):
        return (((epoch + 1) % self.arg.save_interval == 0) or (
            epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser(description='multi-modal survival prediction')
    parser.add_argument('--work_dir', default='work_dir/intra_stage/tcga_lusc/', help='results fold')
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--data_seed', type=int, default=1, help='random seed for n_fold dataset')
    parser.add_argument('--print_log', default=True, help='print logging or not')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model')

    # data_loader
    parser.add_argument('--center', type=str, default=['TCGA-LUSC'], nargs='+', help='the center of data')
    parser.add_argument('--n_fold', type=int, default=5, help='the num of fold for cross validation')
    parser.add_argument('--start_fold', type=int, default=0, help='the start fold for cross validation')
    parser.add_argument('--WSI_info_list_file', default='dataset/WSI_info_list/TCGA-LUSC.json', help='path to the information list of WSI sample')
    parser.add_argument('--WSI_patch_ft_dir', default='get_feature/WSI_patch_features/{}_2K_sample_threshold/patch_efficientnet_ft', help='path to the feature of WSI patch')
    parser.add_argument('--WSI_patch_coor_dir', default='get_feature/WSI_patch_features/{}_2K_sample_threshold/patch_coor', help='path to the coordinate of WSI patch')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--num_worker', type=int, default=1, help='the number of worker for data loader')

    # model
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--H_coors', default=True, help='if use the coors of patches to create H')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--in_channels', default=1792, help='the arguments of model')
    parser.add_argument('--n_target', default=1, help='the arguments of model')
    parser.add_argument('--k_nearest', default=5, help='the arguments of model')
    parser.add_argument('--k_threshold', default=0.1, help='the arguments of model')
    parser.add_argument('--hiddens', default=[128, 128, 128], help='the arguments of model')
    parser.add_argument('--dropout', default=0.3, help='the arguments of model')
    parser.add_argument('--drop_max_ratio', default=0.05, help='the arguments of model')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='')

    # test phase
    parser.add_argument('--weights', default=None, help='initialization')

    # optim
    parser.add_argument('--device', type=int, default=[0], nargs='+', help='the indexes of GPUs')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=100, help='the epoch for reduce the learning rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=200, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss', type=str, default='loss.mse_loss', help='the type of loss function')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()

    init_seed(arg.seed)

    processor = Processor(arg)
    processor.start()

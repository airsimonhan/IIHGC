import os
import sys
import yaml
import json
import time
import glob
import pickle
import shutil
import random
import argparse
import inspect
import traceback
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import utils
from loss import coxph_loss
from dataset.WSI_Dataset import SlidePatch
from model.intra_stage import intraModel
from model.inter_stage import interModel


def init_seed(seed: int) -> None:
    """
    Initialize random seeds for reproducibility.

    Args:
        seed (int): Random seed to set.
    """
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser() -> argparse.ArgumentParser:
    """
    Create and return an argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Multi-modal survival prediction")

    # ---------------------------
    # General configuration
    # ---------------------------
    parser.add_argument('--work_dir', default='work_dir/inter_intra/tcga_lusc/', help='Working directory for storing results and logs.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help='Operation phase: train or test.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for PyTorch.')
    parser.add_argument('--print_log', type=bool, default=True, help='Whether to print logging information.')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval epochs for saving models.')
    parser.add_argument('--save_epoch', type=int, default=0, help='Start epoch index to begin saving models.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Interval epochs for evaluation.')

    # ---------------------------
    # Data-related arguments
    # ---------------------------
    parser.add_argument('--n_fold', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--dataset', default='dataset.WSI_Dataset.SlidePatch', help='Fully qualified name of the Dataset class to use.')
    parser.add_argument('--data_seed', type=int, default=1, help='Seed for data splitting, ensuring reproducibility.')
    parser.add_argument('--WSI_info_list_file', default='dataset/WSI_info_list/TCGA-LUSC.json', help='Path to the WSI information JSON file.')
    parser.add_argument('--WSI_patch_ft_dir', default='get_feature/WSI_patch_features/{}_2K_sample_threshold/patch_efficientnet_ft', help='Directory for WSI patch features.')
    parser.add_argument('--WSI_patch_coor_dir', default='get_feature/WSI_patch_features/{}_2K_sample_threshold/patch_coor', help='Directory for WSI patch coordinates.')
    parser.add_argument('--center', nargs='+', default=['TCGA-LUSC'], help='List of data centers.')
    parser.add_argument('--clinical', type=bool, default=True, help='Whether clinical data is used.')
    parser.add_argument('--num_worker', type=int, default=1,help='Number of data loader workers.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing.')

    # ---------------------------
    # Intra-Model arguments
    # ---------------------------
    parser.add_argument('--in_channels_', default=1792, help='Number of input channels for intra-model.')
    parser.add_argument('--n_target_', default=1, help='Number of output targets for intra-model.')
    parser.add_argument('--k_nearest_', default=5, help='K-nearest argument for intra-model.')
    parser.add_argument('--k_threshold_', default=0.1, help='K-threshold for intra-model.')
    parser.add_argument('--hiddens_', default=[128, 128, 128], help='Hidden dimensions for intra-model.')
    parser.add_argument('--dropout_', default=0.3, help='Dropout rate for intra-model.')
    parser.add_argument('--drop_max_ratio_', default=0.05, help='Max ratio of dropping patches in intra-model.')
    parser.add_argument('--intra_weights', default='work_dir/intra_stage/tcga_lusc/', help='Path to pre-trained intra-model weights.')
    parser.add_argument('--ignore_intra_weights', nargs='+', type=str, default=[], help='List of weight keywords to ignore in intra-model.')

    # ---------------------------
    # Inter-Model arguments
    # ---------------------------
    parser.add_argument('--H_coors', action='store_true', default=True, help='Whether patch coordinates are used for adjacency matrix H.')
    parser.add_argument('--in_channels', type=int, default=262, help='Number of input channels for inter-model.')
    parser.add_argument('--n_target', type=int, default=1, help='Number of output targets for inter-model.')
    parser.add_argument('--k_threshold', type=float, default=0.01, help='K-threshold for inter-model.')
    parser.add_argument('--hiddens', nargs='+', type=int, default=[128, 128, 128], help='Hidden layer dimensions for inter-model.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for inter-model.')
    parser.add_argument('--label_hiddens', nargs='+', type=int, default=[16, 16, 16], help='Hidden dimensions for label sub-network.')
    parser.add_argument('--label_in', type=int, default=1, help='Input dimension for label sub-network.')
    parser.add_argument('--scale', type=int, default=0.5, help='Scaling factor for label sub-network.')

    parser.add_argument('--inter_weights', default=None, help='Path to pre-trained weights for inter-model.')
    parser.add_argument('--ignore_inter_weights', nargs='+', type=str, default=[], help='List of weight keywords to ignore in inter-model.')

    # ---------------------------
    # Optimizer arguments
    # ---------------------------
    parser.add_argument('--device', nargs='+', type=int, default=[0], help='List of GPU devices for training/testing.')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='Optimizer type.')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='Base learning rate.')
    parser.add_argument('--step', nargs='+', type=int, default=100, help='Step size(s) for learning rate scheduler.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch index.')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay parameter.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay factor.')
    parser.add_argument('--loss', default='loss.coxph_loss', help='Loss function to use.')

    return parser


class Processor:
    """
    Processor for multi-modal survival prediction using both intra-stage and inter-stage models.
    Responsible for data loading, model initialization, training, and evaluation.
    """

    def __init__(self, arg: argparse.Namespace):
        """
        Initialize the Processor with provided arguments.

        Args:
            arg (argparse.Namespace): Parsed command-line arguments.
        """
        self.arg = arg
        self.output_device = (self.arg.device[0]
                              if isinstance(self.arg.device, list)
                              else self.arg.device)

        # Set up internal states
        self._initialize_internal_tracking()

        # Create working directory & save config
        self.save_arg()

        # Load dataset
        self.data_loader = {'train': [], 'val': []}
        self.load_data()

        # Initialize models, optimizer, and loss
        self._initialize_model()
        self._initialize_optimizer()
        self.loss = coxph_loss()

        # Move models to devices
        self.inter_model = self.inter_model.cuda(self.output_device)
        self.intra_model = self.intra_model.cuda(self.output_device)

        # Parallelize if multiple GPUs
        self._setup_dataparallel()

    def _initialize_internal_tracking(self) -> None:
        """
        Initialize internal variables used to track best metrics and current state.
        """
        self.intra_lr = self.arg.base_lr
        self.inter_lr = self.arg.base_lr

        # Best metrics trackers
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

        # Placeholder for data accumulations
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

        # Timer placeholders
        self.cur_time = time.time()

    def save_arg(self) -> None:
        """
        Save the command-line arguments to a configuration file (config.yaml) in the work directory.
        """
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        arg_dict = vars(self.arg)
        config_path = os.path.join(self.arg.work_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(f"# Command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_data(self) -> None:
        """
        Load datasets for training and validation according to the n-fold cross validation setup.
        """
        WSI_info_list, survival_time_max, survival_time_min = utils.get_WSI_sample_list(
            self.arg.WSI_info_list_file,
            self.arg.center,
            self.arg.WSI_patch_ft_dir,
            self.arg.WSI_patch_coor_dir,
            clinical=self.arg.clinical
        )

        n_fold_train_list, n_fold_val_list = utils.get_n_fold_data_list(
            WSI_info_list,
            self.arg.n_fold,
            self.arg.data_seed,
            clinical=self.arg.clinical
        )

        for i in range(len(n_fold_train_list)):
            train_loader = torch.utils.data.DataLoader(
                dataset=SlidePatch(
                    n_fold_train_list[i],
                    survival_time_max,
                    survival_time_min,
                    clinical=self.arg.clinical
                ),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed
            )

            val_loader = torch.utils.data.DataLoader(
                dataset=SlidePatch(
                    n_fold_val_list[i],
                    survival_time_max,
                    survival_time_min,
                    clinical=self.arg.clinical
                ),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed
            )

            self.data_loader['train'].append(train_loader)
            self.data_loader['val'].append(val_loader)

    def _initialize_model(self, i: int = 0) -> None:
        """
        Instantiate and initialize both the intra-stage and inter-stage models.

        Args:
            i (int, optional): Fold index used to load pre-trained weights. Defaults to 0.
        """
        # Intra-model instantiation
        self.intra_model = intraModel(
            in_channels=self.arg.in_channels_,
            n_target=self.arg.n_target_,
            k_nearest=self.arg.k_nearest_,
            k_threshold=self.arg.k_threshold_,
            hiddens=self.arg.hiddens_,
            dropout=self.arg.dropout_,
            drop_max_ratio=self.arg.drop_max_ratio_,
        )
        shutil.copy2(inspect.getfile(self.intra_model.__class__), self.arg.work_dir)
        # print(self.intra_model)

        # Inter-model instantiation
        self.inter_model = interModel(
            in_channels=self.arg.in_channels,
            label_in=self.arg.label_in,
            label_hiddens=self.arg.label_hiddens,
            n_target=self.arg.n_target,
            k_threshold=self.arg.k_threshold,
            hiddens=self.arg.hiddens,
            dropout=self.arg.dropout,
            scale=self.arg.scale
        )
        shutil.copy2(inspect.getfile(self.inter_model.__class__), self.arg.work_dir)
        # print(self.inter_model)

        # Load weights
        self._load_weights(i)

    def _load_weights(self, i: int = 0) -> None:
        """
        Load pre-trained weights for both the inter-model and intra-model, if specified.

        Args:
            i (int, optional): Fold index used to locate the correct weight file. Defaults to 0.
        """
        # ---------------------------
        # Load inter-model weights
        # ---------------------------
        if self.arg.inter_weights:
            self.print_log('Load inter-model weights from {}.'.format(self.arg.inter_weights))
            if '.pkl' in self.arg.inter_weights:
                with open(self.arg.inter_weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                inter_path = os.path.join(self.arg.inter_weights, f'inter_{i}_fold_best_model.pt')
                weights = torch.load(inter_path, weights_only=True)

            # Remove "module." prefix if present & ignore specified weights
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)]
                                   for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_inter_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Removed inter-model weight: {}.'.format(key))
                        else:
                            self.print_log('Could not remove inter-model weight: {}.'.format(key))

            # Attempt to load weights
            try:
                self.inter_model.load_state_dict(weights)
            except RuntimeError:
                state = self.inter_model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Could not find these inter-model weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.inter_model.load_state_dict(state)

        # ---------------------------
        # Load intra-model weights
        # ---------------------------
        if self.arg.intra_weights:
            intra_path = os.path.join(self.arg.intra_weights, f'{i}_fold_best_model.pt')
            self.print_log('Load intra-model weights from {}.'.format(intra_path))

            if '.pkl' in intra_path:
                with open(intra_path, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(intra_path, weights_only=True)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)]
                                   for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_intra_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Removed intra-model weight: {}.'.format(key))
                        else:
                            self.print_log('Could not remove intra-model weight: {}.'.format(key))

            # Attempt to load weights
            try:
                self.intra_model.load_state_dict(weights)
            except RuntimeError:
                state = self.intra_model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Could not find these intra-model weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.intra_model.load_state_dict(state)

    def _initialize_optimizer(self) -> None:
        """
        Initialize the optimizer and learning rate scheduler for the inter-model.
        (Currently only the inter-model is being optimized in the provided code.)
        """
        if self.arg.optimizer == 'SGD':
            self.inter_optimizer = optim.SGD(
                self.inter_model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            self.inter_optimizer = optim.Adam(
                self.inter_model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError("Unsupported optimizer type, only 'SGD' or 'Adam' are allowed.")

        self.inter_scheduler = lr_scheduler.StepLR(
            self.inter_optimizer,
            step_size=self.arg.step,
            gamma=self.arg.lr_decay_rate
        )

    def _setup_dataparallel(self) -> None:
        """
        Wrap the models with nn.DataParallel if multiple GPU devices are available.
        """
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.inter_model = nn.DataParallel(
                self.inter_model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )
            self.intra_model = nn.DataParallel(
                self.intra_model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )

    def print_time(self) -> None:
        """
        Print the current local time in a formatted manner.
        """
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time : " + localtime)

    def print_log(self, log_str: str, print_time_flag: bool = True) -> None:
        """
        Print and optionally log the provided message.

        Args:
            log_str (str): The message to print/log.
            print_time_flag (bool): Whether to include the timestamp in the message.
        """
        if print_time_flag:
            localtime = time.asctime(time.localtime(time.time()))
            log_str = "[ " + localtime + " ] " + log_str
        print(log_str)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(log_str, file=f)

    def record_time(self) -> float:
        """
        Record the current time (in seconds).

        Returns:
            float: The current timestamp.
        """
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self) -> float:
        """
        Calculate the time elapsed since the last recorded time,
        then record a new time checkpoint.

        Returns:
            float: Elapsed time in seconds.
        """
        elapsed = time.time() - self.cur_time
        self.record_time()
        return elapsed

    def concat(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        """
        Concatenate two tensors along the first dimension. If the first one is None, return the second.

        Args:
            tensor_a (torch.Tensor): The first tensor (could be None).
            tensor_b (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: Concatenated tensor or the second tensor if the first is None.
        """
        if tensor_a is None:
            return tensor_b
        else:
            return torch.concat((tensor_a, tensor_b), dim=0)

    def compute_loss(self,
                     sorted_output: torch.Tensor,
                     sorted_gt: torch.Tensor,
                     sorted_status: torch.Tensor,
                     model: nn.Module,
                     features: torch.Tensor) -> torch.Tensor:
        """
        Compute the CoxPH loss for the sorted outputs.

        Args:
            sorted_output (torch.Tensor): Model output sorted by survival time.
            sorted_gt (torch.Tensor): Ground truth survival times, sorted.
            sorted_status (torch.Tensor): Censoring status, sorted.
            model (nn.Module): The model (unused in this function, but kept for extensibility).
            features (torch.Tensor): Additional features (unused here, but kept for extensibility).

        Returns:
            torch.Tensor: Loss value.
        """
        loss_value = self.loss(sorted_output, sorted_status).sum()
        return loss_value

    def train(self,
              epoch: int,
              i_fold: int,
              save_model: bool = False,
              train: bool = True) -> None:
        """
        Perform one epoch of training for the inter-model with data from the intra-model.

        Args:
            epoch (int): Current epoch index.
            i_fold (int): Current fold index (for cross-validation).
            save_model (bool): Whether to save the model after this epoch.
            train (bool): Whether to perform backpropagation (True) or skip it (False).
        """
        print('\n\033[92mFold {} - Epoch {:04d}\033[0m'.format(i_fold, epoch + 1))

        self.intra_model.eval()
        self.inter_model.train()

        loader = self.data_loader['train'][i_fold]

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process_bar = tqdm(loader, desc='[Train]')

        # ---------------------------
        # 1) Forward pass of intra-model to get features (once)
        # ---------------------------
        # Only do the full pass if we haven't computed the training set representation yet
        if self.train_gt_value is None:
            for _, (features, survival_time, status, coors, attr, _) in enumerate(process_bar):
                with torch.no_grad():
                    features = features.float().cuda(self.output_device)
                    survival_time = survival_time.float().cuda(self.output_device)
                    coors = coors.float().cuda(self.output_device)
                    status = status.long().cuda(self.output_device)
                    attr = attr.long().cuda(self.output_device)
                timer['dataloader'] += self.split_time()

                # Intra-model forward
                if self.arg.H_coors:
                    output, output_fts = self.intra_model(features, coors)
                else:
                    output, output_fts = self.intra_model(features)

                # Accumulate
                self.train_wsi_fts = self.concat(self.train_wsi_fts, output_fts.detach())
                self.train_wsi_risk = self.concat(self.train_wsi_risk, output.detach())
                self.train_wsi_st = self.concat(self.train_wsi_st, survival_time)
                self.train_wsi_status = self.concat(self.train_wsi_status, status)
                self.train_attr = self.concat(self.train_attr, attr)

                # Compute the intra-model loss just for logging and gradient check
                sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(
                    survival_time, output, status, output_fts
                )
                intra_loss = self.compute_loss(sorted_output,
                                               sorted_gt,
                                               sorted_status,
                                               self.intra_model,
                                               sorted_output_fts)
                torch.autograd.grad(outputs=intra_loss,
                                    inputs=self.intra_model.parameters(),
                                    retain_graph=False,
                                    allow_unused=True)

                self.train_intra_loss_value.append(intra_loss.item())
                timer['model'] += self.split_time()

                # For c-index logging
                self.train_output_value = self.concat(self.train_output_value, sorted_output)
                self.train_status_value = self.concat(self.train_status_value, sorted_status)
                self.train_gt_value = self.concat(self.train_gt_value, sorted_gt)

                timer['statistics'] += self.split_time()

        # ---------------------------
        # 2) Inter-model training
        # ---------------------------
        if train:
            inter_output_s2, inter_output_s1s2, inter_output_fts = self.inter_model(
                self.train_wsi_fts,
                self.train_wsi_risk,
                attr=self.train_attr
            )

            sorted_wsi_st, sorted_inter_output, sorted_wsi_status, sorted_inter_output_fts = utils.sort_survival_time(
                self.train_wsi_st, inter_output_s2, self.train_wsi_status, inter_output_fts
            )

            inter_loss = self.compute_loss(sorted_inter_output,
                                           sorted_wsi_st,
                                           sorted_wsi_status,
                                           self.inter_model,
                                           sorted_inter_output_fts)

            self.inter_optimizer.zero_grad()
            inter_loss.backward()
            self.inter_optimizer.step()
            self.inter_lr = self.inter_optimizer.param_groups[0]['lr']

            # Statistics
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }
            intra_c_index = utils.accuracytest(self.train_gt_value,
                                               -self.train_output_value,
                                               self.train_status_value)
            inter_c_index_s2 = utils.accuracytest(self.train_wsi_st,
                                                  -inter_output_s2,
                                                  self.train_wsi_status)
            inter_c_index_s1s2 = utils.accuracytest(self.train_wsi_st,
                                                    -inter_output_s1s2,
                                                    self.train_wsi_status)

            self.print_log(
                f'Mean training intra_loss: {np.mean(self.train_intra_loss_value):.4f}. '
                f'Mean intra_c-index: {np.mean(intra_c_index)*100:.2f}%.'
            )
            self.print_log(
                f'Mean training inter_loss: {inter_loss:.4f}. '
                f'Mean stage 2 inter_c-index: {np.mean(inter_c_index_s2)*100:.2f}%. '
                f'Mean stage 1-2 c-index: {np.mean(inter_c_index_s1s2)*100:.2f}%. '
                f'inter_lr: {self.inter_lr:.8f}'
            )
            self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # ---------------------------
        # 3) Save the model if requested
        # ---------------------------
        if save_model:
            state_dict = self.inter_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            save_path = os.path.join(self.arg.work_dir, f'inter-{i_fold}-runs-{epoch+1}.pt')
            torch.save(weights, save_path)

    def eval(self,
             epoch: int,
             i_fold: int,
             save_model: bool = False) -> (float, float):
        """
        Perform one epoch of evaluation for both the intra-model and inter-model.

        Args:
            epoch (int): Current epoch index.
            i_fold (int): Current fold index (for cross-validation).
            save_model (bool): Whether to save the model if a new best result is achieved.

        Returns:
            tuple(float, float): Intra-model c-index and Inter-model (stage 1-2) c-index, both in percentages.
        """
        if self.arg.phase == 'test':
            print('\033[92mFold {} - Testing Phase'.format(i_fold))

        self.inter_model.eval()
        self.intra_model.eval()

        process_bar = tqdm(self.data_loader['val'][i_fold], desc='[Valid]')

        # ---------------------------
        # 1) Intra-model pass for evaluation data
        # ---------------------------
        if self.eval_gt_value is None:
            for _, (features, survival_time, status, coors, attr, ids) in enumerate(process_bar):
                with torch.no_grad():
                    features = features.float().cuda(self.output_device)
                    survival_time = survival_time.float().cuda(self.output_device)
                    coors = coors.float().cuda(self.output_device)
                    status = status.long().cuda(self.output_device)
                    attr = attr.long().cuda(self.output_device)

                    if isinstance(self.intra_model, nn.DataParallel):
                        if self.arg.H_coors:
                            output, output_fts = self.intra_model.module.forward(features, coors)
                        else:
                            output, output_fts = self.intra_model.module.forward(features)
                    else:
                        if self.arg.H_coors:
                            output, output_fts = self.intra_model.forward(features, coors)
                        else:
                            output, output_fts = self.intra_model.forward(features)

                    # Accumulate WSI-level features
                    self.eval_wsi_fts = self.concat(self.eval_wsi_fts, output_fts.detach())
                    self.eval_wsi_risk = self.concat(self.eval_wsi_risk, output.detach())
                    self.eval_wsi_st = self.concat(self.eval_wsi_st, survival_time)
                    self.eval_wsi_status = self.concat(self.eval_wsi_status, status)
                    self.eval_attr = self.concat(self.eval_attr, attr)
                    self.eval_wsi_id = self.eval_wsi_id + ids if self.eval_wsi_id else ids

                    # Compute loss for logging
                    sorted_gt, sorted_output, sorted_status, sorted_output_fts = utils.sort_survival_time(
                        survival_time, output, status, output_fts
                    )
                    intra_loss = self.compute_loss(
                        sorted_output,
                        sorted_gt,
                        sorted_status,
                        self.intra_model,
                        sorted_output_fts
                    )
                    if status.sum() == 0:
                        intra_loss[intra_loss != intra_loss] = 0  # handle nan
                    self.eval_intra_loss_value.append(intra_loss.item())
                    self.eval_output_value = self.concat(self.eval_output_value, sorted_output)
                    self.eval_status_value = self.concat(self.eval_status_value, sorted_status)
                    self.eval_gt_value = self.concat(self.eval_gt_value, sorted_gt)

        # ---------------------------
        # 2) Inter-model pass
        # ---------------------------
        with torch.no_grad():
            inter_output_s2, inter_output_s1s2, inter_output_fts = self.inter_model(
                self.eval_wsi_fts,
                self.eval_wsi_risk,
                self.train_wsi_fts,
                self.train_wsi_risk,
                attr=self.eval_attr,
                train_attr=self.train_attr
            )

            sorted_wsi_st, sorted_inter_output, sorted_wsi_status, sorted_inter_output_fts = utils.sort_survival_time(
                self.eval_wsi_st, inter_output_s2, self.eval_wsi_status, inter_output_fts
            )
            inter_loss = self.compute_loss(
                sorted_inter_output,
                sorted_wsi_st,
                sorted_wsi_status,
                self.inter_model,
                sorted_inter_output_fts
            )

            # Compute final metrics
            intra_loss = np.mean(self.eval_intra_loss_value)
            intra_c_index = utils.accuracytest(self.eval_gt_value,
                                               -self.eval_output_value,
                                               self.eval_status_value)
            inter_c_index_s2 = utils.accuracytest(self.eval_wsi_st,
                                                  -inter_output_s2,
                                                  self.eval_wsi_status)
            inter_c_index_s1s2 = utils.accuracytest(self.eval_wsi_st,
                                                    -inter_output_s1s2,
                                                    self.eval_wsi_status)

        # ---------------------------
        # 3) Compare with current best and possibly save
        # ---------------------------
        if intra_c_index > self.intra_best_i_fold_c_index:
            self.intra_best_i_fold_c_index = intra_c_index
            self.intra_best_i_fold_c_index_epoch = epoch + 1

        if intra_c_index > self.intra_best_c_index:
            self.intra_best_c_index = intra_c_index
            self.intra_best_epoch = epoch + 1
            self.intra_best_i_fold = i_fold

        if inter_c_index_s2 > self.inter_best_i_fold_c_index_s2:
            self.inter_best_i_fold_c_index_s2 = inter_c_index_s2
            self.inter_best_i_fold_c_index_s2_epoch = epoch + 1

        if inter_c_index_s1s2 > self.inter_best_i_fold_c_index:
            self.inter_best_i_fold_c_index = inter_c_index_s1s2
            self.inter_best_i_fold_c_index_epoch = epoch + 1
            save_model = True

        if inter_c_index_s1s2 > self.inter_best_c_index:
            self.inter_best_c_index = inter_c_index_s1s2
            self.inter_best_epoch = epoch + 1
            self.inter_best_i_fold = i_fold

        # Save the model if new best
        if save_model:
            state_dict = self.inter_model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            best_model_path = os.path.join(self.arg.work_dir, f'inter_{i_fold}_fold_best_model.pt')
            torch.save(weights, best_model_path)

            # Save evaluation results
            result_dict = {
                'id': self.eval_wsi_id,
                'eval_risk': inter_output_s1s2.cpu().detach().numpy(),
                'eval_survival_time': self.eval_wsi_st.cpu().numpy(),
                'eval_status': self.eval_wsi_status.cpu().numpy()
            }
            with open(os.path.join(self.arg.work_dir, f'inter_{i_fold}_fold_best_model.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)

        self.print_log(
            f'Mean intra_val loss: {intra_loss:.4f}. current epoch intra_c-index: '
            f'{np.mean(intra_c_index)*100:.2f}%. best intra_c-index: {self.intra_best_i_fold_c_index*100:.2f}%.'
        )
        self.print_log(
            f'Mean inter_val loss: {inter_loss:.4f}. current epoch stage 2 inter_c-index: '
            f'{np.mean(inter_c_index_s2)*100:.2f}%. current epoch stage 1-2 inter_c-index: '
            f'{np.mean(inter_c_index_s1s2)*100:.2f}%. best s2_c-index: '
            f'{self.inter_best_i_fold_c_index_s2*100:.2f}% best c-index: '
            f'{self.inter_best_i_fold_c_index*100:.2f}%.'
        )

        return np.mean(intra_c_index) * 100, np.mean(inter_c_index_s1s2) * 100

    def test_best_model(self, i_fold: int, epoch: int, save_model: bool = False) -> float:
        """
        Load the best saved inter-model weights for the specified fold, evaluate, and return the c-index.

        Args:
            i_fold (int): Fold index to evaluate.
            epoch (int): Epoch index (unused in this snippet).
            save_model (bool): Whether to save the model after testing (unused here).

        Returns:
            float: The c-index obtained by evaluating the best model on the validation set.
        """
        inter_weights_path = os.path.join(self.arg.work_dir, f'inter_{i_fold}_fold_best_model.pt')
        inter_weights = torch.load(inter_weights_path, weights_only=True)

        # If using DataParallel
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            inter_weights = OrderedDict([['module.' + k, v.cuda(self.output_device)]
                                         for k, v in inter_weights.items()])

        self.inter_model.load_state_dict(inter_weights)
        self.arg.print_log = False
        c_index = self.eval(epoch=0, i_fold=i_fold)[0]
        self.arg.print_log = True
        return c_index

    def start(self) -> None:
        """
        Main entry point to start the training or testing process.
        """
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Inter-Model Parameters: {count_parameters(self.inter_model)}')
            self.print_log(f'# Intra-Model Parameters: {count_parameters(self.intra_model)}')

            n_fold_val_best_c_index = []
            for i in range(len(self.data_loader['train'])):
                # Initialize or reset states for each fold
                if i > -1:
                    self._initialize_model(i)
                    self._initialize_optimizer()
                    self.inter_model = self.inter_model.cuda(self.output_device)
                    self.intra_model = self.intra_model.cuda(self.output_device)

                    # Reset best metrics
                    self.inter_best_i_fold_c_index = 0
                    self.inter_best_i_fold_c_index_epoch = 0
                    self.intra_best_i_fold_c_index = 0
                    self.intra_best_i_fold_c_index_epoch = 0
                    self.inter_best_i_fold_c_index_s2 = 0
                    self.inter_best_i_fold_c_index_s2_epoch = 0

                    # Reset placeholders
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

                # ---------------------------
                # Training loop
                # ---------------------------
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    save_model = (
                        ((epoch + 1) % self.arg.save_interval == 0)
                        or ((epoch + 1) == self.arg.num_epoch)
                    ) and ((epoch + 1) > self.arg.save_epoch)

                    self.train(epoch, i_fold=i)
                    self.inter_scheduler.step()

                    if (epoch + 1) % self.arg.eval_interval == 0:
                        self.eval(epoch, i_fold=i)

                # ---------------------------
                # Evaluate the best model
                # ---------------------------
                self.print_log('Evaluating the best model with the best intra_c-index ...')
                intra_best_c_index = self.test_best_model(i, self.intra_best_i_fold_c_index_epoch)
                self.print_log('Evaluating the best model with the best inter_c-index ...')
                inter_best_c_index = self.test_best_model(i, self.inter_best_i_fold_c_index_epoch)
                n_fold_val_best_c_index.append([intra_best_c_index, inter_best_c_index])

            # Summarize
            for i, best_c_index_values in enumerate(n_fold_val_best_c_index):
                self.print_log(
                    f'n_fold: {i}, inter best c-index: {best_c_index_values[1]} (intra best: {best_c_index_values[0]})'
                )

            mean_intra = np.mean([x[0] for x in n_fold_val_best_c_index])
            mean_inter = np.mean([x[1] / 100.0 for x in n_fold_val_best_c_index])
            std_inter = np.std([x[1] / 100.0 for x in n_fold_val_best_c_index])
            self.print_log(
                f'{self.arg.n_fold}_fold: inter best mean intra_c-index: {mean_intra}, '
                f'best mean inter_c-index: {mean_inter}, best std inter_c-index: {std_inter}'
            )

            self.print_log(f'Best inter_c-index: {self.inter_best_c_index}')
            self.print_log(f'Best inter_i_fold: {self.inter_best_i_fold}')
            self.print_log(f'Best inter epoch: {self.inter_best_epoch}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.inter_weights is None:
                raise ValueError('Please appoint --inter_weights for testing.')

            self.arg.print_log = False
            n_fold_val_best_c_index = []
            for i in range(len(self.data_loader['val'])):
                self._initialize_model(i=i)
                self.intra_model.to(self.arg.device[0])
                self.inter_model.to(self.arg.device[0])

                # Reset placeholders
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

                # Run one pass of "train" to compute intra-model features (train=False for inter-model)
                self.train(epoch=0, i_fold=i, train=False)
                intra_cindex, inter_cindex = self.eval(epoch=0, i_fold=i)
                n_fold_val_best_c_index.append([intra_cindex, inter_cindex])

            # Final logging
            for i, fold_scores in enumerate(n_fold_val_best_c_index):
                print(f'n_fold: {i}, best intra_c-index: {fold_scores[0]}, best inter_c-index: {fold_scores[1]}')
            mean_intra_c = np.mean([score[0] for score in n_fold_val_best_c_index])
            mean_inter_c = np.mean([score[1] / 100. for score in n_fold_val_best_c_index])
            std_inter_c = np.std([score[1] / 100. for score in n_fold_val_best_c_index])
            print(
                f'{self.arg.n_fold}_fold, inter best mean intra_c-index: {mean_intra_c}, '
                f'best mean inter_c-index: {mean_inter_c}, best std inter_c-index: {std_inter_c}'
            )
            print('Done.\n')


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    init_seed(p.seed)
    processor = Processor(p)
    processor.start()

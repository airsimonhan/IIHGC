import argparse
import torchvision
import torch.nn as nn
import json
import os
import glob
import random
from skimage.filters import threshold_multiotsu
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from get_feature.utils import *
import pickle
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch import nn
import traceback
import numpy as np
from itertools import product
from random import shuffle
import openslide
class Patches(Dataset):
    def __init__(self, slide: openslide, patch_coors) -> None:
        super().__init__()
        self.slide = slide
        self.patch_coors = patch_coors
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int):
        coor = self.patch_coors[idx]
        # img = self.slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        img = self.slide.read_region((coor[0], coor[1]), 0, (256,256)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.patch_coors)
class ResNetFeature(nn.Module):

    def __init__(self, depth=34, pooling=False, pretrained=True):
        super().__init__()
        assert depth in [18, 34, 50, 101, 152]
        self.pooling = pooling

        if depth == 18:
            base_model = torchvision.models.resnet18(pretrained=pretrained)
            self.len_feature = 512
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 34:
            base_model = torchvision.models.resnet34(pretrained=pretrained)
            self.len_feature = 512
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 50:
            base_model = torchvision.models.resnet50(pretrained=pretrained)
            self.len_feature = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 101:
            base_model = torchvision.models.resnet101(pretrained=pretrained)
            self.len_feature = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        elif depth == 152:
            base_model = torchvision.models.resnet152(pretrained=pretrained)
            self.len_feature = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-2])
        else:
            raise NotImplementedError(f'ResNet-{depth} is not implemented!')

    def forward(self, x):
        x = self.features(x)

        if self.pooling:
            # -> batch_size x C x N
            x = x.view(x.size(0), x.size(1), -1)
            # -> batch_size x C
            x = x.mean(dim=-1)
            return x
        else:
            # Attention! No reshape!
            return x
        
class EfficientNetFeautre(nn.Module):
    def __init__(self):
        super().__init__()
        #['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7']
        base_model = EfficientNet.from_pretrained('efficientnet-b4')
        base_model._fc = nn.Identity()
        # print(base_model)
        self.model = base_model
    def forward(self, x):
        x = self.model(x)
        return x

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='multi-modal survival prediction')

    #file_dir
    parser.add_argument('--data_root', default='/pathology_data/' , help='the depth of resnet')
    parser.add_argument('--WSI_info_file',default='../dataset/WSI_info_list/TCGA-LUSC.json',help='the dir of WSI')
    parser.add_argument('--save_dir',default='WSI_patch_features/TCGA-LUSC._2K_sample_threshold',help='the dir of results')
    #sample patch
    parser.add_argument('--num_sample', default=2000, help='the num of sample patch')
    parser.add_argument('--sampled_vis', default=True, help='visualization of sample results')
    parser.add_argument('--patch_size', default=256, type=int, help='the size of patch')
    parser.add_argument('--sample_level', default=0, help='the level of sampled wsi')

    #resnet
    parser.add_argument('--batch_size', default=64, help='the batch size of patch dataloader')
    parser.add_argument('--cnn_depth', default=34, help='the depth of resnet')

    return parser

def get_to_do_list(all_wsi_file, patch_ft_dir):
    to_do_list = []
    done_list = glob.glob(os.path.join(patch_ft_dir, '*_fts.npy')) #
    if len(done_list) > 0:
        done_list = [get_id(_dir) for _dir in done_list]
        for _dir in all_wsi_file:
            id = _dir.split('/')[-1].split('-01Z-00-DX')[0]
            if id not in done_list:
                to_do_list.append(_dir)
    else:
        to_do_list = all_wsi_file
    return to_do_list 

def get_bg_mask(slide, bg_level=2):
    try:
        svs_native_levelimg = slide.read_region((0, 0), bg_level, slide.level_dimensions[bg_level])
    except:
        bg_level = 1
        svs_native_levelimg = slide.read_region((0, 0), bg_level, slide.level_dimensions[bg_level])
    svs_native_levelimg = svs_native_levelimg.convert('L')
    img = np.array(svs_native_levelimg)

    thresholds = threshold_multiotsu(img)
    regions = np.digitize(img, bins=thresholds)
    regions[regions == 1] = 0
    regions[regions == 2] = 1
    return regions

def sample_patch_coors(slide_dir,sample_level=0, num_sample=2000, patch_size=256):

    slide = openslide.open_slide(slide_dir)
    slide_name = os.path.basename(slide_dir)
    slide_name = slide_name[:slide_name.rfind('.')]
    print('curren sample file: ', slide_name)


    patch_coors = []


    bg_mask = get_bg_mask(slide)
    bg_mask = bg_mask.astype(np.uint8)  
    bg_mask = cv2.resize(bg_mask, (slide.level_dimensions[sample_level][0],slide.level_dimensions[sample_level][1]))


    th_num = patch_size * 3 / 4 * patch_size * 3 / 4

    #预定义采样位置
    
    num_row, num_col = bg_mask.shape
    num_row = num_row - patch_size
    num_col = num_col - patch_size
    start = 0
    row_col = list(product(range(start,num_row,patch_size), range(start,num_col,patch_size)))

    num=0
    print("Start sampling")
    cycle_num = 1
    while(num<num_sample):
        row_col = list(product(range(start,num_row,patch_size), range(start,num_col,patch_size)))
        shuffle(row_col)
        for y, x in row_col:
            if num >= num_sample:
                break
            bg_patch = bg_mask[y:y+patch_size,x:x+patch_size]
            if np.count_nonzero(bg_patch[:,:]-1) >= th_num:
                patch_coors.append((x, y))
                num = num + 1
        cycle_num = cycle_num + 1 
        start = patch_size // cycle_num
    
    print('patch_coors********************', len(patch_coors))
    print(f"get patch_coors after sample_regions, and shape is {type(patch_coors)}, len: {len(patch_coors)}")

    return patch_coors, bg_mask
def extract_ft(slide_dir: str, patch_coors, depth=18, batch_size=16):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    slide = openslide.open_slide(slide_dir)


    model_ft = ResNetFeature(depth=depth, pooling=True, pretrained=True) #
    model_ft = model_ft.to(device)
    model_ft.eval()

    dataset = Patches(slide, patch_coors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    fts = []
    for _patches in dataloader:
        _patches = _patches.to(device)
        with torch.no_grad():
            _fts = model_ft(_patches)
        fts.append(_fts)
            

    fts = torch.cat(fts, dim=0)
    assert fts.size(0) == len(patch_coors)
    return fts
if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()
    
    patch_ft_dir = os.path.join(arg.save_dir,'patch_efficient_ft')
    patch_coors_dir = os.path.join(arg.save_dir,'patch_coor')
    sampled_vis_dir = os.path.join(arg.save_dir,'sampled_vis')

    check_dir(arg.save_dir)
    check_dir(patch_ft_dir)
    check_dir(patch_coors_dir)
    check_dir(sampled_vis_dir)


    all_wsi_dir = []
    all_wsi_file = []
    wsi_id_list = []
    to_do_list = []

    all_wsi_dir = [dirpath for dirpath, dirnames, filenames in os.walk(arg.data_root) if dirpath.count(os.sep) == 5 ]


    for i in range(len(all_wsi_dir)):
      file_list = glob.glob(os.path.join(all_wsi_dir[i], '*.svs'))
      if len(file_list)>0:
          all_wsi_file.append(file_list)
      else:
          print(all_wsi_dir[i].split('/')[-1])
    all_wsi_file = np.array(all_wsi_file).reshape(-1)
    to_do_list = get_to_do_list(all_wsi_file,patch_ft_dir)


    if len(to_do_list) > 0 :
        for _idx, _dir in enumerate(to_do_list):
            # try:
                print(f'{_idx + 1}/{len(to_do_list)}: processing slide {_dir}...')

                _id = _dir.split('/')[-1].split('-01Z-00-DX')[0]
                _patch_coors, bg_mask = sample_patch_coors(_dir, arg.sample_level, num_sample=arg.num_sample,
                                                           patch_size=arg.patch_size)

                with open(os.path.join(patch_coors_dir, f'{_id}_coors.pkl'), 'wb') as fp:
                    pickle.dump(_patch_coors, fp)

                if arg.sampled_vis:
                    _vis_img_dir = os.path.join(sampled_vis_dir, f'{_id}_sampled_patches.jpg')
                    print(f'saving sampled patch_slide visualization {_vis_img_dir}...')
                    _vis_img = draw_patches_on_slide(_dir, _patch_coors, mini_frac=32)  # bg_mask,
                    with open(_vis_img_dir, 'w') as fp:
                        _vis_img.save(fp)

                print(f'extracting feature...')
                fts = extract_ft(_dir, _patch_coors, arg.cnn_depth, arg.batch_size)
                np.save(os.path.join(patch_ft_dir, f'{_id}_fts.npy'), fts.cpu().numpy())
    print()
    


    

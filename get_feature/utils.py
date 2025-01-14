import os
import numpy as np
import  openslide
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
SAMPLED = 2
SAMPLED_COLOR = [0, 0, 255]
def check_dir(dir):
    if os.path.exists(dir) is False:
        os.mkdir(dir)
def get_id(_dir):
    return os.path.splitext(os.path.split(_dir)[1])[0].split('_')[0]
def get_just_gt_level(slide: openslide, size):
    level = slide.level_count - 1
    while level >= 0 and slide.level_dimensions[level][0] < size[0] and \
            slide.level_dimensions[level][1] < size[1]:
        level -= 1
    return level
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    regions = {
        "layer1": [],
        "layer2": []
    }
    # Annotations = root.findall('Annotation')
    Annotations = root.findall('Annotations')
    if len(Annotations)>1:
        for i in range(len(Annotations)):
            if i == 0:
                regions_layer1 = Annotations[i].find('Regions').findall('Region')
                for i in range(len(regions_layer1)):
                    vertices = [(int(float(vertex.attrib['X'])), int(float(vertex.attrib['Y']))) for vertex in
                                regions_layer1[i].find('Vertices').findall('Vertex')]
                    regions['layer1'].append(vertices)
                    # regions['layer1'][i] += vertices
            elif i==1:
                regions_layer2 = Annotations[i].find('Regions').findall('Region')
                for i in range(len(regions_layer2)):
                    vertices = [(int(float(vertex.attrib['X'])), int(float(vertex.attrib['Y']))) for vertex in
                                regions_layer2[i].find('Vertices').findall('Vertex')]
                    regions['layer2'].append(vertices)
                
    else:
        regions_layer1 = Annotations[0].find('Annotation').find('Coordinates').findall('Coordinate')
        vertices = [(int(float(vertex.attrib['X'])), int(float(vertex.attrib['Y']))) for vertex in regions_layer1]
        regions['layer1'].append(vertices)

    return regions


def create_polygon(vertices, slide):
    image_size = slide.level_dimensions[0]
    mask = Image.new('L', image_size, 0)
    for vertice in vertices:
        ImageDraw.Draw(mask).polygon(vertice, outline=1, fill=1)
    mask = np.array(mask,dtype='uint8')
    return mask


def gather_sampled_patches(patch_coors, mini_size, mini_frac) -> np.array:
    # generate sampled area mask
    sampled_mask = np.zeros((mini_size[0], mini_size[1]), np.uint8)
    for _coor in patch_coors:
        coor_2 = 256
        coor_3 = 256
        _mini_coor = (int(_coor[0] / mini_frac), int(_coor[1] / mini_frac))
        _mini_patch_size = (int(coor_2 / mini_frac), int(coor_3 / mini_frac))
        sampled_mask[_mini_coor[0]:_mini_coor[0] + _mini_patch_size[0],
        _mini_coor[1]:_mini_coor[1] + _mini_patch_size[1]] = SAMPLED
    sampled_mask = sampled_mask.transpose(1,0)
    return sampled_mask
def fuse_img_mask(img: np.array, mask: np.array, alpha=0.7) -> Image:
    assert img.shape[:2] == mask.shape
    img = img.copy()
    if (mask != 0).any():
        img[mask != 0] = alpha * img[mask != 0] + (1 - alpha) * np.array(SAMPLED_COLOR)

    print(f"return img :{img.shape}")
    img = Image.fromarray(img).convert('RGB')
    return img

def draw_patches_on_slide(slide_dir, patch_coors,bg_mask=None, mini_frac=32):
    slide = openslide.open_slide(slide_dir)
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(np.int64)
    mini_level = get_just_gt_level(slide, mini_size)

    img = slide.read_region((0, 0), mini_level, slide.level_dimensions[mini_level]).convert('RGB')
    img = img.resize(mini_size)
    sampled_mask = gather_sampled_patches(patch_coors, mini_size, mini_frac)
    print(f"sampled_mask shape: {sampled_mask.shape}, type of img:  {type(img)}, type: {type(sampled_mask)}")
    sampled_patches_img = fuse_img_mask(np.asarray(img), sampled_mask)
    if bg_mask is not None:
        sampled_patches_img = fuse_img_mask(np.asarray(img), cv2.resize(bg_mask, mini_size))

    # print(f"sampled_patches_img shape: {sampled_patches_img.shape}")
    img.close()
    return sampled_patches_img
def draw_mask_on_slide(slide_dir, bg_mask=None, mini_frac=32):
    slide = openslide.open_slide(slide_dir)
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(np.int64)
    mini_level = get_just_gt_level(slide, mini_size)

    img = slide.read_region((0, 0), mini_level, slide.level_dimensions[mini_level]).convert('RGB')
    img = img.resize(mini_size)
    if bg_mask is not None:
        sampled_patches_img = fuse_img_mask(np.asarray(img), cv2.resize(bg_mask, mini_size))

    # print(f"sampled_patches_img shape: {sampled_patches_img.shape}")
    img.close()
    return sampled_patches_img
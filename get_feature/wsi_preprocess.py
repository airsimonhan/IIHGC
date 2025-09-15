import os, sys

os.add_dll_directory(r"D:/anaconda3/Library/openslide-bin-4.0.0.8-windows-x64/bin")  # 替换为你的路径

from histolab.slide import Slide
from histolab.tiler import GridTiler, RandomTiler, ScoreTiler
from histolab.scorer import NucleiScorer
from histolab.masks import TissueMask

import json
import csv
import random
import numpy as np
from pathlib import Path
import io
from datetime import datetime  # 新增导入，用于时间戳生成
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import requests
import torch
import torch.nn as nn
import torchvision.transforms as TT
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_B4_Weights

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QFileDialog, QComboBox,
    QLineEdit, QProgressBar, QLabel, QTextEdit, QScrollArea, QCheckBox, QDialog, QScrollArea as QtScrollArea
)
from PySide6.QtCore import QThread, QObject, Signal, Qt
from PySide6.QtGui import QPixmap

import logging



if getattr(sys, 'frozen', False):
    base_path = Path(sys.executable).parent
else:
    base_path = Path(__file__).parent

def get_tile_index(path):
    name = path.stem
    parts = name.split('_')
    if len(parts) >= 2 and parts[0] == 'tile':
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1

# ---------- ROI Mask Class ----------
class ROIMask:
    """基于XML注解的自定义ROI掩码，支持缩略图缩放"""

    def __init__(self, xml_path, scale_factor=64.0):
        logging.info(f"Initializing ROIMask with xml_path: {xml_path}, scale_factor: {scale_factor}")
        self.xml_path = xml_path
        self.scale_factor = scale_factor
        self.annotations = self.parse_xml()

    def parse_xml(self):
        logging.info(f"Parsing XML: {self.xml_path}")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        annotations = []
        for annotation in root.findall('.//Annotation'):
            coords = []
            for coord in annotation.findall('.//Coordinate'):
                x = float(coord.get('X'))
                y = float(coord.get('Y'))
                coords.append((x, y))
            annotations.append(coords)
        logging.info(f"Parsed {len(annotations)} annotations")
        return annotations

    def __call__(self, slide):
        logging.info("Generating mask for slide")
        w, h = slide.dimensions
        thumb_size = (int(w / self.scale_factor), int(h / self.scale_factor))
        mask = Image.new('1', thumb_size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in self.annotations:
            if len(polygon) > 2:
                scaled_polygon = [(x / self.scale_factor, y / self.scale_factor) for (x, y) in polygon]
                draw.polygon(scaled_polygon, fill=1)
        logging.info("Mask generated")
        return np.array(mask)

# ---------- XML Annotation Classes ----------


class XMLAnnotation:
    def __init__(self):
        logging.info("Initializing XMLAnnotation")
        self.root = ET.Element("ASAP_Annotations")
        self.annotations = ET.SubElement(self.root, "Annotations")
        self.groups = ET.SubElement(self.root, "AnnotationGroups")

    def add_tile_annotation(self, tile_index, coordinates, color=None):
        logging.info(f"Adding tile annotation: index {tile_index}, coordinates {coordinates}")
        if color is None:
            color = self.random_hex_color()
        ann = ET.SubElement(self.annotations, "Annotation")
        ann.set("Name", f"Tile_{tile_index}")
        ann.set("Type", "Polygon")
        ann.set("PartOfGroup", "None")
        ann.set("Color", color)
        coords_elem = ET.SubElement(ann, "Coordinates")
        points = [
            (coordinates[0], coordinates[1]),
            (coordinates[2], coordinates[1]),
            (coordinates[2], coordinates[3]),
            (coordinates[0], coordinates[3]),
            (coordinates[0], coordinates[1])
        ]
        for order, (x, y) in enumerate(points):
            coord = ET.SubElement(coords_elem, "Coordinate")
            coord.set("Order", str(order))
            coord.set("X", str(x))
            coord.set("Y", str(y))
        logging.info("Tile annotation added")

    def random_hex_color(self):
        color = f"#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}"
        logging.info(f"Generated random color: {color}")
        return color

    def save(self, filename):
        logging.info(f"Saving XML to {filename}")
        tree = ET.ElementTree(self.root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
        logging.info("XML saved")

# ---------- Feature Extraction Worker ----------


class FeatureWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)

    def __init__(self, tile_dir, model_name, output_file):
        super().__init__()
        logging.info(f"Initializing FeatureWorker: tile_dir {tile_dir}, model {model_name}, output {output_file}")
        self.tile_dir = Path(tile_dir)
        self.model_name = model_name
        self.output_file = output_file
        self.cancel_flag = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.script_dir = base_path
        self.batch_size = 32  # 用于批量处理
        self._load_model()
        self.transform = TT.Compose([
            TT.Resize((224, 224)),
            TT.ToTensor(),
            TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logging.info("FeatureWorker initialized")

    def _load_model(self):
        logging.info(f"Loading model: {self.model_name}")
        model_dir = self.script_dir / 'models'  # self.script_dir 已是 base_path
        model_dir.mkdir(parents=True, exist_ok=True)
        fname_map = {'resnet18': 'resnet18.pth', 'efficientnet_b4': 'efficientnet_b4.pth'}
        weight_file = model_dir / fname_map[self.model_name]

        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
            'efficientnet_b4': 'https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth'
        }
        if not weight_file.exists():
            self.status.emit(f"正在下载 {self.model_name} 权重...")
            logging.info(f"Downloading weights for {self.model_name}")
            try:
                response = requests.get(model_urls[self.model_name], timeout=60, stream=True)
                response.raise_for_status()
                with open(weight_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.status.emit(f"权重已保存至: {weight_file}")
                logging.info(f"Weights saved to: {weight_file}")
            except Exception as e:
                self.status.emit(f"下载失败: {e}, 使用预训练权重")
                logging.error(f"Download failed: {e}, using pretrained weights")
                weight_file = None
        try:
            if self.model_name == 'resnet18':
                if weight_file and weight_file.exists():
                    net = models.resnet18(weights=None)
                    state_dict = torch.load(weight_file, map_location=self.device, weights_only=True)
                    net.load_state_dict(state_dict)
                else:
                    net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                modules = list(net.children())[:-1]
                self.feature_dim = 512
                self.model = nn.Sequential(*modules)
            else:  # efficientnet_b4
                if weight_file and weight_file.exists():
                    net = models.efficientnet_b4(weights=None)
                    state_dict = torch.load(weight_file, map_location=self.device, weights_only=True)
                    net.load_state_dict(state_dict)
                else:
                    net = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                modules = list(net.features)
                self.model = nn.Sequential(*modules, nn.AdaptiveAvgPool2d(1))
                self.feature_dim = 1792
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            self.status.emit(f"模型加载失败: {e}")
            logging.error(f"Model loading failed: {e}")
            raise

    def _safe_open_image(self, image_path):
        """安全打开图像，处理iCCP块过大的问题"""
        logging.info(f"Safely opening image: {image_path}")
        try:
            # 尝试直接打开图像
            img = Image.open(image_path)
            img = img.convert("RGB")  # 确保图像是RGB格式
            logging.info(f"Image opened successfully: {image_path}")
            return img
        except ValueError as e:
            logging.error(f"ValueError opening image {image_path}: {e}")
            if "Decompressed Data Too Large" in str(e):
                # 处理iCCP块过大的问题
                return self._repair_png_image(image_path)
            else:
                raise

    def _repair_png_image(self, image_path):
        """修复包含过大iCCP块的PNG图像"""
        logging.info(f"Repairing PNG image: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()

            # 查找并移除iCCP块
            iccp_marker = b'iCCP'
            if iccp_marker in img_data:
                # 查找iCCP块的起始位置
                start_idx = img_data.index(iccp_marker) - 4  # 块长度字段在iCCP前4字节

                # 查找块的CRC校验字段（在块数据后4字节）
                length = int.from_bytes(img_data[start_idx:start_idx + 4], 'big')
                end_idx = start_idx + 8 + length + 4  # iCCP标记+空字节+压缩方法+数据+CRC

                # 移除iCCP块
                safe_img_data = img_data[:start_idx] + img_data[end_idx:]

                # 从修复的数据创建图像
                img = Image.open(io.BytesIO(safe_img_data))
                img = img.convert("RGB")
                logging.info(f"Image repaired successfully: {image_path}")
                return img
            else:
                raise ValueError("iCCP块未找到，但解压错误仍存在")
        except Exception as e:
            self.status.emit(f"无法修复图像 {image_path}: {str(e)}")
            logging.error(f"Failed to repair image {image_path}: {e}")
            raise

    def run(self):
        logging.info("Starting feature extraction run")
        tile_paths = sorted(self.tile_dir.glob('tile_*_*.png'), key=get_tile_index)
        if not tile_paths:
            self.status.emit(f"目录中未找到瓦片文件: {self.tile_dir}")
            logging.warning(f"No tile files found in {self.tile_dir}")
            return
        feats = []
        total = len(tile_paths)
        batch = []
        for i in range(0, total, self.batch_size):
            if self.cancel_flag or QThread.currentThread().isInterruptionRequested():
                self.status.emit("特征提取已取消")
                logging.info("Feature extraction canceled")
                break
            batch_paths = tile_paths[i:i + self.batch_size]
            self.status.emit(f"[{self.model_name}] 正在处理批次 {i // self.batch_size + 1}/{(total - 1) // self.batch_size + 1}")
            logging.info(f"Processing batch {i // self.batch_size + 1}/{(total - 1) // self.batch_size + 1}")
            try:
                imgs = []
                names = []
                for p in batch_paths:
                    img = self._safe_open_image(p)
                    if img is None:
                        self.status.emit(f"加载图像失败 {p.name}")
                        logging.warning(f"Failed to load image {p.name}")
                        continue
                    imgs.append(self.transform(img))
                    names.append(p.name)
                if not imgs:
                    continue
                x = torch.stack(imgs).to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                vecs = out.view(out.size(0), -1).cpu().numpy()
                for name, vec in zip(names, vecs):
                    feats.append((name, vec))
                self.progress.emit(int((i + len(batch_paths)) / total * 100))
                logging.info(f"Batch processed, progress: {int((i + len(batch_paths)) / total * 100)}%")
            except Exception as e:
                self.status.emit(f"处理批次失败: {str(e)}")
                logging.error(f"Batch processing failed: {e}")
        if not feats:
            self.status.emit("无有效瓦片用于特征提取")
            logging.warning("No valid tiles for feature extraction")
            return
        try:
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['tile_name'] + [f'f{i}' for i in range(self.feature_dim)])
                for name, vec in feats:
                    writer.writerow([name] + vec.tolist())
            self.finished.emit(self.output_file)
            logging.info(f"CSV saved to {self.output_file}")
        except Exception as e:
            self.status.emit(f"CSV 保存失败: {str(e)}")
            logging.error(f"CSV save failed: {e}")

# ---------- Sampling & Processing Worker ----------


class Worker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    mask_image = Signal(str)
    locate_image = Signal(str)
    log = Signal(str)

    def __init__(self, files, params, locate_params):
        super().__init__()
        logging.info(f"Initializing Worker with {len(files)} files")
        self.files = files
        self.params = params
        self.locate_params = locate_params
        self.cancel_flag = False
        self.current_progress = 0

    def _timestamp(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Generated timestamp: {ts}")
        return ts

    def run(self):
        logging.info("Starting Worker run")
        total_files = len(self.files)
        for i, (svs, xml) in enumerate(self.files):
            try:
                if self.cancel_flag or QThread.currentThread().isInterruptionRequested():
                    self.log.emit(f"[{self._timestamp()}] 处理已取消")
                    logging.info("Processing canceled")
                    self.finished.emit("Canceled")
                    break
                self.current_progress = int(i / total_files * 100)
                self.progress.emit(self.current_progress)
                logging.info(f"Progress updated: {self.current_progress}%")
                sample = Path(svs).stem
                out_dir = Path(self.params['output_dir']) / sample
                out_dir.mkdir(parents=True, exist_ok=True)
                self.log.emit(f"[{self._timestamp()}] 正在处理 {sample} ({i + 1}/{total_files})")
                self.log.emit(f"[{self._timestamp()}] 输出目录设置为: {out_dir}")
                logging.info(f"Processing {sample} ({i + 1}/{total_files}), output dir: {out_dir}")

                # 加载 slide
                try:
                    logging.info(f"Loading slide: {svs}")
                    slide = Slide(svs, processed_path=str(out_dir))
                    self.log.emit(f"[{self._timestamp()}] 切片已加载，尺寸: {slide.dimensions}")
                    logging.info(f"Slide loaded, dimensions: {slide.dimensions}")
                except Exception as e:
                    self.log.emit(f"[{self._timestamp()}] 加载切片 {svs} 失败: {str(e)}")
                    logging.error(f"Failed to load slide {svs}: {e}")
                    continue

                # 处理掩码
                mask = None
                if self.params['use_manual'] and xml and os.path.exists(xml):
                    try:
                        logging.info(f"Using manual annotation: {xml}")
                        mask = ROIMask(xml, scale_factor=self.locate_params['scale_factor'])
                        self.log.emit(f"[{self._timestamp()}] 使用人工注解: {xml}")
                    except Exception as e:
                        self.log.emit(f"[{self._timestamp()}] 人工注解失败: {str(e)}, 回退至自动掩码")
                        logging.error(f"Manual annotation failed: {e}, falling back to auto mask")
                        mask = TissueMask()
                else:
                    mask = TissueMask()
                    logging.info("Using automatic TissueMask")

                # 生成并保存掩码图像
                try:
                    logging.info("Generating mask image")
                    mask_img = slide.locate_mask(mask, scale_factor=self.locate_params['scale_factor'], alpha=self.locate_params['alpha'], outline=self.locate_params['outline'])
                    fp = out_dir / f"{sample}_tissue_mask.png"
                    mask_img.save(str(fp))
                    self.mask_image.emit(str(fp))
                    self.log.emit(f"[{self._timestamp()}] 组织掩码已保存至: {fp}")
                    logging.info(f"Mask image saved to: {fp}")
                except Exception as e:
                    self.log.emit(f"[{self._timestamp()}] 掩码生成失败: {str(e)}")
                    logging.error(f"Mask generation failed: {e}")
                    continue

                # 初始化 tiler
                m = self.params['sampling_method']
                ts = (self.params['tile_size'], self.params['tile_size'])
                lvl = self.params['level']
                pct = self.params['tissue_percent']
                chk = self.params['tissue_detection']
                nt = self.params['num_tiles']
                tiler = None
                logging.info(f"Initializing tiler: method {m}, tile_size {ts}, level {lvl}")
                if m == 'dense':
                    tiler = GridTiler(
                        tile_size=ts, level=lvl, check_tissue=chk, tissue_percent=pct,
                        pixel_overlap=0, suffix='.png'
                    )
                elif m == 'random':
                    tiler = RandomTiler(
                        tile_size=ts, n_tiles=nt, level=lvl, seed=7,
                        check_tissue=chk, tissue_percent=pct, suffix='.png'
                    )
                else:  # score-based
                    tiler = ScoreTiler(
                        scorer=NucleiScorer(), tile_size=ts, n_tiles=nt, level=lvl,
                        check_tissue=chk, tissue_percent=pct, pixel_overlap=0,
                        suffix='.png'
                    )
                logging.info("Tiler initialized")

                # 提取 tile
                if tiler:
                    try:
                        self.log.emit(f"[{self._timestamp()}] 开始 {m} 采样 {sample}...")
                        logging.info(f"Starting {m} sampling for {sample}")
                        tiler.extract(slide, mask)
                        # 更新进度（假设提取占50%）
                        self.progress.emit(self.current_progress + int(50 / total_files))
                        logging.info("Tile extraction completed, progress updated")
                        tile_files = sorted([p for p in out_dir.glob('tile_*_*.png') if get_tile_index(p) != -1],
                                            key=get_tile_index)
                        self.log.emit(f"[{self._timestamp()}] 采样完成，在 {out_dir} 中找到 {len(tile_files)} 个瓦片")
                        logging.info(f"Sampling complete, found {len(tile_files)} tiles in {out_dir}")
                        if not tile_files:
                            self.log.emit(f"[{self._timestamp()}] 警告: 未生成瓦片。请检查切片内容、掩码或参数。")
                            logging.warning("No tiles generated")
                        else:
                            # 保存 tile 坐标
                            tile_coords = []
                            for idx, tile_file in enumerate(tile_files):
                                parts = tile_file.stem.split('_')
                                if len(parts) >= 4:
                                    coords_str = parts[-1]
                                    x1, y1, x2, y2 = map(int, coords_str.split('-'))
                                    tile_coords.append({
                                        'wsi_name': sample,
                                        'tile_index': idx,
                                        'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                                    })
                            json_path = out_dir / f"{sample}_tile_coords.json"
                            with open(json_path, 'w') as f:
                                json.dump(tile_coords, f, indent=4)
                            self.log.emit(f"[{self._timestamp()}] 瓦片坐标已保存至: {json_path}")
                            logging.info(f"Tile coordinates saved to {json_path}")
                    except Exception as e:
                        self.log.emit(f"[{self._timestamp()}] 瓦片提取失败: {str(e)}")
                        logging.error(f"Tile extraction failed: {e}")
                        continue

                # 生成定位图像
                try:
                    self.log.emit(f"[{self._timestamp()}] 正在生成瓦片定位图像...")
                    logging.info("Generating tile location images")
                    no_label_img = tiler.locate_tiles(slide, mask, scale_factor=self.locate_params['scale_factor'], alpha=self.locate_params['alpha'], outline=self.locate_params['outline'])
                    no_label_path = out_dir / f"{sample}_tiles_boxes.png"
                    no_label_img.save(str(no_label_path))
                    self.locate_image.emit(str(no_label_path))
                    self.log.emit(f"[{self._timestamp()}] 已保存瓦片定位图像: {no_label_path}")
                    logging.info(f"Saved tile boxes image: {no_label_path}")

                    # 生成带标签图像
                    label_img = no_label_img.copy()  # 复制原始图像以添加标签
                    draw = ImageDraw.Draw(label_img)

                    # 加载字体，字体大小为8，失败时使用默认字体
                    try:
                        font = ImageFont.truetype("arial.ttf", 8)
                        logging.info("Font 'arial.ttf' loaded")
                    except IOError:
                        self.log.emit(f"[{self._timestamp()}] 字体 'arial.ttf' 未找到，使用默认字体")
                        logging.warning("Font 'arial.ttf' not found, using default")
                        font = ImageFont.load_default()

                    # 从文件名中获取瓦片信息
                    tile_files = sorted([p for p in out_dir.glob('tile_*_*.png') if get_tile_index(p) != -1],
                                        key=get_tile_index)
                    if not tile_files:
                        self.log.emit(f"[{self._timestamp()}] 未找到瓦片，跳过标签生成")
                        logging.info("No tiles found, skipping label generation")
                    else:
                        logging.info(f"Adding labels to {len(tile_files)} tiles")
                        for tile_file in tile_files:
                            # 解析文件名，例如 tile_0_level0_1760-4449-2272-4961.png
                            parts = tile_file.stem.split('_')
                            if len(parts) >= 4:
                                index = get_tile_index(tile_file)  # 获取瓦片索引
                                coords_str = parts[-1]  # 获取坐标部分
                                x1, y1, x2, y2 = map(int, coords_str.split('-'))  # 解析坐标
                                # 使用瓦片左上角作为标签位置
                                x = x1
                                y = y1
                                # 根据 scale_factor 缩放坐标
                                scaled_x = int(x / self.locate_params['scale_factor'])
                                scaled_y = int(y / self.locate_params['scale_factor'])
                                # 在缩放后的位置绘制索引
                                draw.text((scaled_x, scaled_y), str(index), fill='white', font=font)

                    # 保存带标签图像
                    label_path = out_dir / f"{sample}_tiles_boxes_with_labels.png"
                    label_img.save(str(label_path))
                    self.locate_image.emit(str(label_path))
                    self.log.emit(f"[{self._timestamp()}] 已保存带标签瓦片定位图像: {label_path}")
                    logging.info(f"Saved labeled tile boxes image: {label_path}")

                except Exception as e:
                    self.log.emit(f"[{self._timestamp()}] 生成瓦片定位图像失败: {str(e)}")
                    logging.error(f"Tile location image generation failed: {e}")

                self.log.emit(f"[{self._timestamp()}] {sample} 处理完成")
                logging.info(f"Processing of {sample} completed")
                self.finished.emit(str(out_dir))
            except Exception as e:
                self.log.emit(f"[{self._timestamp()}] 处理 {svs} 时发生意外错误: {str(e)}")
                logging.error(f"Unexpected error processing {svs}: {e}")
                continue

        if not self.cancel_flag and not QThread.currentThread().isInterruptionRequested():
            self.progress.emit(100)
            self.log.emit(f"[{self._timestamp()}] 所有文件处理成功")
            logging.info("All files processed successfully")

# ---------- Image Dialog for Original View ----------


class ImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        logging.info(f"Initializing ImageDialog for {image_path}")
        super().__init__(parent)
        self.setWindowTitle("原始图像")
        layout = QVBoxLayout()
        scroll_area = QtScrollArea()
        scroll_area.setWidgetResizable(True)
        self.label = QLabel()
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap)
        scroll_area.setWidget(self.label)
        layout.addWidget(scroll_area)
        self.setLayout(layout)
        screen = QApplication.primaryScreen().geometry()
        self.setMaximumSize(screen.width() - 100, screen.height() - 100)
        self.resize(min(pixmap.width(), screen.width() - 100), min(pixmap.height(), screen.height() - 100))
        self.label.mousePressEvent = self.close_on_click
        scroll_area.setStyleSheet("""
            QScrollBar:horizontal {
                height: 30px;
            }
            QScrollBar:vertical {
                width: 30px;
            }
        """)
        logging.info("ImageDialog initialized")

    def close_on_click(self, event):
        logging.info("Closing ImageDialog on click")
        self.close()

# ---------- Main Window ----------


class MainWindow(QMainWindow):
    def __init__(self):
        logging.info("Initializing MainWindow")
        super().__init__()
        self.setWindowTitle('SVS 工具与特征提取')
        self.resize(1200, 800)
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # --- Sampling Configuration Tab ---
        sampling_tab = QWidget()
        samp_layout = QVBoxLayout(sampling_tab)
        gb_samp = QGroupBox('采样参数')
        grid_samp = QGridLayout(gb_samp)
        grid_samp.setRowStretch(1, 1)
        for r in range(2, 9):
            grid_samp.setRowStretch(r, 3)
        self.btn_file = QPushButton('选择 SVS 文件')
        self.btn_folder = QPushButton('选择 文件夹')
        self.txt_files = QTextEdit()
        self.txt_files.setReadOnly(True)
        self.txt_files.setFixedHeight(100)
        self.le_output = QLineEdit(str(base_path / 'output'))
        self.le_output.setToolTip("指定输出目录，用于存储采样结果和相关文件。默认路径为当前脚本目录下的'output'文件夹。")
        self.btn_output_dir = QPushButton('选择 输出目录')
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(['密集采样', '随机采样', '基于评分的采样'])
        self.cmb_method.setToolTip("选择采样方法：密集采样覆盖整个区域；随机采样生成指定数量的随机瓦片；基于评分的采样优先选择核密度高的区域。")
        self.le_tile_size = QLineEdit('512')
        self.le_tile_size.setToolTip("设置瓦片尺寸（像素），推荐值为256-1024，过小可能导致细节丢失，过大增加计算负担。")
        self.le_num_tiles = QLineEdit('100')
        self.le_num_tiles.setToolTip("指定生成瓦片数量（适用于随机或基于评分采样），范围1-1000，值越大处理时间越长。")
        self.cmb_level = QComboBox()
        self.cmb_level.addItems(['0', '1', '2'])
        self.cmb_level.setToolTip("选择切片级别：0为最高分辨率，2为最低；较高级别加速处理但牺牲细节。")
        self.le_tissue_pct = QLineEdit('70.0')
        self.le_tissue_pct.setToolTip("组织百分比阈值，用于过滤瓦片中组织含量；范围0.0-100.0，推荐70.0以排除空白区域。")
        self.cb_tissue = QCheckBox('启用组织检测')
        self.cb_tissue.setChecked(True)
        self.cb_tissue.setToolTip("启用后，仅提取包含足够组织的瓦片，提高采样效率。")
        self.cb_manual = QCheckBox('使用人工标注')
        self.cb_manual.setChecked(False)
        self.cb_manual.setToolTip("启用后，使用XML文件中的人工注解作为掩码；需确保对应XML文件存在。")
        self.has_xml_files = False
        self.le_scale_factor = QLineEdit('32')
        self.le_scale_factor.setToolTip("缩放因子，用于生成缩略图和掩码；推荐16-64，值越大计算越快但精度降低。")
        self.le_alpha = QLineEdit('128')
        self.le_alpha.setToolTip("掩码透明度，范围0-255；0为完全透明，255为不透明。")
        self.le_outline = QLineEdit('red')
        self.le_outline.setToolTip("掩码轮廓颜色，支持英文颜色名如'red'或十六进制如'#FF0000'。")
        self.le_linewidth = QLineEdit('1')
        self.le_linewidth.setToolTip("轮廓线宽（像素），范围1-10；用于可视化定位图像。")
        for w in (self.le_output, self.cmb_method, self.le_tile_size,
                  self.le_num_tiles, self.cmb_level, self.le_tissue_pct,
                  self.le_scale_factor, self.le_alpha, self.le_outline, self.le_linewidth):
            w.setMinimumHeight(30)
        for b in (self.btn_file, self.btn_folder, self.btn_output_dir):
            b.setMinimumHeight(40)
        grid_samp.addWidget(QLabel('文件：'), 0, 0)
        grid_samp.addWidget(self.btn_file, 0, 1)
        grid_samp.addWidget(self.btn_folder, 0, 2)
        grid_samp.addWidget(self.txt_files, 1, 0, 1, 3)
        grid_samp.addWidget(QLabel('输出目录：'), 2, 0)
        grid_samp.addWidget(self.le_output, 2, 1)
        grid_samp.addWidget(self.btn_output_dir, 2, 2)
        grid_samp.addWidget(QLabel('采样方式：'), 3, 0)
        grid_samp.addWidget(self.cmb_method, 3, 1)
        grid_samp.addWidget(QLabel('瓦片大小：'), 4, 0)
        grid_samp.addWidget(self.le_tile_size, 4, 1)
        grid_samp.addWidget(QLabel('数量：'), 5, 0)
        grid_samp.addWidget(self.le_num_tiles, 5, 1)
        grid_samp.addWidget(QLabel('级别：'), 6, 0)
        grid_samp.addWidget(self.cmb_level, 6, 1)
        grid_samp.addWidget(QLabel('组织百分比：'), 7, 0)
        grid_samp.addWidget(self.le_tissue_pct, 7, 1)
        grid_samp.addWidget(self.cb_tissue, 7, 2)
        grid_samp.addWidget(QLabel('人工标注：'), 8, 0)
        grid_samp.addWidget(self.cb_manual, 8, 1)
        grid_samp.addWidget(QLabel('缩放因子：'), 9, 0)
        grid_samp.addWidget(self.le_scale_factor, 9, 1)
        grid_samp.addWidget(QLabel('透明度：'), 10, 0)
        grid_samp.addWidget(self.le_alpha, 10, 1)
        grid_samp.addWidget(QLabel('轮廓颜色：'), 11, 0)
        grid_samp.addWidget(self.le_outline, 11, 1)
        grid_samp.addWidget(QLabel('线宽：'), 12, 0)
        grid_samp.addWidget(self.le_linewidth, 12, 1)
        samp_layout.addWidget(gb_samp)
        tabs.addTab(sampling_tab, '设置')

        # --- Processing Tab ---
        processing_tab = QWidget()
        proc_layout = QVBoxLayout(processing_tab)
        proc_layout.setContentsMargins(0, 0, 0, 0)  # 消除布局内边距以压缩空白
        proc_layout.setSpacing(0)  # 保留全局零间距
        self.btn_start = QPushButton('开始处理')
        self.btn_cancel = QPushButton('取消')
        self.pb = QProgressBar()
        self.lb_status = QLabel('就绪')
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        view_layout = QHBoxLayout()
        self.lbl_mask = QLabel()
        self.lbl_loc = QLabel()
        self.lbl_mask.setScaledContents(True)
        self.lbl_loc.setScaledContents(True)
        self.lbl_mask.setMinimumHeight(300)
        self.lbl_loc.setMinimumHeight(300)
        self.scroll_mask = QScrollArea()
        self.scroll_loc = QScrollArea()
        self.scroll_mask.setWidgetResizable(True)
        self.scroll_loc.setWidgetResizable(True)
        self.scroll_mask.setWidget(self.lbl_mask)
        self.scroll_loc.setWidget(self.lbl_loc)
        view_layout.addWidget(self.scroll_mask, 1)
        view_layout.addWidget(self.scroll_loc, 1)
        thumb_layout = QHBoxLayout()
        self.thumbs = []
        for _ in range(4):
            thumb = QLabel()
            thumb.setFixedSize(120, 120)
            thumb.setScaledContents(True)
            thumb_layout.addWidget(thumb)
            self.thumbs.append(thumb)
        proc_layout.addWidget(self.btn_start)
        proc_layout.addWidget(self.btn_cancel)
        proc_layout.addWidget(QLabel('进度：'))
        proc_layout.addWidget(self.pb)
        proc_layout.addWidget(QLabel('状态：'))
        proc_layout.addWidget(self.lb_status)
        proc_layout.addLayout(view_layout, stretch=8)  # 图像区分配较小伸展因子
        proc_layout.addLayout(thumb_layout, stretch=8)  # 最小化缩略图区的扩展
        proc_layout.addWidget(QLabel('日志：'))
        proc_layout.addWidget(self.log_text, stretch=3)  # 日志区分配较大伸展因子
        proc_layout.setSpacing(0)  # 已设置间距为0，确保图像区与日志区无缝衔接
        tabs.addTab(processing_tab, '采样')

        # --- Feature Extraction Tab ---
        feature_tab = QWidget()
        feat_layout = QVBoxLayout(feature_tab)
        self.le_tile_dir = QLineEdit()
        self.btn_tile_dir = QPushButton('选择瓦片文件夹')
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(['resnet18', 'efficientnet_b4'])
        self.le_feat_out = QLineEdit()
        self.btn_out_csv = QPushButton('输出CSV路径')
        self.btn_feat_start = QPushButton('开始提取特征')
        self.btn_feat_stop = QPushButton('停止提取')
        self.pb_feat = QProgressBar()
        self.lb_feat_status = QLabel('未开始')
        for w in (self.le_tile_dir, self.cmb_model, self.le_feat_out):
            w.setMinimumHeight(30)
        for b in (self.btn_tile_dir, self.btn_out_csv, self.btn_feat_start, self.btn_feat_stop):
            b.setMinimumHeight(40)
        feat_layout.addWidget(QLabel('瓦片目录：'))
        feat_layout.addWidget(self.le_tile_dir)
        feat_layout.addWidget(self.btn_tile_dir)
        feat_layout.addWidget(QLabel('模型：'))
        feat_layout.addWidget(self.cmb_model)
        feat_layout.addWidget(QLabel('输出CSV：'))
        feat_layout.addWidget(self.le_feat_out)
        feat_layout.addWidget(self.btn_out_csv)
        feat_layout.addWidget(self.btn_feat_start)
        feat_layout.addWidget(self.btn_feat_stop)
        feat_layout.addWidget(self.pb_feat)
        feat_layout.addWidget(self.lb_feat_status)
        tabs.addTab(feature_tab, '特征提取')

        # Signals
        self.btn_file.clicked.connect(self.on_file)
        self.btn_folder.clicked.connect(self.on_folder)
        self.btn_output_dir.clicked.connect(self.on_out)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.cmb_method.currentIndexChanged.connect(self.on_mode)
        self.btn_tile_dir.clicked.connect(self.on_tile_dir)
        self.cmb_model.currentIndexChanged.connect(self.update_feature_output)
        self.btn_out_csv.clicked.connect(self.on_out_csv)
        self.btn_feat_start.clicked.connect(self.on_feat_start)
        self.btn_feat_stop.clicked.connect(self.on_feat_stop)
        self.lbl_mask.mousePressEvent = self.show_original_mask
        self.lbl_loc.mousePressEvent = self.show_original_loc

        # Internal state
        self.files = []
        self.xml_files = []
        self.current_mask_path = None
        self.current_loc_path = None
        logging.info("MainWindow initialized")

    def check_xml_files(self):
        logging.info("Checking XML files")
        self.has_xml_files = True
        missing_xml = []
        for svs_path, xml_path in self.files:
            if not xml_path or not os.path.exists(xml_path):
                self.has_xml_files = False
                missing_xml.append(Path(svs_path).name)
        if self.files:
            if self.has_xml_files:
                self.lb_status.setText("所有文件均有对应的XML注解")
                logging.info("All files have XML annotations")
            else:
                self.cb_manual.setChecked(False)
                if missing_xml:
                    self.lb_status.setText(f"缺少XML注解的文件: {', '.join(missing_xml)}")
                    logging.warning(f"Missing XML for files: {', '.join(missing_xml)}")

    def on_file(self):
        logging.info("Selecting SVS files")
        self.files.clear()
        self.txt_files.clear()
        fs, _ = QFileDialog.getOpenFileNames(self, '选择 SVS 文件', '', 'SVS 文件 (*.svs *.SVS);;所有文件 (*)')
        for f in fs:
            xml = f[:-4] + '.xml'
            if os.path.exists(xml):
                self.files.append((f, xml))
                self.txt_files.append(f"{Path(f).name} (有标注)")
            else:
                self.files.append((f, None))
                self.txt_files.append(f"{Path(f).name} (无标注)")
        self.check_xml_files()
        logging.info(f"Selected {len(self.files)} files")

    def on_folder(self):
        logging.info("Selecting folder")
        self.files.clear()
        self.txt_files.clear()
        d = QFileDialog.getExistingDirectory(self, '选择 文件夹')
        for f in Path(d).iterdir():
            if f.suffix.lower() == '.svs':
                xml = f.with_suffix('.xml')
                if xml.exists():
                    self.files.append((str(f), str(xml)))
                    self.txt_files.append(f"{f.name} (有标注)")
                else:
                    self.files.append((str(f), None))
                    self.txt_files.append(f"{f.name} (无标注)")
        self.check_xml_files()
        logging.info(f"Selected {len(self.files)} files from folder")

    def on_out(self):
        logging.info("Selecting output directory")
        d = QFileDialog.getExistingDirectory(self, '选择 输出目录')
        if d:
            self.le_output.setText(d)
            logging.info(f"Output directory set to {d}")

    def on_mode(self, idx):
        logging.info("Changing sampling mode")
        m = self.cmb_method.currentText()
        self.le_num_tiles.setEnabled(m != '密集采样')
        self.le_tissue_pct.setEnabled(m != '基于评分的采样')
        logging.info(f"Mode changed to {m}")

    def on_start(self):
        logging.info("Starting processing")
        if not self.files:
            self.lb_status.setText('请先选择文件')
            logging.warning("No files selected")
            return
        try:
            tile_size = int(self.le_tile_size.text())
            num_tiles = int(self.le_num_tiles.text())
            tissue_pct = float(self.le_tissue_pct.text())
            scale_factor = int(self.le_scale_factor.text())
            alpha = int(self.le_alpha.text())
            outline = self.le_outline.text()
            linewidth = int(self.le_linewidth.text())
            if tile_size <= 0:
                raise ValueError("瓦片大小必须大于0")
            if num_tiles <= 0 and self.cmb_method.currentText() != '密集采样':
                raise ValueError("瓦片数量必须大于0")
            if tissue_pct < 0 or tissue_pct > 100:
                raise ValueError("组织百分比必须在0-100之间")
            logging.info("Parameters validated")
        except ValueError as e:
            self.lb_status.setText(f"参数错误: {str(e)}")
            logging.error(f"Parameter error: {e}")
            return
        params = {
            'sampling_method': {'密集采样': 'dense', '随机采样': 'random', '基于评分的采样': 'score'}[
                self.cmb_method.currentText()],
            'tile_size': tile_size,
            'num_tiles': num_tiles,
            'level': int(self.cmb_level.currentText()),
            'tissue_percent': tissue_pct,
            'tissue_detection': self.cb_tissue.isChecked(),
            'use_manual': self.cb_manual.isChecked(),
            'output_dir': self.le_output.text()
        }
        locate_params = {
            'scale_factor': scale_factor,
            'alpha': alpha,
            'outline': outline,
            'linewidth': linewidth
        }
        logging.info("Creating Worker")
        self.worker = Worker(self.files, params, locate_params)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.lb_status.setText)
        self.worker.progress.connect(self.pb.setValue)
        self.worker.log.connect(self.log_text.append)
        self.worker.mask_image.connect(self.update_mask_image)
        self.worker.locate_image.connect(self.update_loc_image)
        self.worker.finished.connect(self.on_processing_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.btn_start.setEnabled(False)
        logging.info("Processing thread started")

    def update_mask_image(self, path):
        logging.info(f"Updating mask image: {path}")
        self.current_mask_path = path
        pixmap = QPixmap(path).scaled(self.lbl_mask.size(), Qt.KeepAspectRatio)
        self.lbl_mask.setPixmap(pixmap)
        logging.info("Mask image updated")

    def update_loc_image(self, path):
        logging.info(f"Updating location image: {path}")
        self.current_loc_path = path
        pixmap = QPixmap(path).scaled(self.lbl_loc.size(), Qt.KeepAspectRatio)
        self.lbl_loc.setPixmap(pixmap)
        logging.info("Location image updated")

    def show_original_mask(self, event):
        logging.info("Showing original mask image")
        try:
            if self.current_mask_path:
                dialog = ImageDialog(self.current_mask_path, self)
                dialog.exec()
                logging.info("Original mask shown")
        except Exception as e:
            self.lb_status.setText(f"打开图像失败: {str(e)}")
            logging.error(f"Failed to show original mask: {e}")

    def show_original_loc(self, event):
        logging.info("Showing original location image")
        try:
            if self.current_loc_path:
                dialog = ImageDialog(self.current_loc_path, self)
                dialog.exec()
                logging.info("Original location shown")
        except Exception as e:
            self.lb_status.setText(f"打开图像失败: {str(e)}")
            logging.error(f"Failed to show original location: {e}")

    def on_processing_finished(self, result):
        logging.info(f"Processing finished: {result}")
        self.btn_start.setEnabled(True)
        if result != "Canceled":
            self.lb_status.setText("处理完成")
        self.thread.quit()

    def on_cancel(self):
        logging.info("Canceling processing")
        if hasattr(self, 'worker') and hasattr(self, 'thread') and self.thread.isRunning():
            self.worker.cancel_flag = True
            self.thread.requestInterruption()
            self.lb_status.setText("正在取消...")
            self.thread.quit()
            self.thread.wait()
            logging.info("Processing canceled")

    def on_tile_dir(self):
        logging.info("Selecting tile directory")
        d = QFileDialog.getExistingDirectory(self, '选择瓦片文件夹')
        if not d or not os.path.exists(d):
            self.lb_feat_status.setText(f"无效目录: {d}")
            logging.warning(f"Invalid directory: {d}")
            return
        self.le_tile_dir.setText(d)
        self.update_feature_output()
        logging.info(f"Tile directory selected: {d}")

    def update_feature_output(self):
        logging.info("Updating feature output")
        d = self.le_tile_dir.text()
        if not d:
            return
        sample = Path(d).stem
        model = self.cmb_model.currentText()
        tile_dir = Path(d)
        if tile_dir.exists() and tile_dir.is_dir():
            nt = len(list(tile_dir.glob('*.png')))
        else:
            nt = 0
        dim_map = {'resnet18': 512, 'efficientnet_b4': 1792}
        dim = dim_map.get(model, 0)
        default = f"{sample}_{model}_{nt}tiles_{dim}dim.csv"
        self.le_feat_out.setText(str(Path(d) / default))
        logging.info("Feature output updated")

    def on_out_csv(self):
        logging.info("Selecting output CSV path")
        f, _ = QFileDialog.getSaveFileName(self, '选择输出CSV', self.le_feat_out.text(), 'CSV 文件 (*.csv)')
        if f:
            dir_path = os.path.dirname(f)
            if not os.path.exists(dir_path):
                self.lb_feat_status.setText(f"输出目录不存在: {dir_path}")
                logging.warning(f"Output directory does not exist: {dir_path}")
                return
            self.le_feat_out.setText(f)
            logging.info(f"Output CSV path selected: {f}")

    def cleanup_feature_worker(self):
        logging.info("Cleaning up feature worker")
        if hasattr(self, 'fworker') and hasattr(self, 'fthread'):
            self.fworker.cancel_flag = True
            self.fthread.quit()
            self.fthread.wait()
            del self.fworker
            del self.fthread
        logging.info("Feature worker cleaned up")

    def on_feat_start(self):
        logging.info("Starting feature extraction")
        td = self.le_tile_dir.text()
        out = self.le_feat_out.text()
        mdl = self.cmb_model.currentText()
        if not td or not out:
            self.lb_feat_status.setText('请设置目录和输出路径')
            logging.warning("Missing directory or output path")
            return
        if not os.path.exists(td):
            self.lb_feat_status.setText(f"目录不存在: {td}")
            logging.warning(f"Directory does not exist: {td}")
            return
        self.cleanup_feature_worker()
        self.btn_feat_start.setEnabled(False)
        self.fworker = FeatureWorker(td, mdl, out)
        self.fthread = QThread()
        self.fworker.moveToThread(self.fthread)
        self.fworker.progress.connect(self.pb_feat.setValue)
        self.fworker.status.connect(self.lb_feat_status.setText)
        self.fworker.finished.connect(self.on_feat_finished)
        self.fthread.started.connect(self.fworker.run)
        self.fthread.finished.connect(self.fthread.deleteLater)
        self.fthread.start()
        logging.info("Feature extraction thread started")

    def on_feat_stop(self):
        logging.info("Stopping feature extraction")
        if hasattr(self, 'fworker'):
            self.fworker.cancel_flag = True
            self.fthread.requestInterruption()
            self.lb_feat_status.setText("正在停止...")
        self.btn_feat_start.setEnabled(True)
        logging.info("Feature extraction stopped")

    def on_feat_finished(self, fpath):
        logging.info(f"Feature extraction finished, CSV: {fpath}")
        self.lb_feat_status.setText(f'CSV 已保存: {fpath}')
        self.btn_feat_start.setEnabled(True)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # 添加这一行

    # 配置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'app_{timestamp}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Application started")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
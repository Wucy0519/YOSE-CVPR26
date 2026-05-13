# base_path =: /home/u1120240335/share/VPData/data/videovo/raw_video
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import glob
import cv2
import ffmpeg
import numpy as np

import os
import io
import cv2
import random
import numpy as np
from PIL import Image, ImageOps
import zipfile
import math

import torch
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
from torchvision import transforms
import imageio
import random


def read_dirnames_under_root(root_dir):
    dirnames = [
        name for i, name in enumerate(sorted(os.listdir(root_dir)))
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    print(f'Reading directories under {root_dir}, num: {len(dirnames)}')
    return dirnames

class TrainZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TrainZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TrainZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TrainZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        #
        im = Image.open(io.BytesIO(data))
        return im

class TestZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TestZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TestZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TestZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # im = Image.open(io.BytesIO(data))
        return im

def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])

class GroupRandomHorizontalFlowFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, flowF_group, flowB_group):
        v = random.random()
        if v < 0.5:
            ret_img = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group
            ]
            ret_flowF = [ff[:, ::-1] * [-1.0, 1.0] for ff in flowF_group]
            ret_flowB = [fb[:, ::-1] * [-1.0, 1.0] for fb in flowB_group]
            return ret_img, ret_flowF, ret_flowB
        else:
            return img_group, flowF_group, flowB_group

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img

def create_random_shape_with_random_motion(video_length,
                                           imageHeight=240,
                                           imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10

    region = get_random_shape(edge_num=edge_num,
                              ratio=ratio,
                              height=height,
                              width=width)
    
    region_width, region_height = region.size
    x, y = random.randint(0, imageHeight - region_height), random.randint(
        0, imageWidth - region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks * video_length
    paste_small = int(random.uniform(0, 1)*(video_length-2))
    for _index in range(video_length - 1):
        x, y, velocity = random_move_control_points(x,
                                                    y,
                                                    imageHeight,
                                                    imageWidth,
                                                    velocity,
                                                    region.size,
                                                    maxLineAcceleration=(3,
                                                                         0.5),
                                                    maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert('L'))
    return masks

def create_random_shape_with_random_motion_zoom_rotation(video_length, zoomin=0.9, zoomout=1.1, rotmin=1, rotmax=10, imageHeight=240, imageWidth=432):
    # get a random shape
    assert zoomin < 1, "Zoom-in parameter must be smaller than 1"
    assert zoomout > 1, "Zoom-out parameter must be larger than 1"
    assert rotmin < rotmax, "Minimum value of rotation must be smaller than maximun value !"
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=7)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks*video_length  # -> directly copy all the base masks
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        ### add by kaidong, to simulate zoon-in, zoom-out and rotation
        extra_transform = random.uniform(0, 1)
        # zoom in and zoom out
        if extra_transform > 0.75:
            resize_coefficient = random.uniform(zoomin, zoomout)
            region = region.resize((math.ceil(region_width * resize_coefficient), math.ceil(region_height * resize_coefficient)), Image.NEAREST)
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            region_width, region_height = region.size
        # rotation
        elif extra_transform > 0.5:
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            m = m.rotate(random.randint(rotmin, rotmax))
            # region_width, region_height = region.size
        ### end
        else:
            m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
    return masks

def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
    生成一个具有随机形状的图像，并添加细小纹理的线条。

    edge_num: 可能的尖锐边缘的数量，默认为 9。
    ratio: (0, 1) 范围内的一个值，用于控制贝塞尔曲线的或动幅度，默认为 0.7。
    width: 生成图像的宽度，默认为 432。
    height: 生成图像的高度，默认为 240。
    '''
    points_num = edge_num * 3 + 1
    angles = np.linspace(0, 2 * np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO

    # 使用这个代替 Path.CLOSEPOLY 避免不必要的直线
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
            (2 * ratio * np.random.random(points_num) + 1 - ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)

    # 绘制主路径
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)

    # 添加细小纹理的线条
    # 生成一些随机的起点和终点
    num_texture_lines = int(np.random.uniform(0, 1)*5)  # 纹理线条的数量
    for _ in range(num_texture_lines):
        # start_angle = np.random.uniform(0, 2 * np.pi)
        # end_angle = np.random.uniform(0, 2 * np.pi)
        start_point = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        end_point = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        if np.random.uniform(0, 1) > 0.4:
            line = patches.PathPatch(Path([start_point, end_point], [Path.MOVETO, Path.LINETO]), facecolor='none', edgecolor='black', lw=10)
        else:
            line = patches.PathPatch(Path([start_point, end_point], [Path.MOVETO, Path.LINETO]), facecolor='none', edgecolor='white', lw=10)
        ax.add_patch(line)

    ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.axis('off')  # 移除轴，只保留形状
    fig.canvas.draw()

    # 将 matplotlib 图形转换为 numpy 数组
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1]) + (3,))
    plt.close(fig)

    # 后处理
    data = cv2.resize(data, (width, height))
    data = (1 - np.array(data[:, :, 0] > 0).astype(np.uint8)) * 255
    coordinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(coordinates[0]), np.max(coordinates[0]), np.min(coordinates[1]), np.max(coordinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region

def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)

def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (7, 0.5*np.pi+angle)#+speed

def random_move_control_points(X,
                               Y,
                               imageHeight,
                               imageWidth,
                               lineVelocity,
                               region_size,
                               maxLineAcceleration=(3, 0.5),
                               maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity,
                                     maxLineAcceleration,
                                     dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0)
            or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity

def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean

def mask_to_rectangular(mask, method='bounding_box', padding=0):
    mask = mask[:, 0:1]
    batch_size, _, num_frames, height, width = mask.shape
    rectangular_mask = torch.zeros_like(mask)
    
    for b in range(batch_size):
        for f in range(num_frames):
            current_mask = mask[b, 0, f]  # 当前帧的mask

            rect_mask = _bounding_box_method(current_mask, padding)
            
            rectangular_mask[b, 0, f] = rect_mask
    
    return rectangular_mask.repeat(1, 3, 1, 1, 1)

def _bounding_box_method(mask, padding=0):
    nonzero_indices = torch.nonzero(mask)
    
    if len(nonzero_indices) == 0:
        return mask
    
    y_min = torch.min(nonzero_indices[:, 0]) - padding
    y_max = torch.max(nonzero_indices[:, 0]) + padding + 1
    x_min = torch.min(nonzero_indices[:, 1]) - padding
    x_max = torch.max(nonzero_indices[:, 1]) + padding + 1
    
    y_min = max(0, y_min.item())
    y_max = min(mask.shape[0], y_max.item())
    x_min = max(0, x_min.item())
    x_max = min(mask.shape[1], x_max.item())
    
    rect_mask = torch.zeros_like(mask)
    rect_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 1
    
    return rect_mask

class IVJointPaintSelfDataset(Dataset):
    def __init__(self, 
                 video_path = r"/home/wx1435326/paint/diffpaint/data/debug_video",
                 image_path = r"/home/wx1435326/paint/diffpaint/data/debug_image",
                 mask_path=r"/home/wx1435326/paint/diffpaint/data/debug_mask",
                 modality='video',
                 max_num_frames = 81,
                 frame_interval=1, 
                 load_fps = 8,
                 num_frames=81, 
                 height=480, 
                 width=640,
                 random_mask_prob=0.9,
                 jvxing = False,
                 is_outpainting = False):
        self.is_outpainting = is_outpainting
        self.video_path = video_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.modality = modality.lower()
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.load_fps = load_fps
        self.height = height
        self.width = width
        self.jvxing = jvxing
        
        self.random_mask_prob = random_mask_prob

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.mask_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
        ])

        video_paths = []
        image_paths = []

        video_dir_path = glob.glob(self.video_path+"//*")
        # print(video_dir_path)

        for sub_dir_path in video_dir_path:
            sub_video_path = glob.glob(sub_dir_path+"//*.mp4")
            video_paths = video_paths+sub_video_path
        self.video_paths = video_paths

        for ext in ['*.jpg', '*.png', '*.JPEG']:
            image_paths += glob.glob(os.path.join(self.image_path, '**', ext), recursive=True)
        self.image_paths = image_paths

        if self.modality == 'video':
            self.all_paths = [(p, 'video') for p in self.video_paths]
        elif self.modality == 'image':
            self.all_paths = [(p, 'image') for p in self.image_paths]
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        self.mask_subdirs = []
        if self.mask_path is not None:
            self.mask_subdirs = glob.glob(os.path.join(self.mask_path, "*"))
            
        print(f"Found {len(self.video_paths)} videos, {len(self.image_paths)} images and {len(self.mask_subdirs)} masks.")
        print(f"Loaded {len(self.all_paths)} samples of modality: {self.modality}")
    
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, index):
        path, data_type = self.all_paths[index]

        use_random_mask = True#random.random() < self.random_mask_prob

        if data_type == 'video':
            width = self.width
            height = self.height

            try:
                out, err = (
                        ffmpeg
                        .input(path)
                        .filter_("scale", self.width, self.height, flags='fast_bilinear', force_original_aspect_ratio='increase')
                        .filter_("crop", self.width, self.height)
                        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=180, vsync='0')
                        .run(capture_stdout=True, capture_stderr=True)
                    )
            except:
                return self.__getitem__(index-1)
            frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])  # [..., ::-1]
            video = np.array(frames)
            frame_jiange = max(1, video.shape[0] // self.num_frames)
            video = video[::frame_jiange]
            all_frame_num = video.shape[0]
            # print(video.shape)
            if all_frame_num < self.num_frames:
                return self.__getitem__(index-1)

            # mask
            if use_random_mask:
                get_masks = create_random_shape_with_random_motion_zoom_rotation(
                    self.num_frames, 
                    imageHeight=height, 
                    imageWidth=width)
            else:
                mask_subdir = random.choice(self.mask_subdirs)
                mask_pngs = sorted(glob.glob(os.path.join(mask_subdir, "*.png")))
                if len(mask_pngs) < self.num_frames:
                    get_masks = create_random_shape_with_random_motion_zoom_rotation(
                        self.num_frames,
                        imageHeight=height,
                        imageWidth=width)
                else:
                    min_mask_interval = 1
                    max_mask_interval = 5
                    max_mask_possible_interval = max(min_mask_interval, len(mask_pngs) // self.num_frames)
                    mask_interval = random.randint(min_mask_interval, min(max_mask_interval, max_mask_possible_interval))
                    selected_pngs = mask_pngs[::mask_interval][:self.num_frames]
                    get_masks = []
                    for png in selected_pngs:
                        mask = Image.open(png).convert("L")
                        mask = np.array(mask)
                        mask = (mask > 0).astype(np.float32)
                        get_masks.append(Image.fromarray(mask))

            video_frames = []
            mask_frames = []
            for i in range(self.num_frames):
                this_frame = video[i]
                video_frames.append(self.frame_process(Image.fromarray(this_frame)))
                mask_frames.append(self.mask_process(get_masks[i]))

            video = torch.stack(video_frames, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]
            mask = torch.stack(mask_frames, dim=0)
            mask = mask.repeat(1, 3, 1, 1).permute(1, 0, 2, 3)
            # print(video.shape, mask.shape)
            if self.jvxing:
                mask = mask_to_rectangular(mask.unsqueeze(0)).squeeze(0)

            if self.is_outpainting:
                c, t, h, w = mask.shape
                out_rate = int((0.4/2)*w)
                mask = mask*0
                mask[:, :, :, :out_rate] = 1
                mask[:, :, :, w-out_rate:] = 1

            if mask.sum() == 0 or (1-mask).sum() == 0:
                return self.__getitem__(index-1)

            masked_video = video*(1-mask)

            return {
                "text": "",
                "video": video,
                "mask": mask,
                "masked_video": masked_video,
                "path": path,
                "index": index,
                "modality": "video"
            }
        else:
            try:
                image = Image.open(path).convert("RGB")
                image = self.frame_process(image)
                mask = create_random_shape_with_random_motion_zoom_rotation(1, imageHeight=self.height, imageWidth=self.width)[0]
                mask = self.mask_process(mask).repeat(3, 1, 1)

                video = image.unsqueeze(1)         # [C, T=1, H, W]
                mask = mask.unsqueeze(1)           # [C, T=1, H, W]
                masked_video = video * (1 - mask)

                return {
                    "video": video,
                    "mask": mask,
                    "masked_video": masked_video,
                    "path": path,
                    "modality": "image"
                }
            except Exception as e:
                print(f"[Image error] {e}")
                return self.__getitem__((index - 1) % len(self))
            
class VideoPaintDatasetTest_wcy(Dataset):
    def __init__(self, 
                 base_path = r"/home/share/VPData/data/videovo/raw_video",
                 max_num_frames = 81,
                 frame_interval=1, 
                 load_fps = 8,
                 num_frames=81, 
                 height=480, 
                 width=832,
                 dilate=False,
                 if_random=False):
        self.base_path = base_path
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.load_fps = load_fps
        self.height = height
        self.width = width
        self.dilate = dilate

        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.mask_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
        ])

        if "image_dataset" in base_path:
            img_path = "JPEGImages_432_240"
            ob_mask = "mask_object"
            rm_mask = "mask_random"
        elif "video_dataset" in base_path:
            rm_mask = "random_mask"
            if "davis" in base_path:
                img_path = "video"
                ob_mask = "mask"
            elif "youtube" in base_path:
                img_path = "vos_video"
                ob_mask = "vos_mask" 

        video_dir = os.path.join(base_path, img_path)
        frame_dir = os.path.join(base_path, img_path)
        if not if_random:
            mask_dir = os.path.join(base_path, ob_mask)#mask, Annotations/480p
        else:
            mask_dir = os.path.join(base_path, rm_mask)
        
        self.has_mp4 = os.path.isdir(video_dir) and os.listdir(video_dir) and \
              os.path.isdir(mask_dir) and any(f.endswith(".mp4") for f in os.listdir(mask_dir))
        self.has_frame = os.path.isdir(frame_dir) and os.path.isdir(mask_dir) and \
                 os.listdir(frame_dir) and os.listdir(mask_dir)

        if self.has_mp4:
            video_paths = glob.glob(video_dir+"/*.mp4")
            mask_paths = glob.glob(mask_dir+"/*.mp4")

            video_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in video_paths}
            mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}

            common_keys = sorted(set(mask_dict.keys()) & set(video_dict.keys()))
            self.paired_paths = [(video_dict[k], mask_dict[k]) for k in common_keys]
            
        elif self.has_frame:
            self.paired_paths = []
            seq_names = sorted(list(set(os.listdir(frame_dir)) & set(os.listdir(mask_dir))))
            for name in seq_names:
                frame_seq_dir = os.path.join(frame_dir, name)
                mask_seq_dir = os.path.join(mask_dir, name)
                if os.path.isdir(frame_seq_dir) and os.path.isdir(mask_seq_dir):
                    self.paired_paths.append((frame_seq_dir, mask_seq_dir))

        # print("path>>>>>", self.paired_paths)
        # text
        self.text = [""] * len(self.paired_paths) 

        print("Found {} paired videos with masks.".format(len(self.paired_paths)))
    
    def __len__(self):
        return len(self.paired_paths)
    
    def __getitem__(self, index):
        
        if self.has_mp4:
            return self._load_from_video(index)
        elif self.has_frame:
            return self._load_from_frame(index)

    def _load_from_video(self, index):
        text = self.text[index]
        this_video_path, this_mask_path = self.paired_paths[index]

        cap = cv2.VideoCapture(this_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_jiange = int(fps//self.load_fps)
        try:
            out, _ = (
                    ffmpeg
                    .input(this_video_path)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, capture_stderr=True)
                )
        except:
            return self.__getitem__(index-1)
        video_frames_np = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])   # [..., ::-1]
        video = np.array(video_frames_np)
        all_frame_num = video.shape[0]
        if all_frame_num < self.num_frames:
            return self.__getitem__(index-1)

        # mask
        try:
            mask_out, _ = (
                    ffmpeg
                    .input(this_mask_path)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, capture_stderr=True)
                )
        except:
            return self.__getitem__(index-1)
        mask_frames_np = np.frombuffer(mask_out, np.uint8).reshape([-1, height, width, 3])
        mask_gray = mask_frames_np[..., 0]
        mask = mask_gray
        if mask_frames_np.shape[0] < self.num_frames:
            return self.__getitem__(index - 1)
        
        video_frames = []
        mask_frames = []
        for i in range(self.num_frames):
            this_frame = video[i]
            video_frames.append(self.frame_process(Image.fromarray(this_frame)))
            mask_frames.append(self.mask_process(Image.fromarray(mask[i])))

        video = torch.stack(video_frames, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]
        mask = torch.stack(mask_frames, dim=0)
        mask = mask.repeat(1, 3, 1, 1).permute(1, 0, 2, 3)

        masked_video = video*(1-mask)

        return {
            "text": text,
            "video": video,
            "mask": mask,
            "masked_video": masked_video,
            "path": this_video_path
        }

    def _load_from_frame(self, index):
        text = self.text[index]
        this_frame_path, this_mask_path = self.paired_paths[index]
        print("this_frame_path:", this_frame_path)
        frame_files = sorted(glob.glob(os.path.join(this_frame_path, "*.jpg")))
        mask_files = sorted(glob.glob(os.path.join(this_mask_path, "*.png")))

        video = []
        mask = []
        for f, m in zip(frame_files, mask_files):
            frame = Image.open(f).convert("RGB")
            mask_frame = Image.open(m).convert("L")
            mask_np = np.array(mask_frame)
            mask_binary = np.where(mask_np == 0, 0, 255).astype(np.uint8)
            if self.dilate:
                kernel = np.ones((5, 5), np.uint8) 
                mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
            mask_frame = Image.fromarray(mask_binary)
            video.append(np.array(frame))
            mask.append(np.array(mask_frame))

        video = np.stack(video)  # [T, H, W, 3]
        mask = np.stack(mask)    # [T, H, W]

        video_frames = []
        mask_frames = []
        for i in range(len(frame_files)):
            this_frame = video[i]
            this_mask = mask[i]
            video_frames.append(self.frame_process(Image.fromarray(this_frame)))
            mask_frames.append(self.mask_process(Image.fromarray(this_mask)))

        video = torch.stack(video_frames, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]
        mask = torch.stack(mask_frames, dim=0)
        mask = mask.repeat(1, 3, 1, 1).permute(1, 0, 2, 3)  # [C, T, H, W]

        masked_video = video * (1.0 - mask)

        return {
            "text": text,
            "video": video,
            "mask": mask,
            "masked_video": masked_video,
            "path": this_frame_path
        }
    

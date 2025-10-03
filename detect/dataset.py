import os
import torch
from torchvision import transforms
from timm.data import create_transform
from timm.data import Mixup
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset
from config import get_config
import cv2
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

#root:dataset/train
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images_path = os.path.join(root, 'images')
        self.labels_path = os.path.join(root, 'labels')

        # 获取并排序图片和标签文件名
        self.images_title = sorted(os.listdir(self.images_path))
        self.labels_title = sorted(os.listdir(self.labels_path))

        #数据预处理
        self.transform = transform
        
        print("---------------------------------")
        print(len(self.images_title))
        print(len(self.labels_title))
        print("---------------------------------")

        # 确保图片和标签数量相同
        assert(len(self.images_title) == len(self.labels_title))

        # 确保每个图片对应的标签文件名相同
        for i in range(len(self.images_title)):
            assert(self.images_title[i].split('.')[0] == self.labels_title[i].split('.')[0])

    def __len__(self):
        # 假设所有标签文件夹中的文件数量相同
        return len(self.images_title)

    def __getitem__(self, idx):
        # 获取图像路径
        image_path = os.path.join(self.images_path, self.images_title[idx])   
        # 读取图像和标签
        image = Image.open(image_path).convert('RGB')  # 转换为 RGB
        img_width, img_height = image.size  # 获取图像宽度和高度
        image = np.array(image)  # 转换为 NumPy 数组
  
        # 获取所有标签路径
        label_path = os.path.join(self.labels_path, self.labels_title[idx])
        boxes = []  # 存储所有边界框
        select_label = -1
        max_area = -1.0

        # 读取并解析标签文件
        # cnt = 0
        with open(label_path, 'r') as f:
            for line in f:
                # cnt += 1
                parts = line.strip().split()
                if len(parts) != 5:
                    sys.exit()  # 跳过格式错误的行
                
                # 解析YOLO格式的边界框（忽略类别ID）
                # YOLO格式: class_id x_center y_center width height
                label, x_center, y_center, width, height = parts
                label = int(label)
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)
                
                # 将归一化坐标转换为绝对坐标
                x_center_abs = x_center * img_width
                y_center_abs = y_center * img_height
                width_abs = width * img_width
                height_abs = height * img_height
                
                # 计算边界框的左上角和右下角坐标
                xmin = x_center_abs - (width_abs / 2)
                ymin = y_center_abs - (height_abs / 2)
                xmax = x_center_abs + (width_abs / 2)
                ymax = y_center_abs + (height_abs / 2)
                
                # 确保坐标在图像范围内
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)
                
                # 将边界框添加到列表中
                if len(boxes) == 0:
                    boxes.extend([xmin, ymin, xmax, ymax])
                else:
                    boxes[0] = min(boxes[0], xmin)
                    boxes[1] = min(boxes[1], ymin)
                    boxes[2] = max(boxes[2], xmax)
                    boxes[3] = max(boxes[3], ymax)

                area = (xmax - xmin) * (ymax - ymin)
                if area > max_area:
                    max_area = area
                    select_label = label

        # boxes.append(self.images_title[idx])

        # if cnt > 1:
        #     cvimage = cv2.imread(image_path)
        #     cv2.rectangle(cvimage, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), 2)
        #     cv2.imwrite(f'./rect/{self.images_title[idx]}', cvimage)
        #     print(f"结果图像已保存到: rect/{self.images_title[idx]}")

        box = boxes # [xmin, ymin, xmax, ymax]
        # 应用 Albumentations 变换（如果有）
        if self.transform:
            transformed = self.transform(image=image, bboxes=[box])
            image = transformed['image']
            boxes = transformed['bboxes']
            box = boxes[0]
        
        # 将边界框转换为 Tensor
        box = torch.tensor(box, dtype=torch.float32)
        
        # 返回图像和边界框列表
        return image, box, select_label - 1

def get_transform(config, phase='train'):
    if phase == 'train':
        return A.Compose([
            A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            A.ColorJitter(
                brightness=config.AUG.COLOR_JITTER,
                contrast=config.AUG.COLOR_JITTER,
                saturation=config.AUG.COLOR_JITTER,
                hue=config.AUG.COLOR_JITTER,
                p=0.5
            ) if config.AUG.COLOR_JITTER > 0 else A.NoOp(),
            # A.AutoAugment(policy=config.AUG.AUTO_AUGMENT, p=1.0) if config.AUG.AUTO_AUGMENT is not None else A.NoOp(),
            A.RandomRain(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))  # 不传递类别标签
    else:  # val
        return A.Compose([
            A.Resize(height=config.DATA.IMG_SIZE, width=config.DATA.IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
# 在现有代码下，训练和验证数据基本满足独立同分布的要求：
# 一致的预处理：图像尺寸调整和标准化操作一致，确保数据的分布相同。
# 合理的数据增强：仅在训练阶段进行数据增强，不影响验证数据的独立性和分布。
# 这可以保证模型在训练过程中不会因为数据分布不一致而导致验证阶段表现异常，因此符合独立同分布的要求。

def build_loader(data_path, config):
    # 定义图像处理和数据增强的转换流程
    # 图像将被处理成224x224像素的尺寸，这是许多深度学习模型常用的输入尺寸。
    
    # data_transform = {
    #     "train": create_transform(
    #         input_size=config.DATA.IMG_SIZE,
    #         is_training=True,
    #         color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    #         auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    #         re_prob=config.AUG.REPROB,
    #         re_mode=config.AUG.REMODE,
    #         re_count=config.AUG.RECOUNT,
    #         interpolation=config.DATA.INTERPOLATION,
    #     ),
    #     "val": transforms.Compose([transforms.Resize(int((256 / 224) * config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    #                                transforms.CenterCrop(config.DATA.IMG_SIZE),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 定义图像变换
    data_transform = {
        "train": get_transform(config, phase='train'),
        "val": get_transform(config, phase='val'),
    }
    
    # 实例化训练数据集
    train_data_set = MyDataset(root=os.path.join(data_path, "train"),
                                          transform=data_transform["train"])
    
    # 实例化验证数据集
    val_data_set = MyDataset(root=os.path.join(data_path, "val"),
                                        transform=data_transform["val"])
    
    # # 设置 DataLoader 的工作进程数量，以优化数据加载过程
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))

    # pin_memory 参数指定是否将数据保存在固定内存（页锁定内存）中。
    # 如果设置为 True，则 DataLoader 在返回张量时会将它们放到固定内存中，这可以加速数据在 CPU 和 GPU 之间的传输。
    # num_workers 参数定义了用于数据加载的子进程数量。
    # 这些工作进程可以在 CPU 上预加载数据，而主进程则在 GPU 上训练模型，从而可以并行化数据加载和模型训练，提高效率。
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=config.DATA.BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=config.DATA.PIN_MEMORY,
                                               num_workers=config.DATA.NUM_WORKERS,
                                               drop_last=False,)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=config.DATA.BATCH_SIZE,
                                             shuffle=False,
                                             pin_memory=config.DATA.PIN_MEMORY,
                                             num_workers=config.DATA.NUM_WORKERS,
                                             drop_last=False)
    
    # setup mixup / cutmix 数据增强
    mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. #or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=None,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    
    return train_loader, val_loader, mixup_fn, val_data_set

# def wash():
#     train_loader, val_loader , mixup_fn= build_loader(data_path="./data_cla_new", config=get_config())
#     for batchsz, (data, label) in enumerate(train_loader):
#         if len(label) == 1:
#             print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
#             imgname = label[0][0]
#             txtname = imgname.split('.')[0] + '.txt'
#             os.remove(f'./data_cla_new/train/images/{imgname}')
#             os.remove(f'./data_cla_new/train/labels/{txtname}')
#             # break

def check():
    train_loader, val_loader, mixup_fn = build_loader(data_path="./data_cla_new", config=get_config())
    for batchsz, (data, label) in enumerate(train_loader):
        if label.shape != torch.Size([1, 4]):
            # print(len(label))
            print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
            # break

    for batchsz, (data, label) in enumerate(val_loader):
        if label.shape != torch.Size([1, 4]):
            # print(len(label))
            print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
            # break

def show():
    train_loader, val_loader, mixup_fn = build_loader(data_path="./data_cla_new", config=get_config())
    for batchsz, (data, label) in enumerate(train_loader):
        # print(label.shape == torch.Size([1, 4]))
        print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))

    for batchsz, (data, label) in enumerate(val_loader):
        print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))

# check()
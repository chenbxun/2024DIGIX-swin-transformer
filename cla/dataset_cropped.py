#国赛dataset，使用自定义dataset按指定格式读取目录data_cla_new下的数据
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import sys
from timm.data import create_transform
from timm.data import Mixup
from torchvision.transforms import InterpolationMode

#root:dataset/train
class MyDataset(Dataset):
    def __init__(self, root, transform=None, cropped=True):
        self.images_path = os.path.join(root, 'images')
        self.labels_path = os.path.join(root, 'labels')

        # 获取并排序图片和标签文件名
        self.images_title = sorted(os.listdir(self.images_path))
        self.labels_title = sorted(os.listdir(self.labels_path))

        #数据预处理
        self.transform = transform

        #是否使用裁剪后的图像训练
        self.cropped = cropped
        
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

        # 获取标签路径
        label_path = os.path.join(self.labels_path, self.labels_title[idx])
    
        # 初始化最大面积和对应的标签及边界框
        max_area = 0
        selected_label = -1
        selected_box = [0, 0, 0, 0]

        # 读取并解析标签文件
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    sys.exit()
            
                # 解析YOLO格式的边界框
                label, x_center, y_center, width, height = parts
                label = int(label) - 1
                x_center = float(x_center) * img_width
                y_center = float(y_center) * img_height
                width = float(width) * img_width * 1.2
                height = float(height) * img_height * 1.2
            
                # 计算边界框的左上角和右下角坐标
                xmin = max(0, x_center - width / 2)
                ymin = max(0, y_center - height / 2)
                xmax = min(img_width, x_center + width / 2)
                ymax = min(img_height, y_center + height / 2)
            
                # 计算面积
                area = (xmax - xmin) * (ymax - ymin)
            
                # 更新最大面积及对应的标签和边界框
                if area > max_area:
                    max_area = area
                    selected_label = label
                    selected_box = [xmin, ymin, xmax, ymax]

        # 裁剪图像使用选中的边界框
        cropped_image = image.crop((selected_box[0], selected_box[1], selected_box[2], selected_box[3])) if self.cropped else image

        # # 保存裁剪后的图像以进行验证
        # cropped_image.save(os.path.join('./cropped', self.images_title[idx]))
        # print(f'保存图片{idx}')

        # 应用预处理
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # 返回裁剪后的图像和选中的标签
        return cropped_image, selected_label

def build_loader(config):
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
    data_transform = {
        "train": transforms.Compose([
            # 将图像调整为指定的尺寸，不保持长宽比，避免裁剪
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            
            # 数据增强
            transforms.RandomHorizontalFlip(),
            
            # 颜色抖动
            transforms.ColorJitter(
                brightness=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else 0,
                contrast=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else 0,
                saturation=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else 0,
                hue=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else 0
            ) if config.AUG.COLOR_JITTER > 0 else transforms.Identity(),
            
            # # 自动增强
            # transforms.AutoAugment(policy=config.AUG.AUTO_AUGMENT) if config.AUG.AUTO_AUGMENT != 'none' else transforms.Identity(),
            
            transforms.ToTensor(),
            
            # 随机擦除
            transforms.RandomErasing(
                p=config.AUG.REPROB,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random',
                inplace=False
            ) if config.AUG.REPROB > 0 else transforms.Identity(),    

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        
        "val": transforms.Compose([
            # 将图像调整为指定的尺寸，不保持长宽比，避免裁剪
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    
    # 实例化训练数据集
    train_data_set = MyDataset(root=os.path.join(config.DATA.DATA_PATH, "train"), transform=data_transform["train"], cropped=config.DATA.CROPPED)
    
    # 实例化验证数据集
    val_data_set = MyDataset(root=os.path.join(config.DATA.DATA_PATH, "val"), transform=data_transform["val"], cropped=config.DATA.CROPPED)
    
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=config.DATA.BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=config.DATA.PIN_MEMORY,
                                               num_workers=config.DATA.NUM_WORKERS,
                                               drop_last=True,)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=config.DATA.BATCH_SIZE,
                                             shuffle=False,
                                             pin_memory=config.DATA.PIN_MEMORY,
                                             num_workers=config.DATA.NUM_WORKERS,
                                             drop_last=False)
    
    # setup mixup / cutmix 数据增强
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. #or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=None,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    
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

# def check():
#     train_loader, val_loader, mixup_fn = build_loader(data_path="./data_cla_new", config=get_config())
#     for batchsz, (data, label) in enumerate(train_loader):
#         if label.shape != torch.Size([1, 4]):
#             # print(len(label))
#             print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
#             # break

#     for batchsz, (data, label) in enumerate(val_loader):
#         if label.shape != torch.Size([1, 4]):
#             # print(len(label))
#             print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
#             # break

# def show():
#     train_loader, val_loader, mixup_fn = build_loader(data_path="./data_cla_new", config=get_config())
#     for batchsz, (data, label) in enumerate(train_loader):
#         # print(label.shape == torch.Size([1, 4]))
#         print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))

#     for batchsz, (data, label) in enumerate(val_loader):
#         print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))

# check()
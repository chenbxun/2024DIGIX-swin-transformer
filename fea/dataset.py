from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from timm.data import Mixup
from timm.data import create_transform
from config import get_config
from torchvision.transforms import InterpolationMode

#root:dataset/train
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images_path = os.path.join(root, 'images')
        self.images_title = os.listdir(self.images_path) 
        self.images_title = sorted(self.images_title)

        self.labels_types = ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']
        self.labels_paths = {}
        self.labels_titles = {}     
        for labels_type in self.labels_types:
            self.labels_paths[labels_type] = os.path.join(root, labels_type)
            self.labels_titles[labels_type] = os.listdir(self.labels_paths[labels_type])
            self.labels_titles[labels_type] = sorted(self.labels_titles[labels_type])
        #数据预处理
        self.transform = transform
        
        for labels_type in self.labels_types:
            # print(f"self.images_title:{},self.labels_titles[labels_type]:{}",len(self.images_title),len(self.labels_titles[labels_type]))
            print("---------------------------------")
            print(len(self.images_title))
            print(len(self.labels_titles[labels_type]))
            print("---------------------------------")
            assert(len(self.images_title) == len(self.labels_titles[labels_type]))
        for i in range(len(self.images_title)):
            for labels_type in self.labels_types:
                assert(self.images_title[i].split('.')[0] == self.labels_titles[labels_type][i].split('.')[0])

    def __len__(self):
        # 假设所有标签文件夹中的文件数量相同
        return len(self.images_title)

    def __getitem__(self, idx):
        # 获取图像路径
        image_path = os.path.join(self.images_path, self.images_title[idx])     
        # 获取所有标签路径
        labels = {}
        for labels_type in self.labels_types:
            labels_path = os.path.join(self.labels_paths[labels_type], self.labels_titles[labels_type][idx])
            with open(labels_path, 'r') as f:
                label_str = f.read().split(' ')[0]    
                labels[labels_type] = int(label_str)
        # 读取图像和标签
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image) 
        # # 将文件名加入字典  
        # labels['filename'] = self.images_title[idx]
        # 返回图像和标签字典
        return image, labels

def build_loader(config):
    # 定义图像处理和数据增强的转换流程
    # 图像将被处理成224x224像素的尺寸，这是许多深度学习模型常用的输入尺寸。
    
    data_transform = {
        "train": create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        ),
        "val": transforms.Compose([transforms.Resize(int((256 / 224) * config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
                                   transforms.CenterCrop(config.DATA.IMG_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        # "val": transforms.Compose([transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        #                            transforms.ToTensor(),
        #                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataset(root=os.path.join(config.DATA.DATA_PATH, "train"),
                                transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataset(root=os.path.join(config.DATA.DATA_PATH, "val"),
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
                                               drop_last=True)

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
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
        
    return train_loader, val_loader, mixup_fn, val_data_set

def wash():
    train_loader, _ , mixup_fn= build_loader(data_path="./data_fea", config=get_config())
    for batchsz, (data, label) in enumerate(train_loader):
        for Type in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']:
            if label[Type][0] == '':
                print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
                imgname = label['filename'][0]
                txtname = imgname.split('.')[0] + '.txt'
                os.remove(f'./data_fea/train/images/{imgname}')
                os.remove(f'./data_fea/train/boundary_labels/{txtname}')
                os.remove(f'./data_fea/train/calcification_labels/{txtname}')
                os.remove(f'./data_fea/train/direction_labels/{txtname}')
                os.remove(f'./data_fea/train/shape_labels/{txtname}')
                break

def check():
    train_loader, val_loader, mixup_fn = build_loader(data_path="./data_fea", config=get_config())
    for batchsz, (data, label) in enumerate(train_loader):
        for Type in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']:
            if label[Type][0] == '':
                print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
                break

    for batchsz, (data, label) in enumerate(val_loader):
        for Type in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']:
            if label[Type][0] == '':
                print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))
                break

def show():
    train_loader, val_loader, mixup_fn = build_loader(data_path="./data_fea", config=get_config())
    for batchsz, (data, label) in enumerate(train_loader):
        print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))

    for batchsz, (data, label) in enumerate(val_loader):
        print("第 {} 个Batch label {} and size of data{}".format(batchsz, label, data.shape))

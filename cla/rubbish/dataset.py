import os
import torch
from torchvision import transforms, datasets
from timm.data import create_transform
from timm.data import Mixup
from torchvision.transforms import InterpolationMode

def build_loader(data_path, config):
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
    
    # 实例化训练数据集
    train_data_set = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                          transform=data_transform["train"])
    
    # 实例化验证数据集
    val_data_set = datasets.ImageFolder(root=os.path.join(data_path, "val"),
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
    
    return train_loader, val_loader, mixup_fn



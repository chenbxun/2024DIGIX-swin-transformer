from torchvision import transforms
from torchvision.transforms import InterpolationMode

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
        
        # 自动增强
        transforms.AutoAugment(policy=config.AUG.AUTO_AUGMENT) if config.AUG.AUTO_AUGMENT != 'none' else transforms.Identity(),
        
        # 随机擦除
        transforms.RandomErasing(
            p=config.AUG.REPROB,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random',
            inplace=False
        ) if config.AUG.REPROB > 0 else transforms.Identity(),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    
    "val": transforms.Compose([
        # 将图像调整为指定的尺寸，不保持长宽比，避免裁剪
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

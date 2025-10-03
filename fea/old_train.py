import torch
import torch.nn as nn
import torch.optim as optim
# from swin_transformer_pytorch import swin_t  # 假设你已经将Swin Transformer的实现保存在swin_transformer.py中
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm  
from timm.utils import AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from torch.utils.tensorboard import SummaryWriter  # TensorBoard记录器
import time
import datetime
from config import get_config
from dataset import build_loader  
from build import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, NativeScalerWithGradNormCount

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('runs_logs/swin_tiny')
# # 打印训练集和验证集的数据大小
# print(f'Train Dataset Size: {len(train_loader.dataset)}')
# print(f'Validation Dataset Size: {len(val_loader.dataset)}')

# # 打印 batch 数量
# print(f'Train DataLoader Batch Count: {len(train_loader)}')
# print(f'Validation DataLoader Batch Count: {len(val_loader)}')

# # 查看第一个 batch 的数据形状和数据类型
# train_iter = iter(train_loader)
# inputs, labels = next(train_iter)


# print(f'Input Batch Shape: {inputs.shape}')  # 打印输入的形状
# print(f'Label Batch Shape: {labels.shape}')  # 打印标签的形状
# print(f'Input Data Type: {inputs.dtype}')    # 打印输入数据类型
# print(f'Label Data Type: {labels.dtype}')    # 打印标签数据类型

def validate_model(model, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= total
    accuracy = correct / total
    return val_loss, accuracy

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):# 启用自动混合精度训练
            outputs = model(samples)# 前向传播
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)# 更新学习率调度器
        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            print(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

if __name__ == '__main__':
    config  = get_config()

    data_path = 'data_fea'
    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    #加载数据集
    train_loader, val_loader, mixup_fn = build_loader(data_path, config)
    # 创建模型
    print((f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}"))
    model = build_model(config)
    model.to(device)

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler)
        val_loss, accuracy = validate_model(model, val_loader)
        print(f"Accuracy of the network on the test images: {accuracy}")

    # 开始训练
    max_accuracy = 0.0
    print("Start training")
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # train_loader.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)

        val_loss, accuracy = validate_model(model, val_loader)
        print(f"Loss, Accuracy of the network on the test images: {val_loss},{accuracy * 100}%")

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), 'output/best_model.pth')
            print('save best model')

        print(f'Max accuracy: {max_accuracy}%')
    # writer.close()

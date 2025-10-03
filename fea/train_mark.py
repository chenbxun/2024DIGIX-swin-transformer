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
from sklearn.metrics import f1_score
from config import get_config
from dataset import build_loader  
from build import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, NativeScalerWithGradNormCount
import numpy as np
import matplotlib.pyplot as plt
import os
import random

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
# 用于存储原始图像和对应特征图的列表
original_imgs = []
feature_maps0 = []
feature_maps1 = []
#[batch_size, channels, height, width]
# 在 PyTorch 中，钩子函数（hook function）的 input 参数是一个元组，即使它只包含一个张量。
# 因此，当你访问 input 时，你需要通过下标来获取实际的张量。而在 output 的情况下，它通常直接就是输出张量，不需要通过下标访问。
def hook_patch_embed(module, input, output):
    global original_imgs
    global feature_maps0
    # print(input[0].shape)
    original_imgs.append(input[0])
    feature_maps0.append(output)

def hook_layer0(module, input, output):
    global feature_maps1
    # print(output.shape)
    feature_maps1.append(output)

def visualize_feature_maps(original_img, feature_map, title_o, title_f, save_path=None):
    # print("original_img.shape:", original_img.shape)
    # print("feature_map.shape:", feature_map.shape)
    # 将特征图从GPU移到CPU，并转换为numpy数组
    feature_map = feature_map.detach().cpu().numpy()
    
    # 计算所有通道的平均值
    avg_feature_map = np.mean(feature_map, axis=2)
    
    # 将原始图像从GPU移到CPU，并转换为numpy数组
    original_img = original_img.detach().cpu().numpy()
    
    # 选择批次中的第一个样本
    original_img = original_img[0]
    avg_feature_map = avg_feature_map[0]
    
    # 反归一化处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_img = original_img * std[:, None, None] + mean[:, None, None]
    original_img = np.clip(original_img, 0, 1)  # 确保像素值在 [0, 1] 范围内
    original_img = np.transpose(original_img, (1, 2, 0))  # 改变维度顺序以匹配matplotlib
    
    # 创建一个子图网格
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示原始图像
    ax1.imshow(original_img)
    ax1.set_title(title_o)
    ax1.axis('off')
    
    # 显示平均特征图
    # print("avg_feature_map.shape:", avg_feature_map.shape)
    size = int(np.sqrt(avg_feature_map.shape[0]))
    ax2.imshow(avg_feature_map.reshape(size, size), cmap='viridis')
    ax2.set_title(title_f)
    ax2.axis('off')
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)  # 关闭图像以释放内存
    else:
        # 否则显示图像
        plt.show()

def validate_model(model, dataloader):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    val_loss = 0.0
    total = 0
    correct = {key: 0 for key in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']}
    true_positive = {key: 0 for key in correct.keys()}
    false_positive = {key: 0 for key in correct.keys()}
    false_negative = {key: 0 for key in correct.keys()}
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), {k: v.to(device) for k, v in labels.items()}
            outputs = model(inputs)

            # print("outputs:",outputs)
            # print("labels:",labels)

            # 将标签字典的值转换为张量，并确保其形状为 (batch_size, 4)
            labels_tensor = torch.stack([
                labels['boundary_labels'],
                labels['calcification_labels'],
                labels['direction_labels'],
                labels['shape_labels']
            ], dim=1).float()

            # print("labels_tensor:",labels_tensor)

            # 计算损失
            loss = criterion(outputs, labels_tensor)
            val_loss += loss.item() * inputs.size(0)

            # 计算每个特征的准确率
            predicted = torch.round(torch.sigmoid(outputs))

            # print("predicted:",predicted)

            for i, key in enumerate(correct.keys()):
                correct[key] += (predicted[:, i] == labels[key]).sum().item()
                true_positive[key] += ((predicted[:, i] == 1) & (labels[key] == 1)).sum().item()
                false_positive[key] += ((predicted[:, i] == 1) & (labels[key] == 0)).sum().item()
                false_negative[key] += ((predicted[:, i] == 0) & (labels[key] == 1)).sum().item()

            total += inputs.size(0)

    val_loss /= total
    accuracy = {key: correct[key] / total for key in correct}
    f1_scores = {}

    for key in correct.keys():
        precision = true_positive[key] / (true_positive[key] + false_positive[key] + 1e-6)  # Avoid division by zero
        recall = true_positive[key] / (true_positive[key] + false_negative[key] + 1e-6)  # Avoid division by zero
        f1_scores[key] = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
    
    return val_loss, accuracy, f1_scores


def train_one_epoch(config, model, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    criterion = torch.nn.BCEWithLogitsLoss()
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
        targets = {k: v.cuda(non_blocking=True) for k, v in targets.items()}

        # 将标签字典的值转换为张量，并确保其形状为 (batch_size, 4)
        targets_tensor = torch.stack([
            targets['boundary_labels'],
            targets['calcification_labels'],
            targets['direction_labels'],
            targets['shape_labels']
            ], dim=1).float()

        if mixup_fn is not None:
            samples, targets_tensor = mixup_fn(samples, targets_tensor)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):# 启用自动混合精度训练
            outputs = model(samples)# 前向传播
        loss = criterion(outputs, targets_tensor)
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

        loss_meter.update(loss.item(), targets['boundary_labels'].size(0))
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
    save_dir = './output/run5'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config  = get_config()

    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    #加载数据集
    train_loader, val_loader, mixup_fn, val_data_set = build_loader(config)
    # 创建模型
    print((f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}"))
    model = build_model(config)
    model.to(device)

    # 注册钩子
    hook_o = model.patch_embed.register_forward_hook(hook_patch_embed)
    hook_f0 = model.layers[0].register_forward_hook(hook_layer0)

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler)
        val_loss, accuracy, f1_scores = validate_model(model, val_loader)
        print(f"Accuracy of the network on the test images: {accuracy}")
        print(f"F1 Score of the network on the test images: {f1_scores}")

    # 开始训练
    max_mark = 0.0
    print("Start training")
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # train_loader.sampler.set_epoch(epoch)

        train_one_epoch(config, model, train_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)

        val_loss, accuracy, f1_scores = validate_model(model, val_loader)

        print("for the test images:")
        print(f"loss:{val_loss}")
        print(f"accuracy: {accuracy}")
        print(f"f1_score: {f1_scores}")

        avg_accuracy = sum([v for v in accuracy.values()])/len(accuracy)
        avg_f1_score = sum([v for v in f1_scores.values()])/len(f1_scores)
        print(f"avg_accuracy: {avg_accuracy}")
        print(f"avg_f1_score:{avg_f1_score}")

        mark = 0.3 * avg_accuracy + 0.2 * avg_f1_score
        print(f"mark:{mark}")

        if mark > max_mark:
            max_mark = mark
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print('save best model')

        print(f'max mark: {max_mark}')

        # 可视化特征图
        if (epoch + 1) % 1 == 0:  # 每5个epoch可视化一次
            sample_idx = random.randint(0, len(val_data_set) - 1)
            image, _ = val_data_set[sample_idx]

            model.eval()
            with torch.no_grad():
                input_image = image.unsqueeze(0).to(device)  # 添加 batch 维度
                _ = model(input_image)  # 前向传播以触发钩子
                img_path = os.path.join(save_dir, 'feature_maps')
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                f_save_path = os.path.join(img_path, f"feature_map_epoch_{epoch + 1}.png")
                visualize_feature_maps(original_imgs[-1], feature_maps1[-1], f"Orignal Image at Epoch {epoch + 1}", f"Feature Map at Epoch {epoch + 1}", f_save_path)

        original_imgs.clear()
        feature_maps0.clear()  #确保在每个 epoch 结束时不会累积过多的数据，从而减少显存占用。
        feature_maps1.clear()

    # writer.close()

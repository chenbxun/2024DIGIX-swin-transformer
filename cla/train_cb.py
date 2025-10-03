import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  
from timm.utils import AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import time
import datetime
import torchvision
from sklearn.metrics import f1_score, confusion_matrix
from config import get_config
from dataset import build_loader  
from build import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, NativeScalerWithGradNormCount
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model(model, dataloader, device, writer, epoch, num_classes=6):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 计算每一类的正确预测数和总数
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
            
            # 收集预测和标签以计算F1分数
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss /= total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 计算每一类的准确率
    class_accuracy = []
    for i in range(num_classes):
        if class_total[i] == 0:
            acc = 0
        else:
            acc = class_correct[i] / class_total[i]
        class_accuracy.append(acc)
        writer.add_scalar(f'Validation/Class_{i}_Accuracy', acc, epoch)
        print(f'Validation Accuracy of Class {i}: {acc * 100:.2f}%')
    
    return val_loss, accuracy, macro_f1

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer, feature_maps_hook):
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
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            if config.TRAIN.LR_SCHEDULER.NAME != 'reduce_on_plateau':
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        
        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
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

        # 记录训练损失和学习率到 TensorBoard
        global_step = epoch * num_steps + idx
        writer.add_scalar('Train/Loss', loss_meter.val, global_step)
        writer.add_scalar('Train/LR', lr, global_step)

        # 可视化特征图（仅在第一个 batch 中进行，以避免大量数据）
        if idx == 0 and feature_maps_hook is not None:
            for name, feature in feature_maps_hook.items():
                # 选择第一个样本的特征图
                fmap = feature[0].detach().cpu()
                # 归一化特征图到 [0, 1] 之间
                fmap_min = fmap.min()
                fmap_max = fmap.max()
                fmap = (fmap - fmap_min) / (fmap_max - fmap_min + 1e-5)
                grid = torchvision.utils.make_grid(fmap.unsqueeze(1), nrow=8, normalize=True)
                writer.add_image(f'FeatureMaps/{name}', grid, epoch)

    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    # 记录训练损失到 TensorBoard
    writer.add_scalar('Train/Epoch_Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/Epoch_GradNorm', norm_meter.avg, epoch)
    writer.add_scalar('Train/Epoch_Loss_Scale', scaler_meter.avg, epoch)

def register_feature_maps(model, layers_to_hook):
    feature_maps = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            feature_maps[name] = output
        return hook

    for name, module in model.named_modules():
        if name in layers_to_hook:
            hooks.append(module.register_forward_hook(get_hook(name)))

    return feature_maps, hooks

if __name__ == '__main__':
    config  = get_config()

    # 初始化 TensorBoard 的 SummaryWriter
    log_dir = os.path.join('runs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log directory: {log_dir}")

    data_path = config.DATA.DATA_PATH
    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    # 加载数据集
    train_loader, val_loader, mixup_fn = build_loader(data_path, config)
    
    # 可视化一些训练图像（可选）
    # visualize_augmentations(train_loader)

    # 创建模型
    print((f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}"))
    model = build_model(config)
    model.to(device)

    # 注册特征图的钩子
    # 假设您想要捕获模型中的某些层，例如 'blocks.0.attn' 和 'blocks.0.mlp'
    layers_to_hook = ['blocks.0.attn', 'blocks.0.mlp']  # 根据您的模型结构修改
    feature_maps_hook, hooks = register_feature_maps(model, layers_to_hook)

    # 创建优化器
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    # 构建学习率调度器
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # 定义损失函数（已包含类别权重）
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        # 使用加权损失函数
        # 注意：如果在 build_loader 中已经通过 WeightedRandomSampler 平衡了采样，这里可以不使用权重
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device))  # 替换为实际的权重
        # 或者动态计算权重
        # class_weights = compute_class_weights(...)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler)
        val_loss, accuracy, macro_f1 = validate_model(model, val_loader, device, writer, 0)
        print(f"Accuracy of the network on the test images: {accuracy * 100:.2f}%")
        print(f"Macro F1 Score of the network on the test images: {macro_f1:.4f}")

    # 定义早停参数（可选）
    early_stopping_patience = 30
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 开始训练
    max_combined_metric = 0.0
    print("Start training")
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                       loss_scaler, writer, feature_maps_hook)

        # 在每个 epoch 结束后进行验证
        val_loss, accuracy, macro_f1 = validate_model(model, val_loader, device, writer, epoch)
        combined_metric = 0.3 * accuracy + 0.2 * macro_f1
        print(f"Loss, Accuracy of the network on the test images: {val_loss}, {accuracy * 100:.2f}%")
        print(f"Macro F1 Score of the network on the test images: {macro_f1:.4f}")
        print(f"Combined Metric (0.3*Accuracy + 0.2*F1): {combined_metric:.4f}")

        # 记录验证指标到 TensorBoard
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        writer.add_scalar('Validation/Macro_F1', macro_f1, epoch)
        writer.add_scalar('Validation/Combined_Metric', combined_metric, epoch)

        # 更新学习率调度器
        if config.TRAIN.LR_SCHEDULER.NAME == 'reduce_on_plateau':
            lr_scheduler.step(val_loss)  # 使用验证损失作为依据
        elif config.TRAIN.LR_SCHEDULER.NAME == 'cosine_restart':
            lr_scheduler.step(epoch + 1)
        elif config.TRAIN.LR_SCHEDULER.NAME == 'cyclic':
            pass  # CyclicLR 在 train_one_epoch 中已调用 scheduler.step()
        else:
            lr_scheduler.step()

        # 早停检查（可选）
        if config.TRAIN.LR_SCHEDULER.NAME == 'reduce_on_plateau':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'output/run4/best_model.pth')
                print('Saved best model')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
        else:
            # 对于其他调度器，直接使用 combined_metric 进行保存和早停
            if combined_metric > max_combined_metric:
                max_combined_metric = combined_metric
                torch.save(model.state_dict(), 'output/run4/best_model.pth')
                print('Saved best model')

        print(f'Max Combined Metric: {max_combined_metric:.4f}')

    # 关闭 TensorBoard 的 SummaryWriter
    writer.close()

    # 移除钩子
    for hook in hooks:
        hook.remove()

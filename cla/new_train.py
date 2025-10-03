import torch
import torchvision
from tqdm import tqdm  
from timm.utils import AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import time
import datetime
from sklearn.metrics import f1_score
from config import get_config
from dataset import build_loader  
from build import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, NativeScalerWithGradNormCount
from torch.utils.tensorboard import SummaryWriter  # 导入 SummaryWriter
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

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer):
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

        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        #     outputs = model(samples)
        outputs = model(samples)
        
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        
        reg = 1e-7
        orth_loss = torch.zeros(1, device='cuda')
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0]).to(param.device)
                orth_loss = orth_loss + (reg * sym.abs().sum())

        loss += orth_loss.item()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
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

    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    # 记录训练损失到 TensorBoard
    writer.add_scalar('Train/Epoch_Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/Epoch_GradNorm', norm_meter.avg, epoch)
    writer.add_scalar('Train/Epoch_Loss_Scale', scaler_meter.avg, epoch)

if __name__ == '__main__':
    config  = get_config()

    # 初始化 TensorBoard 的 SummaryWriter
    log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log directory: {log_dir}")

    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    # 加载数据集
    train_loader, val_loader, mixup_fn, val_data_set = build_loader(config)
    # 创建模型
    print((f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}"))
    model = build_model(config)
    model.to(device)

    # 注册钩子
    hook_o = model.patch_embed.register_forward_hook(hook_patch_embed)
    hook_f0 = model.layers[0].register_forward_hook(hook_layer0)

    # 这里用的是adamw
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     # 这里为0.8
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler)
        val_loss, accuracy, macro_f1 = validate_model(model, val_loader, device, writer, 0)
        print(f"Accuracy of the network on the test images: {accuracy * 100:.2f}%")
        print(f"Macro F1 Score of the network on the test images: {macro_f1:.4f}")

    # 开始训练
    max_combined_metric = 0.0
    max_acc = 0
    print("Start training")
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                       loss_scaler, writer)

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

        if combined_metric > max_combined_metric:
            max_combined_metric = combined_metric
            # torch.save(model.state_dict(), 'output/run4/best_model.pth')
            torch.save(model, os.path.join(log_dir, 'best_model.pth'))
            print('Saved best model')

        print(f'Max Combined Metric: {max_combined_metric:.4f}')

        max_acc = max(max_acc, accuracy)
        print(f'Max Accuracy: {max_acc:.4f}')


        # 可视化特征图
        if (epoch + 1) % 1 == 0:  # 每5个epoch可视化一次
            sample_idx = random.randint(0, len(val_data_set) - 1)
            image, _ = val_data_set[sample_idx]

            model.eval()
            with torch.no_grad():
                input_image = image.unsqueeze(0).to(device)  # 添加 batch 维度
                _ = model(input_image)  # 前向传播以触发钩子
                img_path = os.path.join(log_dir, "visualize")
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                f_save_path = os.path.join(img_path, f"feature_map_epoch_{epoch + 1}.png")
                visualize_feature_maps(original_imgs[-1], feature_maps1[-1], f"Orignal Image at Epoch {epoch + 1}", f"Feature Map at Epoch {epoch + 1}", f_save_path)

        original_imgs.clear()
        feature_maps0.clear()  #确保在每个 epoch 结束时不会累积过多的数据，从而减少显存占用。
        feature_maps1.clear()

    # 关闭 TensorBoard 的 SummaryWriter
    writer.close()

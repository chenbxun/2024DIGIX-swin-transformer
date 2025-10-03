import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm  
from timm.utils import AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import time
import datetime
from config import get_config
from dataset import build_loader  
from build import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_checkpoint, NativeScalerWithGradNormCount
from visualize import visualize_prediction
from metrics import calculate_iou, calculate_oc_ear 
import random
import os

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model(model, dataloader, criterion_detect, criterion_cla, img_size=224):
    model.eval()
    val_loss = 0.0
    total = 0
    total_iou = 0.0
    total_oc = 0.0
    total_ear = 0.0
    correct = 0
    
    with torch.no_grad():
        for inputs, boxes, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)
            outputs = model(inputs) # outputs.size : (batch_size, 4)
            
            # 在未经任何修正的情况下计算损失，以确保模型能够从错误中学习。
            loss_cla = criterion_cla(outputs[:, 4:], labels)
            loss_detect = criterion_detect(outputs[:, :4], boxes)
            loss = loss_cla + loss_detect
            # print(f"修正前outputs: {outputs}")  
            # 输出限制在图像范围内
            outputs[:, 0] = outputs[:, 0].clamp(min=0, max=img_size)  # xmin
            outputs[:, 1] = outputs[:, 1].clamp(min=0, max=img_size)  # ymin
            outputs[:, 2] = outputs[:, 2].clamp(min=0, max=img_size)  # xmax
            outputs[:, 3] = outputs[:, 3].clamp(min=0, max=img_size)  # ymax

            # 修正 xmin > xmax 或 ymin > ymax 的情况，将边框大小设为0
            invalid_x = outputs[:, 0] > outputs[:, 2]
            invalid_y = outputs[:, 1] > outputs[:, 3]
            outputs[invalid_x, 0] = outputs[invalid_x, 2]
            outputs[invalid_y, 1] = outputs[invalid_y, 3]
            # print(f"修正后outputs: {outputs}")  
            
           
            # 在计算指标（IoU、OC、EAR）之前，对模型输出进行修正，以获得合理的评估结果。
            # 计算 IoU
            # outputs 和 labels 的形状均为 (batch_size, 4)
            ious = calculate_iou(outputs[:, :4], boxes)  # 输出形状为 (batch_size,)

            # print(f"outputs: {outputs}")
            # print(f"labels:{labels}")

            ocs, ears = calculate_oc_ear(outputs[:, :4], boxes)

            # print(f"ocs: {ocs}, ears: {ears}")

            _, predicted = torch.max(outputs[:, 4:].data, 1)
            correct += (predicted == labels).sum().item()

            # 累积损失和 IoU
            val_loss += loss.item() * inputs.size(0)
            total_iou += ious.sum().item()
            total_oc += ocs.sum().item()
            total_ear += ears.sum().item()
            total += inputs.size(0)
    
    # 计算平均损失和平均 IoU
    avg_val_loss = val_loss / total
    avg_iou = total_iou / total
    avg_oc = total_oc / total
    avg_ear = total_ear / total

    avg_acc = correct / total

    return avg_val_loss, avg_iou, avg_oc, avg_ear ,avg_acc

def train_one_epoch(config, model, criterion_detect, criterion_cla, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    # 清空优化器的梯度缓存
    optimizer.zero_grad()

    num_steps = len(data_loader)    # 数据加载器中总共有多少个 batch
    batch_time = AverageMeter()     # 计算每个 batch 所需的时间
    loss_meter = AverageMeter()     # 记录每个 batch 的损失


    norm_meter = AverageMeter()     # 记录梯度的范数
    scaler_meter = AverageMeter()   # 记录损失缩放的情况

    start = time.time()
    end = time.time()

    for idx, (samples, targets, labels) in enumerate(data_loader):  # 遍历所有的 batch
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if mixup_fn is not None:    # 分类的训练代码启用了 Mixup
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):# 启用自动混合精度训练
            outputs = model(samples)# 前向传播

        loss_detect = criterion_detect(outputs[:, :4], targets) 
        loss_cla = criterion_cla(outputs[:, 4:], labels)
        loss = loss_cla + loss_detect / 30.0
        # print('---------------------')
        # print(f"loss_cla:{loss_cla}, loss_detect:{loss_detect}")
        # print('---------------------')
        loss = loss / config.TRAIN.ACCUMULATION_STEPS   #这里没使用梯度累加，ACCUMULATION_STEPS=1

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        #使用 loss_scaler 进行混合精度训练中的梯度缩放。
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            if config.TRAIN.LR_SCHEDULER.NAME != 'reduce_on_plateau':
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
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

    data_path = 'data_cla_new'
    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    #加载数据集
    train_loader, val_loader, mixup_fn, val_data_set = build_loader(data_path, config)
    # 创建模型
    print((f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}"))
    model = build_model(config)
    model.to(device)

    #这里用的是adamw
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     #这里为0.8
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion_detect = torch.nn.SmoothL1Loss()
    criterion_cla = torch.nn.CrossEntropyLoss()

    if config.MODEL.RESUME:
        load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler)
        val_loss, iou, oc, ear, acc = validate_model(model, val_loader, criterion_detect, criterion_cla, img_size)
        print(f"Val loss of the network on the test images: {val_loss}")
        print(f"Iou of the network on the test images: {iou}")
        print(f"Oc of the network on the test images: {oc}")
        print(f"Ear of the network on the test images: {ear}")
        print(f"Acc of the network on the test images: {acc}")

    # 开始训练
    max_iou = 0.0
    max_score = -100.0
    max_acc =0.0
    print("Start training")
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # train_loader.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion_detect, criterion_cla, train_loader, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)

        # 验证模型
        val_loss, iou, oc, ear, acc = validate_model(model, val_loader, criterion_detect, criterion_cla, img_size)
        score = 0.6 * oc - 0.3 * ear + 0.1 * iou
        print(f"Val loss of the network on the test images: {val_loss:.4f}")
        print(f"Iou of the network on the test images: {iou:.4f}")
        print(f"Oc of the network on the test images: {oc:.4f}")
        print(f"Ear of the network on the test images: {ear:.4f}")
        print(f"Score of the network on the test images: {score:.4f}")  
        print(f"Acc of the network on the test images: {acc:.4f}")  

        # # 保存最佳模型
        # if score > max_score:
        #     max_score = score
        #     torch.save(model.state_dict(), 'output/test/best_model.pth')
        #     print('save best model')
        max_score = max(max_score, score)
        print(f'Max Score: {max_score:.4f}')

        max_iou = max(max_iou, iou)
        print(f'Max Iou: {max_iou:.4f}')

        max_acc = max(max_acc, acc)
        print(f'Max Acc: {max_acc:.4f}')

        # labels = [1, 2, 3, 4, 5, 6]
        # label = -1
        # while True:
        #     # 选择一张验证集中的图片（固定或随机）
        #     sample_idx = random.randint(0, len(val_data_set) - 1)  # 随机选择
        #     # 获取样本
        #     image, true_box, label = val_data_set[sample_idx]
        #     if label in labels:
        #         break
    
        # # 运行模型预测
        # model.eval()
        # with torch.no_grad():
        #     input_image = image.unsqueeze(0).to(device)  # 添加 batch 维度
        #     pred_box = model(input_image)  # 输出形状: [1, 4]
        #     pred_box = pred_box.squeeze(0)  # 去除 batch 维度, 形状: [4]
        #     # pred_box = torch.tensor([1,1,223,223])
    
        # # 可视化并保存图像
        # save_path = os.path.join('./output/test/img', f'epoch_{epoch}_sample_{sample_idx}_label_{label}.png')
        # visualize_prediction(image, true_box, pred_box, save_path=save_path)
        # print(f"Visualized sample {sample_idx} for epoch {epoch} and saved to {save_path}")
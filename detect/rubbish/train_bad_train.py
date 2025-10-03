import torch
import torch.optim as optim
# from swin_transformer_pytorch import swin_t  # 假设你已经将Swin Transformer的实现保存在swin_transformer.py中
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm  
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.utils.tensorboard import SummaryWriter  # TensorBoard记录器
from dataset import build_loader  
from build import build_model
from config import get_config

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
    
# 定义模型参数
# model_config = {
#     'hidden_dim': 96,  
#     'layers': [2, 2, 6, 2],  
#     'heads': [3, 6, 12, 24],  
#     'channels': 3,
#     'num_classes': None,  
#     'head_dim': 32,
#     'window_size': 7,
#     'downscaling_factors': (4, 2, 2, 2),  
#     'relative_pos_embedding': True  
# }

data_path = 'data_cla'
batch_size = 16
img_size = 224

train_loader, val_loader = build_loader(data_path, batch_size, img_size)

# 更新类别数
# model_config['num_classes'] = len(train_loader.dataset.classes)

# 创建模型
# model = swin_t(**model_config)
model = build_model(get_config())

pretrained_path = 'pretrained/ckpt_epoch_240.pth'  # 修改为你自己的预训练权重路径
pretrained_dict = torch.load(pretrained_path)
model.load_state_dict(torch.load(pretrained_path), strict=True)  # strict=False允许部分加载
model_dict = model.state_dict()

print("Pretrained keys:", pretrained_dict.keys())
print("Model keys:", model_dict.keys())

# # 过滤出匹配的键
# matched_keys = []
# unmatched_keys = []

# for key in pretrained_dict.keys():
#     if key in model_dict:
#         matched_keys.append(key)
#     else:
#         unmatched_keys.append(key)

# print("Matched keys:")
# for key in matched_keys:
#     print(key)

# print("Unmatched keys:")
# for key in unmatched_keys:
#     print(key)

# 再次检查模型的状态字典，确保加载成功
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.requires_grad}")


def get_warmup_lr_lambda(current_step, warmup_steps, total_steps, base_lr, min_lr):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        return max(min_lr / base_lr, (total_steps - current_step) / (total_steps - warmup_steps))

def get_cosine_annealing_scheduler(optimizer, epochs, warmup_epochs, total_steps, base_lr, min_lr):
    warmup_steps = warmup_epochs * (total_steps // epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=min_lr)
    
    # Combine warmup and cosine annealing
    warmup_lambda = lambda step: get_warmup_lr_lambda(step, warmup_steps, total_steps, base_lr, min_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler._cosine_scheduler = cosine_scheduler
    
    return scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer_config = {
    "name": "adamw",
    "base_lr": 0.0125,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
}


optimizer = optim.AdamW(
    model.parameters(),
    lr=optimizer_config["base_lr"],
    weight_decay=optimizer_config["weight_decay"],
    betas=optimizer_config["betas"],
    eps=optimizer_config["eps"]
)


# 定义损失函数和优化器
criterion = SoftTargetCrossEntropy()
writer = SummaryWriter('runs_logs/swin_tiny')


def validate_model(model, dataloader, criterion):
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch') as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=loss.item(), accuracy=correct / total)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total

        # 验证模型
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)

        # 记录训练和验证的损失及准确率
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'output/best_model.pth')

# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=300)
writer.close()

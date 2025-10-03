import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from config import get_config  # 获取配置的方法
from build import build_model  # 加载模型的方法
import pytorch_grad_cam as pgc
from sklearn.metrics import f1_score, accuracy_score  # 导入所需的评估指标

# 配置
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load('output/run1/best_model.pth')  # 加载训练好的模型权重
model.to(device)
# model.eval()

# 构建测试数据集和数据加载器

test_transform = transforms.Compose([
    transforms.Resize(int((256 / 224) * config.DATA.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(config.DATA.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

config.DATA.BATCH_SIZE = 1
test_data_set = datasets.ImageFolder(root=os.path.join('data_cla', "val"), transform=test_transform)
test_loader = DataLoader(test_data_set, batch_size=config.DATA.BATCH_SIZE, shuffle=False, pin_memory=config.DATA.PIN_MEMORY, num_workers=config.DATA.NUM_WORKERS)

# 准备CSV文件
ids = []
image_names = []
true_labels = []
predicted_labels = []

target_layers = [model.layers[-1].blocks[-1].norm2]
cam = pgc.GradCAM(model=model, target_layers=target_layers)
flag = 0

# 测试模型并收集结果
# with torch.no_grad():
for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc='Testing')):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if labels == 3:
        print(inputs)
        grayscale_cam = cam(input_tensor=inputs)[0, :]
        visualization = pgc.utils.image.show_cam_on_image(inputs[0].cpu().numpy().transpose(1, 2, 0), grayscale_cam, use_rgb=True)
        import matplotlib.pyplot as plt
        plt.imshow(visualization)
        plt.title(f"True: {labels.item() + 1}, Pred: {preds.item() + 1}")
        plt.axis('off')
        plt.close()
        flag += 1
        if flag == 10:
            break
    # 收集ID、图像名称、真实标签和预测标签
    start_id = idx * config.DATA.BATCH_SIZE + 1
    ids.extend(range(start_id, start_id + len(preds)))
    image_names.extend([os.path.splitext(os.path.basename(test_data_set.samples[i][0]))[0] for i in range(start_id - 1, start_id - 1 + len(preds))])
    true_labels.extend((labels.cpu().numpy() + 1))  # 将真实标签加1
    predicted_labels.extend((preds.cpu().numpy() + 1))  # 将预测标签加1


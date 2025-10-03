import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from config import get_config  # 获取配置的方法
from build import build_model  # 加载模型的方法
from dataset import MyDataset, build_loader  # 加载数据集的方法

# 配置
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = build_model(config)
model.load_state_dict(torch.load('output/run2/best_model.pth'))  # 加载训练好的模型权重
model.to(device)
model.eval()

# 构建测试数据集和数据加载器
test_transform = transforms.Compose([
    transforms.Resize(int((256 / 224) * config.DATA.IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(config.DATA.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data_set = MyDataset(root=os.path.join("data_fea", "val"), transform=test_transform)
test_loader = DataLoader(test_data_set, batch_size=config.DATA.BATCH_SIZE, shuffle=False, pin_memory=config.DATA.PIN_MEMORY, num_workers=config.DATA.NUM_WORKERS)

# 准备CSV文件
ids = []
image_names = []
true_labels = {'boundary_labels': [], 'calcification_labels': [], 'direction_labels': [], 'shape_labels': []}
predicted_labels = {'boundary_labels': [], 'calcification_labels': [], 'direction_labels': [], 'shape_labels': []}

#保存模型的测试指标
total = 0
correct = {key: 0 for key in ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']}
true_positive = {key: 0 for key in correct.keys()}
false_positive = {key: 0 for key in correct.keys()}
false_negative = {key: 0 for key in correct.keys()}

# 测试模型并收集结果
with torch.no_grad():
    for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc='Testing')):
        inputs = inputs.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        outputs = model(inputs)
        
        # 将输出转换为概率并取阈值
        predicted = torch.round(torch.sigmoid(outputs))

        #计算模型的测试指标
        for i, key in enumerate(correct.keys()):
            correct[key] += (predicted[:, i] == labels[key]).sum().item()
            true_positive[key] += ((predicted[:, i] == 1) & (labels[key] == 1)).sum().item()
            false_positive[key] += ((predicted[:, i] == 1) & (labels[key] == 0)).sum().item()
            false_negative[key] += ((predicted[:, i] == 0) & (labels[key] == 1)).sum().item()
        total += inputs.size(0)

        predicted = predicted.cpu().numpy().astype(int)

        # 收集ID、图像名称、真实标签和预测标签
        start_id = idx * config.DATA.BATCH_SIZE + 1
        ids.extend(range(start_id, start_id + len(predicted)))
        image_names.extend([os.path.splitext(os.path.basename(test_data_set.images_title[i]))[0] for i in range(start_id - 1, start_id - 1 + len(predicted))])
        
        # 真实标签
        for key in true_labels.keys():
            true_labels[key].extend(labels[key].cpu().numpy())
        
        # 预测标签
        for i, key in enumerate(true_labels.keys()):
            predicted_labels[key].extend(predicted[:, i])

# 写入CSV文件
df_order = pd.DataFrame({'id': ids, 'image_names': image_names})
df_order.to_csv('fea_order.csv', index=False)

df_gt = pd.DataFrame({'id': ids, 'boundary': true_labels['boundary_labels'], 'calcification': true_labels['calcification_labels'], 'direction': true_labels['direction_labels'], 'shape': true_labels['shape_labels']})
df_gt.to_csv('fea_gt.csv', index=False)

df_pre = pd.DataFrame({'id': ids, 'boundary': predicted_labels['boundary_labels'], 'calcification': predicted_labels['calcification_labels'], 'direction': predicted_labels['direction_labels'], 'shape': predicted_labels['shape_labels']})
df_pre.to_csv('fea_pre.csv', index=False)

print("All CSV files have been created successfully.")


accuracy = {key: correct[key] / total for key in correct}
avg_accuracy = sum([v for v in accuracy.values()])/len(accuracy)
f1_scores = {}
for key in correct.keys():
    precision = true_positive[key] / (true_positive[key] + false_positive[key] + 1e-6)  # Avoid division by zero
    recall = true_positive[key] / (true_positive[key] + false_negative[key] + 1e-6)  # Avoid division by zero
    f1_scores[key] = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
avg_f1_score = sum([v for v in f1_scores.values()])/len(f1_scores)
print("mark:",avg_accuracy*0.3+avg_f1_score*0.2)
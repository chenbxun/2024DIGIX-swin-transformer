import cv2

# 配置路径
image_path = './data_cla_new/train/images/0086.jpg'      # 替换为您的图像文件路径
label_path = './data_cla_new/train/labels/0086.txt'      # 替换为您的YOLO标签文件路径
output_path = './output.jpg'    # 替换为您希望保存结果图像的路径

# 读取图像
image = cv2.imread(image_path)
height, width = image.shape[:2]

# 读取YOLO标签文件
with open(label_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        print(f"无效的标签格式: {line}")
        continue
    
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    bbox_width = float(parts[3])
    bbox_height = float(parts[4])
    
    # 将归一化坐标转换为绝对坐标
    x_center_abs = x_center * width
    y_center_abs = y_center * height
    bbox_width_abs = bbox_width * width
    bbox_height_abs = bbox_height * height
    
    xmin = int(x_center_abs - (bbox_width_abs / 2))
    ymin = int(y_center_abs - (bbox_height_abs / 2))
    xmax = int(x_center_abs + (bbox_width_abs / 2))
    ymax = int(y_center_abs + (bbox_height_abs / 2))
    
    # 确保坐标在图像范围内
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width - 1, xmax)
    ymax = min(height - 1, ymax)
    
    # 绘制边界框
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), 2)

# 保存结果图像
cv2.imwrite(output_path, image)
print(f"结果图像已保存到: {output_path}")

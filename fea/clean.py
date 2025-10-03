import os
from PIL import Image

def clean():
    root = 'dataset_initial'
    new_root = 'dataset_initial_afterclean'
    mode = 'val'
    images_title = os.listdir(f'./{root}/{mode}/images')

    labels_types = ['boundary_labels', 'calcification_labels', 'direction_labels', 'shape_labels']
    for title in images_title:
        img = Image.open(f'./{root}/{mode}/images/{title}')
        img_width, img_height = img.size # 获取图像宽度和高度

        title, suffix = title.split('.')
        content = {}
        for type in labels_types:
            with open(f'./{root}/{mode}/{type}/{title}.txt', 'r') as f:
                content[type] = f.readlines()
        # if len(content['boundary_labels']) != 1:
        #     print(content)
        valid_tumors = extract_valid_tumors(content)
        cnt = 1
        for tumor in valid_tumors:
            for type in labels_types:
                with open(f'./{new_root}/{mode}/{type}/{title}_{cnt}.txt', 'w') as f:
                    f.write(' '.join(tumor[type]))
            
            _, x_center, y_center, width, height = tumor['shape_labels']
            x_center = float(x_center) * img_width
            y_center = float(y_center) * img_height
            width = float(width) * img_width * 1.3
            height = float(height) * img_height * 1.3
            
            # 计算边界框的左上角和右下角坐标
            xmin = max(0, x_center - width / 2)
            ymin = max(0, y_center - height / 2)
            xmax = min(img_width, x_center + width / 2)
            ymax = min(img_height, y_center + height / 2)

            img.crop((xmin, ymin, xmax, ymax)).save(f'./{new_root}/{mode}/images/{title}_{cnt}.{suffix}')

            cnt += 1
            
        

def extract_valid_tumors(labels_dict):
    threshold = 0.05
    # 初始化一个字典来存储有效的肿瘤数据
    valid_tumors = []

    # 遍历其中一个标签列表，这里选择shape_labels作为参考
    for shape_label in labels_dict['shape_labels']:
        # 去除字符串末尾的换行符，并按空格分割得到各个部分
        parts = shape_label.strip().split()
        # 获取肿瘤的坐标信息，假设坐标信息由后四个值组成
        tumor_coords = list(map(float, parts[1:]))  # 将坐标转换为浮点数

        # 初始化一个布尔变量，用于标记该肿瘤是否有效
        is_valid = True
        # 创建一个字典来存储该肿瘤的有效标签信息
        tumor_data = {'shape_labels': parts}

        # 检查其他三种标签是否存在相同的坐标信息
        for label_type, label_list in labels_dict.items():
            if label_type == 'shape_labels':
                continue  # 跳过shape_labels，因为我们已经从中选取了坐标信息
            found = False
            for label in label_list:
                parts = label.strip().split()
                label_coords = list(map(float, parts[1:]))  # 将坐标转换为浮点数

                # 检查坐标是否在阈值范围内
                if all(abs(tumor_coord - label_coord) <= threshold for tumor_coord, label_coord in zip(tumor_coords, label_coords)):
                    found = True
                    tumor_data[label_type] = parts  # 存储找到的标签信息
                    break
            if not found:
                is_valid = False
                break  # 如果任何一种类型的标签缺失，则该肿瘤无效

        # 如果该肿瘤有效，添加到有效肿瘤列表中
        if is_valid:
            valid_tumors.append(tumor_data)

    return valid_tumors

# # 示例数据
# labels_dict = {'boundary_labels': ['0 0.466037 0.536768 0.134981 0.236387\n', '0 0.699612 0.417303 0.149612 0.288550\n'],
#                'calcification_labels': ['0 0.466037 0.536768 0.134981 0.236387\n', '0 0.699612 0.417303 0.149612 0.288550\n'],
#                'direction_labels': ['0 0.466037 0.536768 0.134981 0.236387\n', '0 0.699612 0.417303 0.149612 0.288550\n'], 
#                'shape_labels': ['0 0.699612 0.417303 0.149612 0.288550\n', '0 0.466037 0.536768 0.134981 0.236387\n']}

# # 调用函数并打印结果
# valid_tumors = extract_valid_tumors(labels_dict)
# print(valid_tumors)
# '''
# [

# {'shape_labels': ['0', '0.699612', '0.417303', '0.149612', '0.288550'],
# 'boundary_labels': ['0', '0.699612', '0.417303', '0.149612', '0.288550'],
# 'calcification_labels': ['0', '0.699612', '0.417303', '0.149612', '0.288550'],
# 'direction_labels': ['0', '0.699612', '0.417303', '0.149612', '0.288550']},

# {'shape_labels': ['0', '0.466037', '0.536768', '0.134981', '0.236387'],
# 'boundary_labels': ['0', '0.466037', '0.536768', '0.134981', '0.236387'],
# 'calcification_labels': ['0', '0.466037', '0.536768', '0.134981', '0.236387'],
# 'direction_labels': ['0', '0.466037', '0.536768', '0.134981', '0.236387']}

# ]
# '''

clean()
    



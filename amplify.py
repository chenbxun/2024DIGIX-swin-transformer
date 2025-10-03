import os
import cv2
from tqdm import tqdm
import numpy as np

def process_images(input_folder, output_folder):
    """
    对指定文件夹中的所有图像进行自适应直方图均衡化和噪声去除处理，并保存到新的文件夹中。
    
    参数:
    - input_folder (str): 输入图像所在的文件夹路径。
    - output_folder (str): 处理后图像保存的文件夹路径。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]


    # 进度条
    with tqdm(total=len(image_files), desc=f"Processing images in {input_folder}") as pbar:
        for image_file in image_files:
            # 读取图像
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to read {image_path}")
                continue
                
            print(image.shape)
            
            # 噪声去除（中值滤波）
            denoised_image = cv2.medianBlur(image, ksize=3)  # 可以修改 ksize 参数
            
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 可以修改 clipLimit 和 tileGridSize 参数
            # enhanced_image = clahe.apply(denoised_image)
            enhanced_image = clahe.apply(image)
            
            # 保存处理后的图像
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, enhanced_image)
            
            # 更新进度条
            pbar.update(1)


def main():
    # 输入文件夹路径列表
    input_folders = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6'
    ]
    
    # 遍历每个输入文件夹
    for input_folder in input_folders:
        # 计算输出文件夹路径
        output_folder = input_folder + '_detection'
        
        # 处理图像
        process_images(input_folder, output_folder)

if __name__ == "__main__":
    main()
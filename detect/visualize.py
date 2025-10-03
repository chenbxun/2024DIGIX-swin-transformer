import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_prediction(image, true_box, pred_box, save_path=None):
    """
    可视化图像与边界框。

    参数:
    - image (Tensor): 形状为 [3, H, W]，已归一化的图像张量。
    - true_box (Tensor): 形状为 [4]，真实边界框坐标 [xmin, ymin, xmax, ymax]。
    - pred_box (Tensor): 形状为 [4]，预测边界框坐标 [xmin, ymin, xmax, ymax]。
    - save_path (str, optional): 如果提供，将图像保存到此路径；否则显示图像。
    """
    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * std) + mean
    image_np = np.clip(image_np, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)

    # 绘制真实边界框
    xmin, ymin, xmax, ymax = true_box.cpu().numpy()
    width = xmax - xmin
    height = ymax - ymin
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), width, height, 
                                      linewidth=2, edgecolor='green', facecolor='none', label='True Box'))

    # 绘制预测边界框
    xmin_p, ymin_p, xmax_p, ymax_p = pred_box.cpu().numpy()
    width_p = xmax_p - xmin_p
    height_p = ymax_p - ymin_p
    plt.gca().add_patch(plt.Rectangle((xmin_p, ymin_p), width_p, height_p, 
                                      linewidth=2, edgecolor='red', facecolor='none', label='Predicted Box'))

    plt.legend()
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
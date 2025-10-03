import torch

def calculate_iou(box1, box2):
    # IoU 衡量预测框和真实框的重叠部分
    """
    计算两个边界框的 IoU,支持批量输入
    box1: Tensor of shape (batch_size, 4) [xmin, ymin, xmax, ymax]
    box2: Tensor of shape (batch_size, 4) [xmin, ymin, xmax, ymax]
    """
    # 计算交集坐标
    inter_xmin = torch.max(box1[:, 0], box2[:, 0])
    inter_ymin = torch.max(box1[:, 1], box2[:, 1])
    inter_xmax = torch.min(box1[:, 2], box2[:, 2])
    inter_ymax = torch.min(box1[:, 3], box2[:, 3])

    # 计算交集面积
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    # 计算两个边界框的面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 计算 IoU
    iou = inter_area / union_area.clamp(min=1e-6)  # 防止除以零

    return iou

# Overlap Completeness (OC) + Excess Area Ratio (EAR)
# 组合使用两个指标来衡量框的“完整性”和“多余度”：
# Overlap Completeness (OC)：预测框与真实框的交集面积占真实框的比例，确保目标被完整框住。
# Excess Area Ratio (EAR)：预测框中不属于真实框的多余部分占预测框的比例，确保多框住的背景不会太多。
# 当 OC 达到 90% 且 EAR 不超过 50% 时，即认为预测框足够好。
def calculate_oc_ear(pred_box, true_box):
    """
    计算 Overlap Completeness (OC) 和 Excess Area Ratio (EAR)
    pred_box: Tensor of shape (batch_size, 4) [xmin, ymin, xmax, ymax] 预测框
    true_box: Tensor of shape (batch_size, 4) [xmin, ymin, xmax, ymax] 真实框
    """
    # 计算交集坐标
    inter_xmin = torch.max(pred_box[:, 0], true_box[:, 0])
    inter_ymin = torch.max(pred_box[:, 1], true_box[:, 1])
    inter_xmax = torch.min(pred_box[:, 2], true_box[:, 2])
    inter_ymax = torch.min(pred_box[:, 3], true_box[:, 3])

    # 计算交集面积
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    # 计算预测框和真实框的面积
    pred_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
    true_area = (true_box[:, 2] - true_box[:, 0]) * (true_box[:, 3] - true_box[:, 1])

    # 计算 Overlap Completeness (OC): 交集区域占真实框区域的比例
    oc = inter_area / true_area.clamp(min=1e-6)  # 防止除以零

    # 计算 Excess Area Ratio (EAR): 非真实框区域占预测框的比例
    excess_area = pred_area - inter_area
    ear = excess_area / pred_area.clamp(min=1e-6)  # 防止除以零

    return oc, ear
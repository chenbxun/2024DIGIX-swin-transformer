from PIL import Image

def get_image_info(image_path):
    """
    读取图片文件并返回图片的宽度、高度和颜色模式。

    参数:
        image_path (str): 图片文件的路径。

    返回:
        tuple: 包含图片宽度、高度和颜色模式的元组。
    """
    # 打开图片文件
    with Image.open(image_path) as img:
        # 获取图片的宽度和高度
        width, height = img.size
        # 获取图片的颜色模式
        mode = img.mode
        return width, height, mode

# 示例使用
image_path = '1/0086.jpg'  # 替换为你的图片路径
width, height, mode = get_image_info(image_path)
print(f"图片的尺寸为: {width}x{height}")
print(f"图片的颜色模式为: {mode}")

# 判断图片是否为单通道或三通道
if mode == 'L':
    print("这是一张单通道（灰度）图片。")
elif mode in ['RGB', 'RGBA']:
    print("这是一张三通道或多通道（彩色）图片。")
else:
    print(f"图片的颜色模式为 {mode}，可能不是常见的单通道或三通道模式。")
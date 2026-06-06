import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def calculate_psnr(img1, img2):
    """
    计算两张图像的 PSNR。
    输入的图像应该是 HxWxC 的 NumPy 数组，范围为 [0, 255]。
    """
    # 确保图像为浮点数，范围 [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 计算 PSNR
    psnr = compare_psnr(img1, img2, data_range=255)
    return psnr


def load_image(image_path):
    """
    加载图像并转换为 RGB 格式。
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calculate_folder_psnr(folder1, folder2):
    """
    计算两个文件夹中对应图像的 PSNR。
    """
    # 获取文件夹中的所有文件名
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    if len(files1) != len(files2):
        raise ValueError("两个文件夹中的图像数量不一致！")

    total_psnr = 0.0
    count = 0

    for file1, file2 in zip(files1, files2):
        # 加载图像
        img1_path = os.path.join(folder1, file1)
        img2_path = os.path.join(folder2, file2)

        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        # 计算 PSNR
        psnr = calculate_psnr(img1, img2)
        total_psnr += psnr
        count += 1

        print(f"Image: {file1} | PSNR: {psnr:.4f} dB")

    # 计算平均 PSNR
    average_psnr = total_psnr / count
    print(f"Average PSNR: {average_psnr:.4f} dB")


# 示例用法
folder1 = "/home/wjh/Deep Learning/去模糊/ECBSR/datasets/test_AB/sharp"  # 替换为第一个文件夹路径
folder2 = "/home/wjh/Deep Learning/去模糊/ECBSR/results/experiment_name/test_latest/images"  # 替换为第二个文件夹路径
calculate_folder_psnr(folder1, folder2)
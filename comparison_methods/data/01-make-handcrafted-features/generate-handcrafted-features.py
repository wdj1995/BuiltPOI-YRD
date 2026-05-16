"""
制作遥感影像手工特征，包括：
    1. 颜色特征：Color Histogram
    2. 纹理特征：GLCM
    3. 结构特征：HOG

注意：
    RGB:
        tight crop + mask pixel statistics
    GLCM:
        tight crop + masked co-occurrence
    HOG:
        expanded bbox + natural gradient
"""

import os
import json
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog


# 1. 颜色直方图特征提取
def extract_rgb(image, mask, bins=16):
    """
        提取基于掩膜的颜色直方图特征
    """
    features = []
    # 创建布尔索引：只关注 mask > 0 的区域（即建筑物本体）
    mask_idx = mask > 0
    for ch in range(3):  # 分别处理 B, G, R 三个通道
        channel = image[:, :, ch]
        # 只取出 mask 覆盖下的像素点，展开成一维
        pixels = channel[mask_idx]
        # 计算 16 个 bin 的直方图，范围固定在 0-255
        hist, _ = np.histogram(pixels, bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        # 归一化：使特征具有尺度不变性（不受建筑像素数量影响）
        hist = hist / (hist.sum() + 1e-6)
        features.extend(hist)

    return np.array(features, dtype=np.float32)


# 2. GLCM 纹理特征提取
def extract_texture(image, mask):
    """
        提取剔除背景干扰的屏蔽灰度共生矩阵特征
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 关键技巧：将原图灰度映射到 [1, 255]，空出 0 给背景
    gray_shifted = np.round(gray * (254.0 / 255.0) + 1).astype(np.uint8)
    # 将 mask 外的区域强制设为 0
    gray_shifted[mask == 0] = 0
    # 计算 GLCM，levels=256 包含 0 级
    glcm = graycomatrix(
        gray_shifted,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],  # 四个方向：水平、45度、垂直、135度
        levels=256,
        symmetric=True,
        normed=False  # 必须先关闭归一化，因为要手动剔除第0级
    )
    # 核心操作：将第0行（包含背景）和第0列（包含背景）的统计值清零
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    # 手动对四个方向分别进行归一化，得到概率分布矩阵
    glcm = glcm.astype(np.float64)
    for i in range(glcm.shape[3]):
        s = np.sum(glcm[:, :, 0, i])
        if s > 0:
            glcm[:, :, 0, i] /= s
    # 提取四种经典的 GLCM 统计属性
    props = ["contrast", "homogeneity", "energy", "correlation"]
    features = []
    for prop in props:
        values = graycoprops(glcm, prop)  # 返回 1x4 数组
        features.extend(values.flatten())

    return np.array(features, dtype=np.float32)


# 3. HOG 结构特征提取
def extract_structure(image):
    """
        提取方向梯度直方图（HOG）特征
    """
    # 转灰度并缩放到统一尺寸，因为 HOG 的特征维度取决于输入图像大小
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # 计算 HOG 特征
    feature = hog(
        gray,
        orientations=9,          # 梯度方向分为 9 个 bin
        pixels_per_cell=(8, 8),  # 每个胞元 8x8 像素
        cells_per_block=(2, 2),  # 每个块由 2x2 个胞元组成
        block_norm='L2-Hys',     # 使用 L2-Hys 归一化抗光照变化
        feature_vector=True      # 展平为一维向量
    )
    return feature.astype(np.float32)


# 4. 辅助函数：裁剪逻辑
def crop_tight_region(image, mask):
    """提取紧凑的最小外接矩形区域"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None, None
    # 计算边界坐标
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]

def crop_hog_region(image, mask, expand_ratio=0.15):
    """提取带有背景上下文的扩张矩形区域"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w, h = x_max - x_min, y_max - y_min
    # 向四周扩张 15% 的长度
    expand_w, expand_h = int(w * expand_ratio), int(h * expand_ratio)
    x_min = max(0, x_min - expand_w)
    y_min = max(0, y_min - expand_h)
    x_max = min(image.shape[1], x_max + expand_w)
    y_max = min(image.shape[0], y_max + expand_h)
    return image[y_min:y_max, x_min:x_max]


def make_handcrafted_features():
    # 配置路径
    data_info_test_file = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\train.jsonl"
    rs_files_folder     = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\dataset\JL_Images"
    mask_files_folder   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\dataset\building_labels"

    # 输出文件夹路径
    output_folder       = r"G:\Python_Project_02\paper-03-comparison-methods\data\01-make-handcrafted-features\data\train"
    os.makedirs(output_folder, exist_ok=True)

    # 建立不同特征的子文件夹
    rgb_folder      = os.path.join(output_folder, "rgb")
    texture_folder  = os.path.join(output_folder, "texture")
    hog_folder      = os.path.join(output_folder, "hog")

    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(texture_folder, exist_ok=True)
    os.makedirs(hog_folder, exist_ok=True)

    # file_name 遥感影像文件名称
    # poi_prompt 文本描述
    # 逐行读取 jsonl 配置文件（里面包含数据对的元信息）
    with open(data_info_test_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            # 将 JSON 字符串解析为字典
            data = json.loads(line.strip())
            # 提取文件名
            file_name = data["file_name"]
            print(f"\n[{line_idx}] Processing: {file_name}")

            # 拼接完整的遥感影像路径
            rs_image_path = os.path.join(rs_files_folder, file_name)

            # 去掉.tif后缀
            file_stem = os.path.splitext(file_name)[0]
            # 拼接png mask文件名
            mask_name = file_stem + ".png"
            # mask 文件路径
            mask_path     = os.path.join(mask_files_folder, mask_name)

            # 预检：如果图像或Mask文件在磁盘上不存在，则跳过
            if not os.path.exists(rs_image_path):
                print(f"[Warning] 遥感影像不存在: {rs_image_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"[Warning] Mask文件不存在: {mask_path}")
                continue

            # 使用 OpenCV 读取遥感图像 (返回BGR矩阵)
            rs_image = cv2.imread(rs_image_path)
            # 使用单通道灰度模式读取 Mask 文件
            mask     = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 后检：防止图像损坏导致的读取为 None
            if rs_image is None:
                print(f"[Error] 无法读取遥感影像: {rs_image_path}")
                continue
            if mask is None:
                print(f"[Error] 无法读取Mask: {mask_path}")
                continue

            # mask二值化 # mask中非0区域视为建筑区域
            binary_mask = (mask > 0).astype(np.uint8)

            # --- 特征提取核心流程 ---
            # 1. 紧凑裁剪用于 RGB 和纹理
            tight_img, tight_mask = crop_tight_region(rs_image, binary_mask)
            if tight_img is None: continue

            # 2. 扩张裁剪用于 HOG
            hog_crop = crop_hog_region(rs_image, binary_mask)

            # 3. 分别调用函数计算特征
            rgb_feat = extract_rgb(tight_img, tight_mask)
            tex_feat = extract_texture(tight_img, tight_mask)
            hog_feat = extract_structure(hog_crop)

            # 保存为 .npy 格式
            np.save(os.path.join(output_folder, "rgb", file_stem + ".npy"), rgb_feat)
            np.save(os.path.join(output_folder, "texture", file_stem + ".npy"), tex_feat)
            np.save(os.path.join(output_folder, "hog", file_stem + ".npy"), hog_feat)

            print(f"Success: {file_name} | RGB: {rgb_feat.shape} | Tex: {tex_feat.shape} | HOG: {hog_feat.shape}")



if __name__ == '__main__':
    make_handcrafted_features() # 制作与视觉有关的手工特征，包括颜色、纹理和结构等特征
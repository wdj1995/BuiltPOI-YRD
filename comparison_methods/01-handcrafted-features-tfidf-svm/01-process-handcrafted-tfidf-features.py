"""
    对各类特征进行降维操作，包括与视觉有关的手工特征和tf-idf生成的高维特征
    构建训练集、验证集和测试集并保存
"""

import os
import json
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD


# 定义函数：读取 jsonl 文件中的样本信息
def load_jsonl_samples(jsonl_path):
    """从 JSONL 格式文件中解析样本元数据"""
    sample_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip()) # 将每一行 JSON 字符串解析为 Python 字典
            sample_list.append(             # 图像ID  # 功能类别标签
                {"image_id": data["image_id"], "fun_cls" : data["fun_cls"]}
            )
    # 返回所有样本列表
    return sample_list


# 定义函数：读取单个特征文件
def load_single_feature(feature_path):
    """读取并预处理单个 numpy 格式的特征向量"""
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"特征文件不存在: {feature_path}")
    feature = np.load(feature_path).flatten() # 读取 numpy 特征文件，并展平成一维向量
    return feature.astype(np.float32)         # 强制转换为单精度浮点数以节省内存，并返回结果


# 定义函数：构建完整数据集
def build_dataset(sample_list, rgb_folder, texture_folder, hog_folder, tfidf_folder):
    """多模态特征聚合函数：将零散的特征文件组合成矩阵"""
    # 用字典保存不同模态的特征
    features = {"rgb": [], "texture": [], "hog": [], "tfidf": []}
    y = [] # 用于保存类别标签
    # 迭代样本元数据，根据图像 ID 检索对应的物理特征文件
    for idx, sample in enumerate(sample_list):
        img_id = sample["image_id"] # 获取当前样本图像 ID
        # 读取 RGB 特征并加入列表
        features["rgb"].append(load_single_feature(os.path.join(rgb_folder, img_id + ".npy")))
        # 读取纹理特征并加入列表
        features["texture"].append(load_single_feature(os.path.join(texture_folder, img_id + ".npy")))
        # 读取 HOG 特征并加入列表
        features["hog"].append(load_single_feature(os.path.join(hog_folder, img_id + ".npy")))
        # 读取 TF-IDF 特征并加入列表
        features["tfidf"].append(load_single_feature(os.path.join(tfidf_folder, img_id + ".npy")))
        # 同步记录对应的分类真值
        y.append(sample["fun_cls"])
    # 将 List 容器转换为高性能的 Numpy Ndarray 矩阵 (N_samples, D_features)
    for k in features:
        features[k] = np.array(features[k], dtype=np.float32)
    # 返回特征矩阵字典和对应的标签列表
    return features, y


# --- 处理与降维 ---
def process_and_save_data():
    # 路径配置
    # 训练集 jsonl 文件路径
    train_jsonl      = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\train.jsonl"
    # 验证集 jsonl 文件路径
    val_jsonl        = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\val.jsonl"
    # 测试集 jsonl 文件路径
    test_jsonl       = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\test.jsonl"

    # 手工特征根目录
    handcrafted_root = r"G:\Python_Project_02\paper-03-comparison-methods\data\01-make-handcrafted-features\data"
    # TF-IDF 特征根目录
    tfidf_root       = r"G:\Python_Project_02\paper-03-comparison-methods\data\02-poi-TF-IDF-features\data"
    # 处理后数据的存储路径
    output_path      = r"G:\Python_Project_02\paper-03-comparison-methods\01-handcrafted-features-tfidf-svm\process_data"
    os.makedirs(output_path, exist_ok=True) # 若输出目录不存在，则自动创建

    # 1. 构建原始特征矩阵
    print("正在构建原始数据集...")
    # 调用 build_dataset 递归载入训练集各模态数据
    train_feat_raw, y_train_raw = build_dataset(
        load_jsonl_samples(train_jsonl),
        os.path.join(handcrafted_root, "train", "rgb"),     # RGB 特征路径
        os.path.join(handcrafted_root, "train", "texture"), # Texture 特征路径
        os.path.join(handcrafted_root, "train", "hog"),     # HOG 特征路径
        os.path.join(tfidf_root, "train")                   # TF-IDF 特征路径
    )
    # 载入验证集（用于模型调参）
    val_feat_raw, y_val_raw = build_dataset(
        load_jsonl_samples(val_jsonl),
        os.path.join(handcrafted_root, "val", "rgb"),
        os.path.join(handcrafted_root, "val", "texture"),
        os.path.join(handcrafted_root, "val", "hog"),
        os.path.join(tfidf_root, "val")
    )
    # 载入测试集（用于最终性能评估）
    test_feat_raw, y_test_raw = build_dataset(
        load_jsonl_samples(test_jsonl),
        os.path.join(handcrafted_root, "test", "rgb"),
        os.path.join(handcrafted_root, "test", "texture"),
        os.path.join(handcrafted_root, "test", "hog"),
        os.path.join(tfidf_root, "test")
    )


    # 2. 标签预处理
    le      = LabelEncoder()                # 初始化标签编码器，将类别名转换为 0,1,2... 整数
    y_train = le.fit_transform(y_train_raw) # 学习训练集分布并进行编码
    y_val   = le.transform(y_val_raw)       # 沿用训练集编码映射处理验证集
    y_test  = le.transform(y_test_raw)      # 沿用训练集编码映射处理测试集
    joblib.dump(le, os.path.join(output_path, "label_encoder.pkl")) # 将编码器保存，以便预测时将数字 ID 反转回原始类别名


    # 3. 模态处理函数（内部封装 PCA 和 Normalize）
    def process_normal_modal(train, val, test, name):
        """通用模态处理：标准化 + L2 范数归一化"""
        scaler = StandardScaler() # 消除特征间量纲差异，使其符合正态分布 (mean=0, std=1)
        train  = scaler.fit_transform(train).astype(np.float32) # 计算均值/方差并转换训练集
        val    = scaler.transform(val).astype(np.float32)       # 应用训练集参数转换验证集
        test   = scaler.transform(test).astype(np.float32)      # 应用训练集参数转换测试集
        # 保存标准化参数，确保推理阶段一致性
        joblib.dump(scaler,os.path.join(output_path, f"{name}_scaler.pkl"))
        # 执行 L2 归一化，将样本向量长度缩放到单位圆内，增强相似度度量（如 SVM/余弦相似度）性能
        train = normalize(train, norm='l2')
        val   = normalize(val, norm='l2')
        test  = normalize(test, norm='l2')
        return train, val, test

    def process_hog_modal(train, val, test, dim=256):
        """HOG 模态处理：标准化 + PCA 线性降维"""
        print("\n处理 HOG 特征...")
        scaler = StandardScaler()  # 预处理：PCA 前通常需要中心化
        train  = scaler.fit_transform(train)
        val    = scaler.transform(val)
        test   = scaler.transform(test)
        joblib.dump(scaler, os.path.join(output_path, "hog_scaler.pkl"))

        # 使用主成分分析将高维 HOG 特征降维至目标维度，减少冗余并抑制过拟合
        pca   = PCA(n_components=dim, random_state=42)
        train = pca.fit_transform(train)
        val   = pca.transform(val)
        test  = pca.transform(test)

        joblib.dump(pca, os.path.join(output_path, "hog_pca.pkl"))  # 持久化 PCA 模型
        # 计算并打印降维后保留的累计方差贡献率，评估信息损失
        print(f"HOG PCA 保留信息比例: {pca.explained_variance_ratio_.sum():.4f}")

        # 再次进行 L2 归一化
        train = normalize(train, norm='l2')
        val   = normalize(val, norm='l2')
        test  = normalize(test, norm='l2')
        return train, val, test

    def process_tfidf_modal(train, val, test, dim=256):
        """TF-IDF 模态处理：TruncatedSVD 降维"""
        print("\n处理 TF-IDF 特征...")
        # 采用截断奇异值分解（LSA），专门用于处理稀疏的词频矩阵
        svd   = TruncatedSVD(n_components=dim, random_state=42)
        train = svd.fit_transform(train).astype(np.float32)
        val   = svd.transform(val)
        test  = svd.transform(test)
        joblib.dump(svd, os.path.join(output_path, "tfidf_svd.pkl"))
        print(f"TF-IDF SVD 保留信息比例: {svd.explained_variance_ratio_.sum():.4f}")
        # 最终归一化，增强分类器的鲁棒性
        train = normalize(train, norm='l2')
        val   = normalize(val, norm='l2')
        test  = normalize(test, norm='l2')
        return train, val, test


    # 4. 执行各模态处理
    # 处理颜色特征（RGB 直方图/矩）
    rgb_tr, rgb_va, rgb_te = process_normal_modal(
        train_feat_raw["rgb"], val_feat_raw["rgb"], test_feat_raw["rgb"], "rgb")

    # 处理纹理特征（LBP/GLCM 等）
    tex_tr, tex_va, tex_te = process_normal_modal(
        train_feat_raw["texture"], val_feat_raw["texture"], test_feat_raw["texture"], "texture")

    # 处理形状/边缘特征（HOG）并降维至 256 维
    hog_tr, hog_va, hog_te = process_hog_modal(
        train_feat_raw["hog"], val_feat_raw["hog"], test_feat_raw["hog"], dim=256)

    # 处理 POI 信息特征（TF-IDF）并降维至 256 维
    tfi_tr, tfi_va, tfi_te = process_tfidf_modal(
        train_feat_raw["tfidf"], val_feat_raw["tfidf"], test_feat_raw["tfidf"], dim=256)


    # 5. 特征融合（Feature-level Fusion）
    def fuse(r, t, h, f):
        """早期融合策略：横向拼接各模态特征向量"""
        # 在列维度上进行串联，各模态权重目前设为 1.0（等权融合）
        return np.concatenate([r * 1.0, t * 1.0, h * 1.0, f * 1.0], axis=1).astype(np.float32)

    # 构建最终训练、验证、测试特征
    X_train, X_val, X_test = (
        fuse(rgb_tr, tex_tr, hog_tr, tfi_tr), # 训练集融合
        fuse(rgb_va, tex_va, hog_va, tfi_va), # 验证集融合
        fuse(rgb_te, tex_te, hog_te, tfi_te)  # 测试集融合
    )

    # 6. 数据序列化保存
    # 使用 savez 压缩格式保存多个数组，方便后续直接加载使用
    np.savez(os.path.join(output_path, "final_dataset.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val,
             X_test=X_test,   y_test=y_test
             )
    # 完成信号
    print(f"数据处理完成并保存至: {output_path}")


if __name__ == "__main__":
    process_and_save_data()




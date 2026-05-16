"""
    对视觉手工特征 + Word2Vec 文本特征进行预处理与融合
    构建训练集、验证集和测试集并保存
"""

import os  # 导入操作系统接口模块，用于处理文件路径和目录创建
import json  # 导入JSON模块，用于解析jsonl格式的元数据文件
import numpy as np  # 导入NumPy库，用于高效的数值计算和矩阵操作
import joblib  # 导入joblib库，用于序列化保存和加载模型（如Scaler, PCA）

# 从sklearn预处理模块导入标签编码、归一化和标准化工具
from sklearn.preprocessing import (
    LabelEncoder,  # 将类别标签转换为整数数值
    normalize,     # 执行向量归一化（如L2范数）
    StandardScaler # 执行标准化，使数据符合均值为0、方差为1的分布
)

from sklearn.decomposition import PCA  # 从sklearn导入主成分分析工具，用于特征降维


# 读取 jsonl 文件中的样本信息
def load_jsonl_samples(jsonl_path):
    """从 JSONL 文件中读取 image_id 与类别标签"""
    sample_list = []  # 初始化空列表，用于存储每个样本的元数据
    with open(jsonl_path, "r", encoding="utf-8") as f:  # 以只读和UTF-8编码打开jsonl文件
        for line in f:  # 逐行遍历文件内容
            data = json.loads(line.strip())  # 将当前行字符串解析为Python字典并去除两端空格
            sample_list.append({  # 将解析出的关键信息打包成字典存入列表
                "image_id": data["image_id"],  # 提取图像的唯一标识符
                "fun_cls" : data["fun_cls"]     # 提取图像所属的功能类别标签
            })
    return sample_list  # 返回包含所有样本信息的列表



# 读取单个特征文件
def load_single_feature(feature_path):
    """读取单个 numpy 特征文件"""
    if not os.path.exists(feature_path):  # 检查指定的特征文件路径是否存在
        raise FileNotFoundError(f"特征文件不存在: {feature_path}")  # 若不存在则抛出异常
    # 加载 .npy 文件，并使用 flatten 确保其被展平为一维向量（1D Vector）
    feature = np.load(feature_path).flatten()
    return feature.astype(np.float32)  # 转换为单精度浮点数以降低内存消耗并返回



# 构建完整数据集
def build_dataset(sample_list, rgb_folder, texture_folder, hog_folder, word2vec_folder):
    """根据 image_id 构建多模态特征数据集"""

    # 初始化字典，用于分类存储不同模态的原始特征矩阵
    features = {
        "rgb": [],       # 存储RGB颜色特征向量
        "texture": [],   # 存储纹理特征向量
        "hog": [],       # 存储HOG梯度特征向量
        "word2vec": []   # 存储Word2Vec文本语义向量
    }

    y = []  # 初始化列表，用于按序存储类别标签

    # 遍历样本元数据列表
    for idx, sample in enumerate(sample_list):
        img_id = sample["image_id"]  # 获取当前样本的图像ID
        # RGB 特征：根据 ID 拼接路径并读取
        rgb_feature = load_single_feature(os.path.join(rgb_folder, img_id + ".npy"))
        features["rgb"].append(rgb_feature)  # 将读取到的特征向量加入rgb列表中

        # Texture 特征：同理读取纹理特征
        texture_feature = load_single_feature(os.path.join(texture_folder, img_id + ".npy"))
        features["texture"].append(texture_feature)

        # HOG 特征：同理读取形状特征
        hog_feature = load_single_feature(os.path.join(hog_folder, img_id + ".npy"))
        features["hog"].append(hog_feature)

        # Word2Vec 特征：同理读取对应的文本向量
        word2vec_feature = load_single_feature(os.path.join(word2vec_folder, img_id + ".npy"))
        features["word2vec"].append(word2vec_feature)

        # 标签：同步记录该样本的真实类别
        y.append(sample["fun_cls"])

    # 循环遍历特征字典，将所有 List 转换为 Numpy 矩阵 (N_samples, Dim)
    for k in features:
        features[k] = np.array(features[k], dtype=np.float32)
    return features, y  # 返回特征字典和标签列表


# 主处理函数
def process_and_save_data():

    # 1. 路径配置：定义输入数据路径和输出保存路径
    train_jsonl = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\train.jsonl"
    val_jsonl   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\val.jsonl"
    test_jsonl  = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\test.jsonl"

    # 视觉手工提取特征根目录路径
    handcrafted_root = r"G:\Python_Project_02\paper-03-comparison-methods\data\01-make-handcrafted-features\data"
    # Word2Vec 特征根目录路径
    word2vec_root    = r"G:\Python_Project_02\paper-03-comparison-methods\02-handcrafted-features-word2vec-svm\process_data\01-make-word2vec-features"

    # 最终处理后数据集的输出路径
    output_path      = "./process_data/02-make-train-val-test-dataset"
    os.makedirs(output_path, exist_ok=True)  # 若输出目录不存在，则递归创建该目录


    # 2. 构建原始数据集：将物理文件加载到内存矩阵中
    print("正在构建原始多模态数据集...")
    # 构建训练集
    train_feat_raw, y_train_raw = build_dataset(
        load_jsonl_samples(train_jsonl),
        os.path.join(handcrafted_root, "train", "rgb"),
        os.path.join(handcrafted_root, "train", "texture"),
        os.path.join(handcrafted_root, "train", "hog"),
        os.path.join(word2vec_root,    "train")
    )
    # 构建验证集
    val_feat_raw, y_val_raw = build_dataset(
        load_jsonl_samples(val_jsonl),
        os.path.join(handcrafted_root, "val", "rgb"),
        os.path.join(handcrafted_root, "val", "texture"),
        os.path.join(handcrafted_root, "val", "hog"),
        os.path.join(word2vec_root,    "val")
    )
    # 构建测试集
    test_feat_raw, y_test_raw = build_dataset(
        load_jsonl_samples(test_jsonl),
        os.path.join(handcrafted_root, "test", "rgb"),
        os.path.join(handcrafted_root, "test", "texture"),
        os.path.join(handcrafted_root, "test", "hog"),
        os.path.join(word2vec_root,    "test")
    )


    # 3. 标签编码：将字符串类名转换为数字 ID
    print("\n正在进行标签编码...")
    le      = LabelEncoder()                # 实例化标签编码器
    y_train = le.fit_transform(y_train_raw) # 训练集 fit + transform，学习类别映射
    y_val   = le.transform(y_val_raw)       # 验证集复用训练集的映射关系
    y_test  = le.transform(y_test_raw)      # 测试集复用训练集的映射关系
    joblib.dump(le, os.path.join(output_path, "label_encoder.pkl"))  # 将编码器模型保存到本地，用于推理时反向解析
    print("标签编码完成")


    # 4. 通用模态处理函数：内部封装了标准化和归一化逻辑
    def process_normal_modal(train, val, test, name):
        """普通模态处理：StandardScaler + L2 Normalize"""
        print(f"\n处理 {name} 特征...")
        scaler = StandardScaler()  # 实例化标准化工具（均值0，方差1）
        train  = scaler.fit_transform(train).astype(np.float32)  # 计算训练集参数并转换
        val    = scaler.transform(val).astype(np.float32)        # 使用训练集均值方差转换验证集
        test   = scaler.transform(test).astype(np.float32)      # 使用训练集均值方差转换测试集
        joblib.dump(scaler, os.path.join(output_path, f"{name}_scaler.pkl"))  # 保存标准化模型
        # 执行 L2 归一化，将每个样本向量转化为单位长度，常用于距离度量算法（如SVM）
        train  = normalize(train, norm='l2')
        val    = normalize(val, norm='l2')
        test   = normalize(test, norm='l2')
        print(f"{name} 处理完成")
        return train, val, test


    # 5. HOG 特征处理：由于维数较高，特别增加了 PCA 降维环节
    def process_hog_modal(train, val, test, dim=256):
        """HOG: StandardScaler + PCA + Normalize"""
        print("\n处理 HOG 特征...")
        scaler = StandardScaler()  # PCA 前通常需要先进行标准化处理
        train  = scaler.fit_transform(train)
        val    = scaler.transform(val)
        test   = scaler.transform(test)
        joblib.dump(scaler, os.path.join(output_path, "hog_scaler.pkl"))  # 保存 HOG 的标准化参数

        # 实例化 PCA，指定降维到 256 维，并设置随机种子保证实验可重复性
        pca    = PCA(n_components=dim, random_state=42)
        train  = pca.fit_transform(train)  # 学习主成分并降维
        val    = pca.transform(val)          # 应用相同主成分到验证集
        test   = pca.transform(test)        # 应用相同主成分到测试集
        joblib.dump(pca, os.path.join(output_path, "hog_pca.pkl"))  # 保存 PCA 降维模型
        print(f"HOG PCA 保留信息比例: {pca.explained_variance_ratio_.sum():.4f}")

        # 降维后再次进行归一化处理
        train = normalize(train, norm='l2')
        val = normalize(val, norm='l2')
        test = normalize(test, norm='l2')
        return train, val, test


    # 6. Word2Vec 特征处理：流程与普通模态一致
    def process_word2vec_modal(train, val, test):
        """Word2Vec: StandardScaler + Normalize"""
        print("\n处理 Word2Vec 特征...")
        scaler = StandardScaler()
        train  = scaler.fit_transform(train).astype(np.float32)
        val    = scaler.transform(val).astype(np.float32)
        test   = scaler.transform(test).astype(np.float32)
        joblib.dump(scaler, os.path.join(output_path, "word2vec_scaler.pkl")) # 保存词向量标准化器
        train  = normalize(train, norm='l2')
        val    = normalize(val, norm='l2')
        test   = normalize(test, norm='l2')
        print("Word2Vec 处理完成")
        return train, val, test


    # 7. 执行各模态处理：依次调用上述定义的嵌套函数
    # 处理颜色模态
    rgb_tr, rgb_va, rgb_te = process_normal_modal(train_feat_raw["rgb"], val_feat_raw["rgb"], test_feat_raw["rgb"], "rgb")
    # 处理纹理模态
    tex_tr, tex_va, tex_te = process_normal_modal(train_feat_raw["texture"], val_feat_raw["texture"], test_feat_raw["texture"], "texture")
    # 处理 HOG 模态（降维至 256）
    hog_tr, hog_va, hog_te = process_hog_modal(train_feat_raw["hog"], val_feat_raw["hog"], test_feat_raw["hog"], dim=256)
    # 处理文本语义模态 (Word2Vec)
    w2v_tr, w2v_va, w2v_te = process_word2vec_modal(train_feat_raw["word2vec"], val_feat_raw["word2vec"], test_feat_raw["word2vec"])


    # 8. 特征融合：定义早期融合策略（Early Fusion）
    def fuse(r, t, h, w):
        """将处理后的各模态向量在水平维度上横向拼接（等权融合）"""
        return np.concatenate([r * 1.0, t * 1.0, h * 1.0, w * 1.0], axis=1).astype(np.float32)

    # 分别构建训练、验证、测试集的融合特征矩阵
    X_train = fuse(rgb_tr, tex_tr, hog_tr, w2v_tr)
    X_val   = fuse(rgb_va, tex_va, hog_va, w2v_va)
    X_test  = fuse(rgb_te, tex_te, hog_te, w2v_te)


    # 9. 保存最终数据集：使用 npz 压缩格式存储
    print("\n正在保存最终数据集...")
    np.savez(
        os.path.join(output_path, "final_dataset.npz"),
        X_train=X_train, y_train=y_train,  # 保存训练集数据和标签
        X_val=X_val,     y_val=y_val,      # 保存验证集数据和标签
        X_test=X_test,   y_test=y_test     # 保存测试集数据和标签
    )

    print("\n数据处理完成")
    print(f"保存路径: {output_path}")


    # 10. 打印最终数据集信息：确认数据形状
    print("\n最终数据集信息:")
    print(f"X_train shape: {X_train.shape}")  # 打印训练集维度
    print(f"X_val   shape: {X_val.shape}")    # 打印验证集维度
    print(f"X_test  shape: {X_test.shape}")   # 打印测试集维度
    print(f"y_train shape: {y_train.shape}")  # 打印训练标签维度
    print(f"y_val   shape: {y_val.shape}")    # 打印验证标签维度
    print(f"y_test  shape: {y_test.shape}")   # 打印测试标签维度


# 程序入口：当脚本被直接运行时执行主处理流程
if __name__ == "__main__":
    process_and_save_data()
"""
    基于TF_IDF方法生成针对poi文本描述的高维特征
"""

import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_jsonl_data(jsonl_file):
    """
        读取jsonl文件中的poi_prompt文本和对应的图像编号（作为重命名的名称）
    """
    # 初始化数据列表
    data_list = []
    # 打开jsonl文件
    with open(jsonl_file, "r", encoding="utf-8") as f:
        # 遍历每一行
        for line_idx, line in enumerate(f):
            # 将json字符串解析为字典
            data = json.loads(line.strip())
            # 获取image_id
            image_id = data["image_id"]
            # 获取poi_prompt
            poi_prompt = data.get("poi_prompt")

            # 异常处理：空文本
            if poi_prompt is None:
                poi_prompt = ""

            # 强制转字符串
            poi_prompt = str(poi_prompt)
            # 去除前后空格
            poi_prompt = poi_prompt.strip()
            # 保存
            data_list.append({
                "image_id"   : image_id,
                "poi_prompt" : poi_prompt
            })
    return data_list


def save_single_feature(feature_vector, image_id, save_folder):
    """
        保存单个样本的TF-IDF特征
    """
    # 去除扩展名
    image_name = os.path.splitext(image_id)[0]
    # 构造保存路径
    save_path  = os.path.join(save_folder, f"{image_name}.npy")
    # 保存numpy特征
    np.save(save_path, feature_vector)


def process_and_save_split(data_list, tfidf_vectorizer, split_save_folder, split_name):
    """
        处理某个数据集并保存每个样本的特征
    """
    print(f"\n开始处理 {split_name} 数据集...")
    # 创建保存目录
    os.makedirs(split_save_folder, exist_ok=True)
    # 遍历每个样本
    for idx, item in enumerate(data_list):
        # 获取image_id
        image_id   = item["image_id"]
        # 获取文本
        poi_prompt = item["poi_prompt"]
        # transform要求输入是list
        text_list = [poi_prompt]
        # 转换TF-IDF特征
        tfidf_feature = tfidf_vectorizer.transform(text_list)
        # 稀疏矩阵 -> 稠密向量
        tfidf_feature = tfidf_feature.toarray()[0]
        # 保存单个特征
        save_single_feature(
            feature_vector = tfidf_feature,
            image_id       = image_id,
            save_folder    = split_save_folder
        )
        # 打印进度
        if (idx + 1) % 100 == 0:
            print(f"{split_name}: 已处理 {idx + 1}/{len(data_list)}")
    print(f"{split_name} 数据集处理完成！")


def train_tfidf_model(train_data_list, output_folder):
    """
        使用训练集训练TF-IDF模型
    """
    print("\n开始训练TF-IDF模型...")
    # 提取所有训练文本
    train_corpus = [item["poi_prompt"] for item in train_data_list]
    # 定义TF-IDF模型
    tfidf_vectorizer = TfidfVectorizer(
        lowercase    = True,       # 转小写
        stop_words   = "english",  # 去停用词
        max_features = 5000,       # 最大词汇量
        ngram_range  = (1, 2)      # unigram + bigram
    )
    # 训练模型
    tfidf_vectorizer.fit(train_corpus)
    print("TF-IDF模型训练完成！")

    # 保存词表
    vocabulary      = tfidf_vectorizer.get_feature_names_out()
    vocab_save_path = os.path.join(output_folder, "tfidf_vocabulary.txt")
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        for word in vocabulary:
            f.write(word + "\n")
    print(f"词表已保存: {vocab_save_path}")

    # 保存模型
    vectorizer_save_path = os.path.join(output_folder, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf_vectorizer, vectorizer_save_path)
    print(f"TF-IDF模型已保存: {vectorizer_save_path}")
    return tfidf_vectorizer


def make_poi_tfidf_features():
    """
        主函数
    :return:
    """
    # 定义输入文件的相对路径
    train_jsonl = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\train.jsonl"
    val_jsonl   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\val.jsonl"
    test_jsonl  = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\test.jsonl"

    # 创建输出目录，exist_ok=True表示文件夹已存在时不报错
    output_folder  = r"G:\Python_Project_02\paper-03-comparison-methods\data\02-poi-TF-IDF-features\data\tfidf_features"
    # 创建总输出目录
    os.makedirs(output_folder, exist_ok=True)

    # train特征目录
    train_save_folder = os.path.join(output_folder, "train")
    # val特征目录
    val_save_folder   = os.path.join(output_folder, "val")
    # test特征目录
    test_save_folder  = os.path.join(output_folder, "test")

    # Step 1: 加载数据（获取原始的poi描述文本与对应的图像编号）
    print("\n开始读取数据...")
    train_data_list = load_jsonl_data(train_jsonl)
    val_data_list   = load_jsonl_data(val_jsonl)
    test_data_list  = load_jsonl_data(test_jsonl)
    print(f"Train数量 : {len(train_data_list)}")
    print(f"Val数量   : {len(val_data_list)}")
    print(f"Test数量  : {len(test_data_list)}")


    # Step 2: 训练TF-IDF模型
    tfidf_vectorizer    = train_tfidf_model(
        train_data_list = train_data_list,
        output_folder   = output_folder
    )

    # Step 3: 处理Train
    process_and_save_split(
        data_list         = train_data_list,
        tfidf_vectorizer  = tfidf_vectorizer,
        split_save_folder = train_save_folder,
        split_name        = "train"
    )

    # Step 4: 处理Val
    process_and_save_split(
        data_list         = val_data_list,
        tfidf_vectorizer  = tfidf_vectorizer,
        split_save_folder = val_save_folder,
        split_name        = "val"
    )

    # Step 5: 处理Test
    process_and_save_split(
        data_list         = test_data_list,
        tfidf_vectorizer  = tfidf_vectorizer,
        split_save_folder = test_save_folder,
        split_name        = "test"
    )

    print("\n===================================")
    print("全部TF-IDF特征提取完成！")
    print("===================================")



if __name__ == "__main__":
    make_poi_tfidf_features() # 基于TF_IDF方法生成针对poi文本描述的高维特征
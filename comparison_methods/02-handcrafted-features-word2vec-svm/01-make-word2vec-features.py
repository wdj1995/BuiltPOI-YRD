"""
    基于word2vec方法生成poi文本语义有关的高维特征
"""

import os
import json
import joblib
import numpy as np
from gensim.models import Word2Vec
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess


def load_jsonl_data(jsonl_file):
    """读取jsonl文件中的poi_prompt文本和对应的图像编号"""
    data_list = [] # 初始化空列表，用于存储提取的数据字典
    # 以只读模式和utf-8编码打开jsonl文件
    with open(jsonl_file, "r", encoding="utf-8") as f:
        # 遍历文件的每一行，line_idx为行索引
        for line in f:
            data = json.loads(line)
            data_list.append({
                "image_id"   : data["image_id"],
                "poi_prompt" : str(data.get("poi_prompt", "")).strip()
            })
    return data_list # 返回处理后的数据列表


def tokenize_text(text):
    """文本分词：Word2Vec必须输入token list"""
    # return text.lower().strip().split()
    return simple_preprocess(text)


def save_single_feature(feature_vector, image_id, save_folder):
    """保存单个样本的特征向量为.npy文件"""
    # 拼接保存路径：文件夹路径 + 文件名 + .npy后缀
    save_path  = os.path.join(save_folder, f"{image_id}.npy")
    # 使用numpy将特征向量保存到本地
    np.save(save_path, feature_vector)


def sentence_to_vector(sentence, keyed_vectors):
    """
        使用预训练向量将英文句子转换为均值向量
        注意：keyed_vectors 是通过 api.load 加载的对象
    """
    # 1. 对原始句子进行分词
    tokens  = tokenize_text(sentence)
    # 2. 存储该句子中在词表里的有效词向量
    vectors = []

    # 遍历分词结果
    for word in tokens:
        # 修改：KeyedVectors 对象直接支持 'in' 判断和索引访问
        if word in keyed_vectors:
            vectors.append(keyed_vectors[word])
    # 如果整句话中没有一个词在词表里（或者句子为空）
    if len(vectors) == 0:
        # 返回对应维度的零向量（google-news模型通常是300维）
        return np.zeros(keyed_vectors.vector_size)
    return np.mean(vectors, axis=0) # 将多个词向量聚合成一个固定长度的句子向量


def process_and_save_split(data_list, keyed_vectors, split_save_folder, split_name):
    """推理并保存特征"""
    print(f"\n开始处理 {split_name} 数据集...")
    # 如果保存目录不存在，则递归创建目录
    os.makedirs(split_save_folder, exist_ok=True)

    # 遍历数据集中的每一个条目
    for idx, item in enumerate(data_list):
        image_id   = item["image_id"]
        poi_prompt = item["poi_prompt"]

        # 直接推理获取向量
        feature = sentence_to_vector(poi_prompt, keyed_vectors)

        # 调用函数：将生成的特征向量保存到磁盘
        save_single_feature(
            feature_vector = feature,
            image_id       = image_id,
            save_folder    = split_save_folder
        )
        # 每处理100张图片，在控制台打印一次进度
        if (idx + 1) % 100 == 0:
            print(f"{split_name}: 已处理 {idx + 1}/{len(data_list)}")

    print(f"{split_name} 数据集处理完成！")


def make_poi_word2vec_features():
    """主函数：编排整个提取特征的流水线"""

    # 1. 路径设置
    train_jsonl   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\train.jsonl"
    val_jsonl     = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\val.jsonl"
    test_jsonl    = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\test.jsonl"

    # 定义 特征向量 输出的总根目录
    output_folder = "./process_data/01-make-word2vec-features"
    # 创建输出根目录
    os.makedirs(output_folder, exist_ok=True)

    # 2. 加载预训练英文模型 (首次运行会从云端下载，约1.6GB)
    # print("\n正在加载预训练英文模型: word2vec-google-news-300...")
    # # 注意：api.load 返回的是 KeyedVectors 对象
    # keyed_vectors = api.load('word2vec-google-news-300')
    # print("模型加载成功！")
    print("\n正在加载本地Word2Vec模型...")
    model_path    = "./download_models/word2vec-google-news-300.model"
    keyed_vectors = KeyedVectors.load(model_path)
    print("本地模型加载成功！")

    # 3. 读取数据
    print("\n读取数据...")
    train_data = load_jsonl_data(train_jsonl)
    val_data   = load_jsonl_data(val_jsonl)
    test_data  = load_jsonl_data(test_jsonl)

    # 4. 执行特征提取流程
    splits = [
        (train_data, "train"),
        (val_data,   "val"),
        (test_data,  "test")
    ]

    for data, name in splits:
        if data:
            save_path = os.path.join(output_folder, name)
            process_and_save_split(data, keyed_vectors, save_path, name)

    print("\n===================================")
    print("全部英文特征提取完成！")
    print("===================================")




if __name__ == "__main__":

    make_poi_word2vec_features()
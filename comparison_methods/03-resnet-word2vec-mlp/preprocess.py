import argparse
import json
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess


def load_jsonl_data(jsonl_file: str) -> list[dict]:
    """读取一个 split 的 jsonl，只保留生成文本特征所需的字段。"""
    data_list = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {jsonl_file}:{line_no}") from exc

            data_list.append(
                {
                    # image_id 用于和影像文件、mask 文件、输出的 .npy 特征文件对齐。
                    "image_id": data["image_id"],
                    # poi_prompt 是当前样本对应的 POI 文本描述。
                    "poi_prompt": str(data.get("poi_prompt", "")).strip(),
                }
            )
    return data_list


def tokenize_text(text: str) -> list[str]:
    """Google News Word2Vec 是英文词向量，这里按英文空格切词并转小写。"""
    # return str(text).lower().strip().split()
    return simple_preprocess(text) # 自动处理


def sentence_to_vector(sentence: str, keyed_vectors: KeyedVectors) -> np.ndarray:
    """将一句 POI 文本转换成固定长度句向量。"""
    vectors = []
    for word in tokenize_text(sentence):
        # 只使用预训练词表中存在的 token，避免 OOV 词导致索引错误。
        if word in keyed_vectors:
            vectors.append(keyed_vectors[word])

    # 如果文本为空或所有 token 都不在词表里，返回零向量保持维度一致。
    if not vectors:
        return np.zeros(keyed_vectors.vector_size, dtype=np.float32)

    # 对句子中的有效词向量做均值池化，得到单个 300 维 POI 语义特征。
    return np.mean(vectors, axis=0).astype(np.float32)


def save_single_feature(feature_vector: np.ndarray, image_id: str, save_folder: str) -> None:
    """把单个样本的文本特征保存为 image_id 对应的 .npy 文件。"""
    save_path = Path(save_folder) / f"{Path(str(image_id)).stem}.npy"
    np.save(save_path, feature_vector)


def process_and_save_split(data_list: list[dict], keyed_vectors: KeyedVectors, split_save_folder: str, split_name: str) -> None:
    """处理 train/val/test 中的一个 split，并把结果写入对应子目录。"""
    print(f"\nStart processing {split_name} split...")
    Path(split_save_folder).mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(data_list):
        # poi_prompt -> 300 维 Word2Vec 均值向量。
        feature = sentence_to_vector(item["poi_prompt"], keyed_vectors)
        # 保存为 split_save_folder/{image_id}.npy，Dataset 会按同样规则读取。
        save_single_feature(feature, item["image_id"], split_save_folder)

        if (idx + 1) % 100 == 0:
            print(f"{split_name}: processed {idx + 1}/{len(data_list)}")

    print(f"{split_name} split finished.")


def make_poi_word2vec_features(
    train_jsonl: str,
    val_jsonl: str,
    test_jsonl: str,
    word2vec_path: str,
    output_folder: str,
) -> None:
    """主流程：加载本地预训练 Word2Vec，并为三个 split 预计算文本特征。"""
    print("\nLoading local Word2Vec KeyedVectors...")
    # mmap="r" 可以减少大模型加载时的内存压力。
    keyed_vectors = KeyedVectors.load(word2vec_path, mmap="r")
    print(f"Loaded Word2Vec model from: {word2vec_path}")

    splits = [
        (load_jsonl_data(train_jsonl), "train"),
        (load_jsonl_data(val_jsonl),   "val"),
        (load_jsonl_data(test_jsonl),  "test"),
    ]

    for data, split_name in splits:
        if data:
            # 输出结构示例：output_folder/train/000001.npy。
            split_save_folder = Path(output_folder) / split_name
            process_and_save_split(data, keyed_vectors, str(split_save_folder), split_name)

    print("\nAll POI Word2Vec features have been generated.")


def main() -> None:
    parser = argparse.ArgumentParser()
    # 三个 jsonl 已经提前划分好，本脚本只读取它们，不重新划分数据集。
    parser.add_argument("--train_jsonl",   default = "./data/splits/train.jsonl")
    parser.add_argument("--val_jsonl",     default = "./data/splits/val.jsonl")
    parser.add_argument("--test_jsonl",    default = "./data/splits/test.jsonl")
    # 本地 Google News Word2Vec KeyedVectors 模型。
    parser.add_argument("--word2vec_path", default = "./download_models/word2vec-google-news-300.model")
    # 输出目录中会生成 train/val/test 三个子目录，每个样本对应一个 .npy 文件。
    parser.add_argument("--output_folder", default = "./process_data/word2vec_features")
    args = parser.parse_args()

    make_poi_word2vec_features(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        word2vec_path=args.word2vec_path,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()

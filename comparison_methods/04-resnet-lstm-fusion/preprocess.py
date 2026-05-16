import argparse
import json
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess


def load_jsonl_data(jsonl_file: str) -> list[dict]:
    """读取一个数据划分的 jsonl 文件，只保留文本特征预处理需要的字段。"""
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
                    "image_id": data["image_id"],
                    "poi_prompt": str(data.get("poi_prompt", "")).strip(),
                }
            )
    return data_list


def tokenize_text(text: str) -> list[str]:
    """对 POI 文本分词，输出可用于 Google News Word2Vec 查询的 token。"""
    return simple_preprocess(text)


def sentence_to_sequence(sentence: str, keyed_vectors: KeyedVectors, max_tokens: int = 64) -> tuple[np.ndarray, int]:
    """将一条 POI 句子转换为定长 Word2Vec token 序列矩阵。"""
    vectors = []
    for word in tokenize_text(sentence):
        # 只保留 Word2Vec 词表中存在的 token，跳过 OOV 词。
        if word in keyed_vectors:
            vectors.append(keyed_vectors[word])

        # 超过 max_tokens 的部分直接截断，保证所有样本序列长度一致。
        if len(vectors) >= max_tokens:
            break

    length = len(vectors)

    # 先创建全 0 矩阵；短句后面的 0 行就是 padding。
    feature = np.zeros((max_tokens, keyed_vectors.vector_size), dtype=np.float32)
    if vectors:
        # 将真实 token 的词向量填入矩阵前 length 行。
        feature[:length] = np.asarray(vectors, dtype=np.float32)
    return feature, length


def save_single_feature(feature_matrix: np.ndarray, length: int, image_id: str, save_folder: str) -> None:
    """保存单个样本的定长 token 矩阵和真实有效长度。"""
    save_path = Path(save_folder) / f"{Path(str(image_id)).stem}.npz"

    # 使用 .npz 将 feature 和 length 绑定保存，避免特征矩阵与长度信息错位。
    np.savez_compressed(save_path, feature=feature_matrix, length=np.asarray(length, dtype=np.int64))


def process_and_save_split(
    data_list: list[dict],
    keyed_vectors: KeyedVectors,
    split_save_folder: str,
    split_name: str,
    max_tokens: int = 64,
) -> None:
    """处理 train/val/test 中的一个划分，并为每个样本写入 Word2Vec 序列特征。"""
    print(f"\nStart processing {split_name} split...")
    Path(split_save_folder).mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(data_list):
        # poi_prompt -> [max_tokens, word_dim]，同时记录真实 token 数 length。
        feature, length = sentence_to_sequence(item["poi_prompt"], keyed_vectors, max_tokens=max_tokens)
        save_single_feature(feature, length, item["image_id"], split_save_folder)

        if (idx + 1) % 100 == 0:
            print(f"{split_name}: processed {idx + 1}/{len(data_list)}")

    print(f"{split_name} split finished.")


def make_poi_word2vec_features(
    train_jsonl: str,
    val_jsonl: str,
    test_jsonl: str,
    word2vec_path: str,
    output_folder: str,
    max_tokens: int = 64,
) -> None:
    """加载本地 Word2Vec，并为 train/val/test 预生成 BiLSTM 所需的序列特征。"""
    print("\nLoading local Word2Vec KeyedVectors...")

    # mmap="r" 可以降低加载大规模 Word2Vec 文件时的内存压力。
    keyed_vectors = KeyedVectors.load(word2vec_path, mmap="r")
    print(f"Loaded Word2Vec model from: {word2vec_path}")

    # 三个 jsonl 已经提前划分好，这里只负责读取并生成对应 split 的特征。
    splits = [
        (load_jsonl_data(train_jsonl), "train"),
        (load_jsonl_data(val_jsonl), "val"),
        (load_jsonl_data(test_jsonl), "test"),
    ]

    for data, split_name in splits:
        if data:
            # 输出结构示例：output_folder/train/SH_42815.npz。
            split_save_folder = Path(output_folder) / split_name
            process_and_save_split(data, keyed_vectors, str(split_save_folder), split_name, max_tokens=max_tokens)

    print("\nAll POI Word2Vec sequence features have been generated.")


def main() -> None:
    parser = argparse.ArgumentParser()

    # 输入是已经划分好的 jsonl 文件，每行至少包含 image_id、poi_prompt 等字段。
    parser.add_argument("--train_jsonl", default="./data/splits/train.jsonl")
    parser.add_argument("--val_jsonl", default="./data/splits/val.jsonl")
    parser.add_argument("--test_jsonl", default="./data/splits/test.jsonl")

    # 本地 Google News Word2Vec KeyedVectors 模型路径。
    parser.add_argument("--word2vec_path", default="./download_models/word2vec-google-news-300/word2vec-google-news-300.model")

    # 输出目录会包含 train/val/test 三个子目录，每个样本一个 .npz 文件。
    parser.add_argument("--output_folder", default="./process_data/word2vec_lstm_features")

    # BiLSTM 的固定输入序列长度：长句截断，短句补 0。
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    make_poi_word2vec_features(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        word2vec_path=args.word2vec_path,
        output_folder=args.output_folder,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

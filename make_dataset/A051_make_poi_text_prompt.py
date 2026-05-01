"""
    根据建筑中心到每个poi的距离进行采样
    根据距离和数量构建text prompt
"""
import cv2
import rasterio
import geopandas as gpd
from shapely.geometry import Point, mapping
from tqdm import tqdm
import os
import numpy as np
from transformers import SiglipTokenizer, SiglipTextModel
import torch
import json
from collections import Counter


def turn_city_shp_into_jsonls():
    """
        将城市对应的shapefile文件汇总为jsonl文件
        逻辑更新：
            1. 贪婪采样：优先取0-100m，不足则取100-200m，再不足取200-300m，上限100个。
            2. 新增 poi_distance 字段，记录 POI 所属的距离区间（100/200/300）。
            3. 移除了经纬度和 Geohash 字段，仅保留文本和距离标签。
    :return:
    """
    # ================= 配置路径和参数区域 =================
    # 建筑Mask文件夹路径，存储建筑的掩码图像
    MASK_DIR         = r"G:\Python_Project_02\paper-03\2_building_masks"
    # 建筑功能标签Shapefile路径，包含建筑的功能分类信息
    LABEL_SHP_PATH   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6-1_add_city_id\add_cls_change_id_sh.shp"
    # POI矢量数据路径 (建议包含待匹配相关的字段)
    POI_SHP_PATH     = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\new_mid_class\shanghai-2023-add-mid-info.shp"
    # 输出JSON保存路径
    OUTPUT_JSON_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\shanghai.jsonl"

    # 待匹配的城市字段
    city_name = "SH"

    # 标签数据LABEL_SHP_PATH 中的字段名
    LABEL_ID_FIELD    = 'build_id'    # 建筑ID字段名，用于与Mask文件名进行匹配
    LABEL_CLS_FIELD   = 'Matched_Fu'  # 建筑功能类别字段名（值为1-7）
    LABEL_SCORE_FIELD = 'Match_Scor'  # 建筑功能类别字段名（值为1-7）
    # POI 字段映射，这些字段必须存在于 POI_SHP_PATH 对应的属性表中
    COL_TYPE = 'type_en'  # POI 类别文本

    # 采样参数配置
    BUFFER_RADIUS = 300  # 缓冲半径（米），用于筛选出靠近建筑质心的POI点
    MAX_POI_COUNT = 100  # 每个样本最大保留的POI数量
    MIN_POI_COUNT = 3    # 每个样本最少要求的POI数量（少于此数则丢弃样本）
    # ================= 配置路径和参数区域 =================

    # 1. 加载POI矢量数据
    print("正在加载 POI 数据...")
    poi_gdf = gpd.read_file(POI_SHP_PATH)  # 使用GeoPandas加载POI数据
    assert poi_gdf.crs.to_epsg() == 3857, f"错误: POI 数据坐标系必须是 EPSG:3857"

    # 构建空间索引以加速查询
    print("构建 POI 空间索引...")
    sindex = poi_gdf.sindex

    print("构建标签映射字典...")
    label_gdf = gpd.read_file(LABEL_SHP_PATH)  # 加载建筑标签数据
    label_map = {
        str(row[LABEL_ID_FIELD]): { # 构建以id为唯一的字典存储结构
            "cls": row[LABEL_CLS_FIELD],
            "score": row[LABEL_SCORE_FIELD]
        }
        for _, row in label_gdf.iterrows()
    }
    print(f"已加载 {len(label_map)} 条标签记录。")

    # 2. 获取所有待处理的 Mask 文件
    mask_files = [
        os.path.join(MASK_DIR, f)
        for f in os.listdir(MASK_DIR)
        if f.lower().endswith('.tif') and city_name in f
    ]
    dataset_list = []

    print(f"开始处理 {len(mask_files)} 个样本...")
    for mask_path in tqdm(mask_files):
        file_name_full = os.path.basename(mask_path)    # 获取文件名（含后缀）
        image_id = os.path.splitext(file_name_full)[0]  # 去掉后缀得到建筑ID（如"NJ_12345"）

        # 匹配标签 (如果匹配不到，则报错提示)
        if image_id not in label_map:
            raise ValueError(f"致命错误: 影像 ID {image_id} 在标签文件(a.shp)中未找到对应记录！处理已终止。")

        # 获取标签值
        current_label = int(float(label_map[image_id]['cls'])) - 1  # 从字典中获取该建筑的功能类别，自动将类别全部减1，统一设置为（0 1 2 3 4 5 6）
        # 获取匹配得分
        matched_score = float(label_map[image_id]["score"])         # 从字典中获取该建筑匹配功能时的得分，在分层采样时需要使用

        # 打开 Mask 影像
        with rasterio.open(mask_path) as src:
            # 断言 Mask 影像必须是 EPSG:3857
            img_epsg = src.crs.to_epsg()
            assert img_epsg == 3857, f"错误: 影像 {file_name_full} 坐标系必须是 EPSG:3857，但检测到: EPSG:{img_epsg}"

            mask_data = src.read(1)    # 读取第一个波段（通常Mask只有一波段，值为0/255或0/1）
            transform = src.transform  # 获取仿射变换参数（像素↔投影坐标转换用）

            # 连通性分析，使用OpenCV进行8连通域分析，返回标签数、标签图、统计信息、质心
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                mask_data.astype(np.uint8), connectivity=8)

            # 过滤并校验：确保只有一个建筑体
            valid_indices = []
            for i in range(1, num_labels):          # 从1开始，跳过背景（label=0）
                if stats[i, cv2.CC_STAT_AREA] > 0:  # 面积大于0（理论上都大于0，但保险）
                    valid_indices.append(i)
            # 若包含多个建筑或无建筑，跳过该样本以保证单体建筑的纯净性
            if len(valid_indices) != 1:
                raise ValueError( f"数据错误: 影像 {image_id} 包含 {len(valid_indices)} 个连通区域（必须有且仅有 1 个），程序已终止。")

            # 获取质心并转为投影坐标
            cx, cy = centroids[valid_indices[0]]                       # 获取唯一连通域的质心（列col, 行row）
            proj_x, proj_y = rasterio.transform.xy(transform, cy, cx)  # 将像素(row, col)转为投影坐标(x, y)
            center_point = Point(proj_x, proj_y)                       # 创建Shapely Point对象（3857坐标）

            # 空间检索：300米缓冲区
            buffer_geom = center_point.buffer(BUFFER_RADIUS)  # 直接基于3857投影坐标系生成指定距离的Buffer

            # 空间索引粗筛
            possible_idx = list(sindex.intersection(buffer_geom.bounds))  # 用空间索引快速获取可能相交的POI索引
            if not possible_idx: continue

            # 精确相交筛选并创建副本
            exact_matches = poi_gdf.iloc[possible_idx].copy()
            exact_matches = exact_matches[exact_matches.intersects(buffer_geom)].copy()
            if exact_matches.empty: continue
            # 计算距离并排序
            exact_matches['dist'] = exact_matches.geometry.distance(center_point)

            # ------------- 贪婪采样策略实现 -------------
            # 直接按距离升序排列，取前 MAX_POI_COUNT 个
            # 这自动满足了：优先填满 0-100m，不够再填 100-200m，以此类推
            final_matches = exact_matches.sort_values('dist').head(MAX_POI_COUNT).copy()

            # 划分距离等级
            def get_dist_tag(d):
                if d <= 100: return "100"
                if d <= 200: return "200"
                return "300"

            final_matches['dist_label'] = final_matches['dist'].apply(get_dist_tag)

            # 提取字段
            texts = final_matches[COL_TYPE].astype(str).tolist()
            dist_labels = final_matches['dist_label'].tolist()

            if len(texts) < MIN_POI_COUNT:
                continue

            # 构建最终输出项
            data_item = {
                "split": "",
                "image_id": image_id,
                "file_name": file_name_full,
                "fun_cls": current_label,
                "fun_score": matched_score,
                "poi_text": "?".join(texts),
                "poi_distance": "?".join(dist_labels)
            }
            dataset_list.append(data_item)

    # 保存 JSON
    print(f"处理完成，共生成 {len(dataset_list)} 条数据，正在保存...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  # 以JSONL格式（每行一个JSON）写入文件，确保中文不被转义
    print(f"文件已保存至: {OUTPUT_JSON_PATH}")



def cal_text_tokens():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\model_cache\models\lcybuaa1111\Git-RSCLIP"
    text = "Within the immediate surroundings of the building, there has a few scattered Apartments, Factories, and Restaurants. Within the nearby area of the building, there shows some visible mostly Restaurants, with some Factories and Apartments. Within the outer area of the building, there shows some visible a mix of Restaurants and Factories and Apartments."
    # 1. 加载 tokenizer
    tokenizer = SiglipTokenizer.from_pretrained(model_path)
    # 2. 加载 SigLIP Text Encoder（仅文本部分）
    text_encoder = SiglipTextModel.from_pretrained(model_path)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    # 3. 文本 tokenize
    inputs = tokenizer(
        text,
        padding=False,
        truncation=True,
        return_tensors="pt",
        max_length=64,
    )
    input_ids = inputs["input_ids"]  # [1, seq_len]
    token_num = input_ids.shape[1]  # token 数量
    # # 4. 前向编码
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids.to(device),
        )
        # SigLIP 的文本特征（句级别表示）
        # last_hidden_state: [1, seq_len, hidden_dim]
        # pooler_output:    [1, hidden_dim]
        text_features = outputs.pooler_output
    print("Token IDs:", input_ids)
    print("Token 数量:", token_num)
    print("Text feature shape:", text_features.shape)

def batch_cal_prompt_tokens():
    """
        遍历一个json文件，提取其中的poi_prompt字段，然后依次遍历每一个prompt的token是否超过了64
    :return:
    """
    ################## 提前一次性加载模型 ##################
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\model_cache\models\lcybuaa1111\Git-RSCLIP"
    # 1. 加载 tokenizer
    tokenizer = SiglipTokenizer.from_pretrained(model_path)
    # # 2. 加载 SigLIP Text Encoder（仅文本部分）
    # text_encoder = SiglipTextModel.from_pretrained(model_path)
    # text_encoder = text_encoder.to(device)
    # text_encoder.eval()
    ################## 提前一次性加载模型 ##################
    jsonl_file = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\20260129_test\test_1.jsonl"
    save_list = []
    with open(jsonl_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line.strip())
            poi_prompt = data["poi_prompt"]
            file_name = data["file_name"]
            ################################
            # 加载tokenizer
            inputs = tokenizer(
                poi_prompt,
                padding=False,
                truncation=True,
                return_tensors="pt",
                max_length=64,
            )
            input_ids = inputs["input_ids"]  # [1, seq_len]
            token_num = input_ids.shape[1]  # token 数量
            ################################
            if int(token_num) >= 62:
                # print("存在超过60的tokens")
                # print(file_name)
                # print("*"*50)
                save_list.append(file_name)
    save_list = list(set(save_list))
    print("超过60token的数量是:", len(save_list))
    print(save_list)



def generate_gradient_prompt(poi_texts, poi_distances):
    """
        根据核心（immediate surroundings）、周边（nearby area）、外围（outer area）三个空间层次，
        独立分析每个层次中 POI（兴趣点）的密度（强度）与功能类型构成，并生成一段连贯的自然语言描述。
        参数:
            poi_texts (list of str): 每个 POI 的功能类别名称（如 "restaurant", "park"）
            poi_distances (list of str or int): 每个 POI 对应的距离层级（100/200/300 米），用于分层
        返回:
            str: 一段描述建筑周围三层空间功能分布的自然语言文本
    """
    # 检查输入是否为空：若两个列表都为空或任一为空，则无法生成描述，抛出异常
    if not poi_texts or not poi_distances: # 检查输入是否为空，如果POI文本或距离为空，则抛出异常
        raise ValueError("The building is located in an area with no recorded urban functions across all proximity layers.")

    # 定义距离层级到语义标签的映射：
    # - 100米 → immediate surroundings（紧邻区域）
    # - 200米 → nearby area（附近区域）
    # - 300米 → outer area（外围区域）
    # 这种命名更具空间方位感，便于生成人类可读的描述
    layers_mapping = {"immediate surroundings": 100, "nearby area": 200, "outer area": 300}

    # 构建每层对应的 POI 列表：
    # 使用字典推导式，对每个 (label, dist) 对，
    # 从 zip(poi_texts, poi_distances) 中筛选出距离等于 dist 的 POI 文本
    layers_data = {
        label: [p for p, d in zip(poi_texts, poi_distances) if int(d) == dist]
        for label, dist in layers_mapping.items()
    }

    # 辅助函数：将一个字符串列表格式化为符合英语语法的自然语言枚举
    # 例如：
    # [] → ""
    # ["park"] → "park"
    # ["cafe", "bank"] → "cafe and bank"
    # ["school", "hospital", "mall"] → "school, hospital, and mall"
    def format_list(items):
        if not items: return ""
        if len(items) == 1: return items[0]
        if len(items) == 2: return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    # 内部函数：分析单一层级的 POI 分布，生成一句描述
    def analyze_layer(poi_list, layer_label):
        count = len(poi_list) # 当前层 POI 总数，作为“密度”指标
        # 修正主谓一致
        verb = "have" if "surroundings" in layer_label else "has"

        # 根据 POI 数量生成密度描述（避免使用“活动强度”等模糊术语，改用视觉可感知的词汇）
        if count == 0: # 若该层无 POI，直接说明为空
            return f"the {layer_label} is empty"
        # 密度描述词
        if count <= 3:
            density_desc = f"{verb} few"
        elif count <= 20:
            density_desc = f"{verb} some"
        elif count <= 40:
            density_desc = f"{verb} many"
        elif count <= 80:
            density_desc = f"is dense with"
        else:
            density_desc = f"is packed with"

        # 统计当前层各类别 POI 的出现频次
        counts    = Counter(poi_list)
        # 取出现频率最高的前 3 类别（即使总数不足 3 个也会返回全部）
        # top_items = counts.most_common(3)           # 找出出现次数最多的前 n 个元素，并按频率从高到低排序。
        top_items = counts.most_common(5)           # 找出出现次数最多的前 n 个元素，并按频率从高到低排序。
        top_types = [item[0] for item in top_items] # 仅提取类别名

        # 根据 POI 总数决定如何描述功能构成
        if count <= 3:
            # POI 极少时，直接列出所有类别，不加修饰词，保持简洁准确
            composition = format_list(top_types)
        else:
            # 计算主导类别的占比
            top_ratio   = top_items[0][1] / count # 得到了主导类别的占比（分子：出现次数最多的POI类别 / 分母：该层所有 POI 的数量）
            main_type   = top_types[0]            # 主导类别
            other_types = top_types[1:]           # 其他次要类别（可能为空）

            # 根据主导性程度选择不同描述策略：
            if top_ratio >= 0.6:
                # 强主导（≥60%）：暗示功能单一（如纯住宅区）
                composition = f"primarily {main_type}" + (f" alongside some {format_list(other_types)}" if other_types else "")
            elif 0.3 <= top_ratio < 0.6:
                # 中等主导（35%~60%）：强调混合但有主次
                composition = f"a mix of {format_list(top_types)}"
            else:
                # 无明显主导（<35%）：完全混合，功能多样
                composition = f"a diverse range of {format_list(top_types)}"

        # 拼接最终句子：In the [方位], there [密度描述] [功能构成].
        return f"the {layer_label} {density_desc} {composition}"

    # 1. 先生成三个层级的描述列表
    layer_descriptions = [analyze_layer(layers_data[k], k) for k in layers_mapping.keys()]
    # 2. 用 "; " 将它们连接起来
    body = "; ".join(layer_descriptions)
    # 3. 加上统一的开头，结尾加句号
    return f"Urban building context: {body}."

def generate_gradient_prompt_test(poi_texts, poi_distances):
    """
        test
    """
    # 检查输入是否为空：若两个列表都为空或任一为空，则无法生成描述，抛出异常
    if not poi_texts or not poi_distances: # 检查输入是否为空，如果POI文本或距离为空，则抛出异常
        raise ValueError("The building is located in an area with no recorded urban functions across all proximity layers.")
    layers_mapping = {"immediate surroundings": 100, "nearby area": 200, "outer area": 300}
    layers_data = {
        label: [p for p, d in zip(poi_texts, poi_distances) if int(d) == dist]
        for label, dist in layers_mapping.items()
    }
    def format_list(items):
        if not items: return ""
        if len(items) == 1: return items[0]
        if len(items) == 2: return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"
    # 内部函数：分析单一层级的 POI 分布，生成一句描述
    def analyze_layer(poi_list, layer_label):
        count = len(poi_list) # 当前层 POI 总数，作为“密度”指标
        # 根据 POI 数量生成密度描述（避免使用“活动强度”等模糊术语，改用视觉可感知的词汇）
        if count == 0: # 若该层无 POI，直接说明为空
            return f"the {layer_label} is empty"
        # 确定助动词
        is_plural = "surroundings" in layer_label
        be_v      = "are" if is_plural else "is"
        have_v    = "have" if is_plural else "has"
        # 密度描述
        if count <= 3:
            density_desc = f"{be_v} sparsely featured with"
        elif count <= 20:
            density_desc = f"{be_v} dotted with"
        elif count <= 40:
            density_desc = f"{be_v} filled with"
        elif count <= 80:
            density_desc = f"{be_v} crowded with"
        else:
            density_desc = f"{be_v} packed with"
        # 统计当前层各类别 POI 的出现频次
        counts    = Counter(poi_list)
        # 取出现频率最高的前 3 类别（即使总数不足 3 个也会返回全部）
        # top_items = counts.most_common(3)           # 找出出现次数最多的前 n 个元素，并按频率从高到低排序。
        top_items = counts.most_common(5)           # 找出出现次数最多的前 n 个元素，并按频率从高到低排序。
        top_types = [item[0] for item in top_items] # 仅提取类别名
        # 根据 POI 总数决定如何描述功能构成
        if count <= 3:
            # POI 极少时，直接列出所有类别，不加修饰词，保持简洁准确
            composition = format_list(top_types)
        else:
            # 计算主导类别的占比
            top_ratio   = top_items[0][1] / count # 得到了主导类别的占比（分子：出现次数最多的POI类别 / 分母：该层所有 POI 的数量）
            main_type   = top_types[0]            # 主导类别
            other_types = top_types[1:]           # 其他次要类别（可能为空）
            # 根据主导性程度选择不同描述策略：
            if top_ratio >= 0.6:
                # 强主导（≥60%）：暗示功能单一（如纯住宅区）
                composition = f"primarily {main_type}" + (f" alongside some {format_list(other_types)}" if other_types else "")
            elif 0.3 <= top_ratio < 0.6:
                # 中等主导（35%~60%）：强调混合但有主次
                composition = f"a mix of {format_list(top_types)}"
            else:
                # 无明显主导（<35%）：完全混合，功能多样
                composition = f"a diverse range of {format_list(top_types)}"
        # 拼接最终句子：In the [方位], there [密度描述] [功能构成].
        return f"the {layer_label} {density_desc} {composition}"
    # 1. 先生成三个层级的描述列表
    layer_descriptions = [analyze_layer(layers_data[k], k) for k in layers_mapping.keys()]
    # 2. 用 "; " 将它们连接起来
    body = "; ".join(layer_descriptions)
    # 3. 加上统一的开头，结尾加句号
    return f"Urban building context: {body}."

def use_jsonl_to_make_poi_text_prompt():
    """
        根据poi点的距离和类别数量，构建poi text prompt
    :return:
    """
    input_file  = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\20260129_test\test.jsonl"
    output_file = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\20260129_test\test_1.jsonl"

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            # 提取字段
            poi_text = data.get('poi_text').split('?')
            poi_dist = data.get('poi_distance').split('?')

            # 生成 Prompt
            # prompt = generate_gradient_prompt(poi_text, poi_dist)
            prompt = generate_gradient_prompt_test(poi_text, poi_dist)

            # 将生成的 Prompt 存回数据并写入新文件
            data['poi_prompt'] = prompt
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')



def turn_7class_into_3class():
    """
        将三个jsonl文件中的7分类标签修改为3分类标签
        1. 工业和商业合并
        2. 保留住宅区
        3. 剩余其他合并为公共服务区public service
    :return:
    """
    jsonl_files = [
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\train.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\val.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\test.jsonl",
    ]
    for file_path in jsonl_files:
        updated_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                label_class = data['fun_cls']
                if label_class >= 1 and label_class <= 2:
                    data['fun_cls'] = 1
                elif label_class >= 3:
                    data['fun_cls'] = 2
                updated_lines.append(json.dumps(data, ensure_ascii=False))
        # 保存回原文件（或者可以保存为新的文件）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(updated_lines) + "\n")
    print("所有 JSONL 文件的 分类标签 已更新完成！")


if __name__ == '__main__':
    # turn_city_shp_into_jsonls()       # 1. 将城市的shp文件整合为jsonl文件，并且根据距离分层采样poi数据

    use_jsonl_to_make_poi_text_prompt() # 2. 基于统计的poi数量、类别信息生成text prompt

    # cal_text_tokens()                 # 计算输入文本的token
    # batch_cal_prompt_tokens()         # 批量计算文本的token

    # turn_7class_into_3class()         # 统一将原来的7个类别的数据修改为3个类别的数据
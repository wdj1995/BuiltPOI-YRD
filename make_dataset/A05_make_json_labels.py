"""
    基于每栋建筑构建POI语料库，其中json文件中还需要标注该建筑对应的功能属性
"""

"""
    现在已经生成了建筑分割mask、遥感影像切片rs影响
    接下来需要基于构建的建筑中心数据构建json标签数据
    ①首先是获取周围500m范围内的所有poi点（包括经纬度坐标数据、类别描述文本、geohash数据）
    ②其次是获取该建筑对应的功能类别数据
    ③上述数据需要保持build_id统一
"""

import cv2
import rasterio
import geopandas as gpd
from shapely.geometry import Point, mapping
from tqdm import tqdm
import pandas as pd
import shutil
import os
import json
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import json
import shutil
from glob import glob


def make_json_labels():
    # ================= 配置路径区域 =================
    # 建筑Mask文件夹路径
    MASK_DIR = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\building_masks"

    # 建筑功能标签Shapefile
    LABEL_SHP_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls\add_cls_nj.shp"

    # POI 矢量数据路径 (建议包含 'name' 或 'type' 字段)
    POI_SHP_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\nanjing-2023-add-info.shp"

    # 输出JSON保存路径
    OUTPUT_JSON_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\nanjing.jsonl"

    # 标签数据LABEL_SHP_PATH 中的字段名
    LABEL_ID_FIELD  = 'build_id'     # 建筑ID字段名，用于匹配Mask文件名
    LABEL_CLS_FIELD = 'Matched_Fu'   # 建筑功能类别字段名（值为1-7）

    # POI 字段映射  # 这些字段必须存在于 POI_SHP_PATH 对应的属性表中
    COL_LNG  = 'lng_wgs'   # 原始 WGS84 经度
    COL_LAT  = 'lat_wgs'   # 原始 WGS84 纬度
    COL_TYPE = 'type_en'   # POI 类别文本
    COL_HASH = 'geohash'   # Geohash 编码
    # ==============================================

    # 加载POI矢量数据
    print("正在加载 POI 数据...")
    poi_gdf = gpd.read_file(POI_SHP_PATH) # 加载POI数据
    poi_epsg = poi_gdf.crs.to_epsg() # 强制要求POI数据为EPSG:3857（米单位，便于缓冲区计算）
    assert poi_epsg == 3857, f"错误: POI 数据坐标系必须是 EPSG:3857，但检测到: EPSG:{poi_epsg}"

    # 构建空间索引以加速查询
    print("构建 POI 空间索引...")
    sindex = poi_gdf.sindex

    # 加载建筑标签数据
    label_gdf = gpd.read_file(LABEL_SHP_PATH)  # 加载建筑功能标注数据
    # 构建字典: { "build_id_str": label_value }
    print("构建标签映射字典...")
    label_map = dict(zip(
        label_gdf[LABEL_ID_FIELD].astype(str), # 将build_id强制转为字符串作为key
        label_gdf[LABEL_CLS_FIELD]             # 对应的功能标签作为value
    ))
    print(f"已加载 {len(label_map)} 条标签记录。")

    # 2. 获取所有Mask文件
    # 只选择文件名（不含路径）中包含 "hz" 的 .tif 文件（不区分大小写）
    mask_files = [
        os.path.join(MASK_DIR, f)
        for f in os.listdir(MASK_DIR)
        if f.lower().endswith('.tif') and 'NJ' in f
    ]
    dataset_list = []

    print(f"开始处理 {len(mask_files)} 个样本...")
    for mask_path in tqdm(mask_files):
        # 解析 ID
        file_name_full = os.path.basename(mask_path)    # 获取文件名（含后缀）
        image_id = os.path.splitext(file_name_full)[0]  # 去掉后缀得到建筑ID（如"12345"）
        image_id = image_id.split("_")[1]

        # B. 匹配标签 (如果匹配不到，跳过该数据)
        if image_id not in label_map:
            raise ValueError(f"致命错误: 影像 ID {image_id} 在标签文件(a.shp)中未找到对应记录！处理已终止。")

        # 获取标签值
        current_label = int(float(label_map[image_id])) - 1 # 从字典中获取该建筑的功能类别，自动将类别全部减1，统一设置为（0 1 2 3 4 5 6）

        # 打开 Mask 影像
        with rasterio.open(mask_path) as src:
            # 断言 Mask 影像必须是 EPSG:3857
            img_epsg = src.crs.to_epsg()
            assert img_epsg == 3857, f"错误: 影像 {file_name_full} 坐标系必须是 EPSG:3857，但检测到: EPSG:{img_epsg}"

            mask_data = src.read(1)    # 读取第一个波段（通常Mask只有一波段，值为0/255或0/1）
            transform = src.transform  # 获取仿射变换参数（像素↔投影坐标转换用）

            # --- 连通性分析 ---
            # 使用OpenCV进行8连通域分析，返回标签数、标签图、统计信息、质心
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                mask_data.astype(np.uint8), connectivity=8)

            # 过滤背景 (label 0)
            valid_indices = []
            for i in range(1, num_labels):          # 从1开始，跳过背景（label=0）
                if stats[i, cv2.CC_STAT_AREA] > 0:  # 面积大于0（理论上都大于0，但保险）
                    valid_indices.append(i)

            # 判断是否存在唯一连通区域
            if len(valid_indices) != 1:
                raise ValueError(f"数据错误: 影像 {image_id} 包含 {len(valid_indices)} 个连通区域（必须有且仅有 1 个），程序已终止。")

            # --- 获取质心并转坐标 ---
            cx, cy = centroids[valid_indices[0]] # 获取唯一连通域的质心（列col, 行row）

            # 像素坐标 -> 投影坐标 (EPSG:3857)
            proj_x, proj_y = rasterio.transform.xy(transform, cy, cx)  # 将像素(row, col)转为投影坐标(x, y)
            center_point = Point(proj_x, proj_y)  # 创建Shapely Point对象（3857坐标）

            # --- 空间匹配 ---
            # 直接基于 3857 生成 200m Buffer
            buffer_geom = center_point.buffer(200)

            # 粗筛 (Spatial Index)
            possible_matches_idx = list(sindex.intersection(buffer_geom.bounds))  # 用空间索引快速获取可能相交的POI索引
            possible_matches = poi_gdf.iloc[possible_matches_idx]                 # 提取候选POI

            # 精筛 (Geometric Intersection)
            exact_matches = possible_matches[possible_matches.intersects(buffer_geom)]  # 精确几何相交筛选

            # --- 字段提取与拼接 ---
            # 提取并转为字符串列表
            texts = exact_matches[COL_TYPE].fillna('').astype(str).str.replace(';', ' ', regex=False).tolist() # 将POI类型中的分号替换为空格（避免后续分隔符冲突）
            lngs = exact_matches[COL_LNG].astype(str).tolist()
            lats = exact_matches[COL_LAT].astype(str).tolist()
            hashes = exact_matches[COL_HASH].fillna('').astype(str).tolist()

            # 拼接经纬度成 "lng,lat" 形式
            locs = [f"{lng},{lat}" for lng, lat in zip(lngs, lats)]

            # 拼接最终字符串
            poi_text_str = "?".join(texts)
            poi_loc_str = "?".join(locs)
            poi_geohash_str = "?".join(hashes)

            if poi_text_str.count('?') < 2:
                # 也可以使用 len(texts) < 3，效果一样且更直观
                # raise ValueError(f"数据不足: 影像 {image_id} 周边仅找到 {len(texts)} 个 POI 点（要求至少 3 个），程序已终止。")
                continue # POI点数量不足，不能建图

            if not (poi_text_str.count('?') == poi_loc_str.count('?') == poi_geohash_str.count('?')):
                raise ValueError(
                    f"数据完整性错误: ID {image_id} 的字段对齐失败！\n"
                    f"检测到 '?' 数量不一致 -> Text: {poi_text_str.count('?')}, Location: {poi_loc_str.count('?')}, Geohash: {poi_geohash_str.count('?')}\n"
                    f"请检查 POI 名称中是否包含未清洗的 '?' 符号。"
                )

            # --- 构建数据 ---
            data_item = {
                "split": "",                    # 预留的划分字段，目前为空
                "image_id":     image_id,       # 建筑ID
                "file_name":    file_name_full, # Mask文件名（含后缀）
                "fun_cls":      current_label,  # 建筑功能标签（1-7）
                "poi_text":     poi_text_str,   # 周边POI类别文本（?分隔）
                "poi_location": poi_loc_str,    # 周边POI经纬度（?分隔）
                "poi_geohash":  poi_geohash_str # 周边POI geohash（?分隔）
            }

            dataset_list.append(data_item)  # 添加到最终数据集列表

    # 保存 JSON
    print(f"处理完成，共生成 {len(dataset_list)} 条数据，正在保存...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n') # 以JSONL格式（每行一个JSON）写入文件，确保中文不被转义
    print(f"文件已保存至: {OUTPUT_JSON_PATH}")
def make_json_labels_consider_POI_distance_number():
    """
        根据建筑周围POI的数量、功能类型进行划分
    :return:
    """
    # ================= 配置路径和参数区域 =================
    # 建筑Mask文件夹路径，存储建筑的掩码图像
    MASK_DIR = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\building_masks"

    # 建筑功能标签Shapefile路径，包含建筑的功能分类信息
    LABEL_SHP_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls\add_cls_sh.shp"

    # POI矢量数据路径 (建议包含待匹配相关的字段)
    POI_SHP_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\shanghai-2023-add-info.shp"

    # 输出JSON保存路径
    OUTPUT_JSON_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\shanghai.jsonl"

    # 待匹配的城市字段
    city_name = "SH"

    # 标签数据LABEL_SHP_PATH 中的字段名
    LABEL_ID_FIELD  = 'build_id'     # 建筑ID字段名，用于与Mask文件名进行匹配
    LABEL_CLS_FIELD = 'Matched_Fu'   # 建筑功能类别字段名（值为1-7）

    # POI 字段映射，这些字段必须存在于 POI_SHP_PATH 对应的属性表中
    COL_LNG  = 'lng_wgs'   # 原始 WGS84 经度
    COL_LAT  = 'lat_wgs'   # 原始 WGS84 纬度
    COL_TYPE = 'type_en'   # POI 类别文本
    COL_HASH = 'geohash'   # Geohash 编码

    # 采样参数配置
    BUFFER_RADIUS = 200  # 缓冲半径（米），用于筛选出靠近建筑质心的POI点
    MAX_POI_COUNT = 50   # 每个样本最大保留的POI数量
    MIN_POI_COUNT = 3    # 每个样本最少要求的POI数量（少于此数则丢弃样本）
    # ================= 配置路径和参数区域 =================

    # 1. 加载POI矢量数据
    print("正在加载 POI 数据...")
    poi_gdf = gpd.read_file(POI_SHP_PATH) # 使用GeoPandas加载POI数据
    poi_epsg = poi_gdf.crs.to_epsg()      # 获取并转换POI数据的EPSG代码
    assert poi_epsg == 3857, f"错误: POI 数据坐标系必须是 EPSG:3857，但检测到: EPSG:{poi_epsg}"

    # 构建空间索引以加速查询
    print("构建 POI 空间索引...")
    sindex = poi_gdf.sindex

    print("构建标签映射字典...")
    label_gdf = gpd.read_file(LABEL_SHP_PATH)  # 加载建筑标签数据
    label_map = dict(zip(
        label_gdf[LABEL_ID_FIELD].astype(str), # 将建筑ID强制转为字符串作为key
        label_gdf[LABEL_CLS_FIELD]             # 对应的功能标签作为value
    )) # 构建字典: { "build_id_str": label_value }
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
        image_id = os.path.splitext(file_name_full)[0]  # 去掉后缀得到建筑ID（如"12345"）
        image_id = image_id.split("_")[1]

        # 匹配标签 (如果匹配不到，则报错提示)
        if image_id not in label_map:
            raise ValueError(f"致命错误: 影像 ID {image_id} 在标签文件(a.shp)中未找到对应记录！处理已终止。")

        # 获取标签值
        current_label = int(float(label_map[image_id])) - 1 # 从字典中获取该建筑的功能类别，自动将类别全部减1，统一设置为（0 1 2 3 4 5 6）

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

            # 过滤背景 (label 0)
            valid_indices = []
            for i in range(1, num_labels):          # 从1开始，跳过背景（label=0）
                if stats[i, cv2.CC_STAT_AREA] > 0:  # 面积大于0（理论上都大于0，但保险）
                    valid_indices.append(i)

            # 若包含多个建筑或无建筑，跳过该样本以保证单体建筑的纯净性
            if len(valid_indices) != 1:
                raise ValueError(f"数据错误: 影像 {image_id} 包含 {len(valid_indices)} 个连通区域（必须有且仅有 1 个），程序已终止。")

            # 获取质心并转为投影坐标
            cx, cy = centroids[valid_indices[0]]                       # 获取唯一连通域的质心（列col, 行row）
            proj_x, proj_y = rasterio.transform.xy(transform, cy, cx)  # 将像素(row, col)转为投影坐标(x, y)
            center_point = Point(proj_x, proj_y)                       # 创建Shapely Point对象（3857坐标）

            # 空间匹配
            buffer_geom = center_point.buffer(BUFFER_RADIUS) # 直接基于3857投影坐标系生成指定距离的Buffer

            # 空间索引粗筛
            possible_matches_idx = list(sindex.intersection(buffer_geom.bounds)) # 用空间索引快速获取可能相交的POI索引
            if not possible_matches_idx:
                continue

            # 精确相交筛选并创建副本
            possible_matches = poi_gdf.iloc[possible_matches_idx].copy()
            exact_matches = possible_matches[possible_matches.intersects(buffer_geom)].copy()

            # 排序与切片逻辑
            if not exact_matches.empty:
                # 计算该范围内所有 POI 到建筑质心的欧氏距离（单位：米）
                exact_matches['dist'] = exact_matches.geometry.distance(center_point)

                # 按距离由近到远排序
                exact_matches = exact_matches.sort_values(by='dist', ascending=True)

                # 筛选前 K 个最近的 POI (K=50) # 若总数不足 50，head(50) 会返回所有匹配到的点
                exact_matches = exact_matches.head(MAX_POI_COUNT)

            # 字段提取与拼接
            # 提取并转为字符串列表
            texts = exact_matches[COL_TYPE].fillna('').astype(str).str.replace(';', ' ', regex=False).tolist()
            lngs = exact_matches[COL_LNG].astype(str).tolist()
            lats = exact_matches[COL_LAT].astype(str).tolist()
            hashes = exact_matches[COL_HASH].fillna('').astype(str).tolist()

            # 数量校验：少于设定阈值则舍弃样本，确保语义足够丰富
            if len(texts) < MIN_POI_COUNT:
                continue

            # 格式化经纬度串
            locs = [f"{lng},{lat}" for lng, lat in zip(lngs, lats)]

            # 拼接最终字符串
            poi_text_str = "?".join(texts)
            poi_loc_str = "?".join(locs)
            poi_geohash_str = "?".join(hashes)

            # 数据对齐完整性校验
            if not (len(texts) == len(locs) == len(hashes)):
                raise ValueError(f"ID {image_id} 字段对齐失败！")

            # 构建数据
            data_item = {
                "split": "",                    # 预留的划分字段，目前为空
                "image_id":     image_id,       # 建筑ID
                "file_name":    file_name_full, # Mask文件名（含后缀）
                "fun_cls":      current_label,  # 建筑功能标签（1-7）
                "poi_text":     poi_text_str,   # 周边POI类别文本（?分隔）
                "poi_location": poi_loc_str,    # 周边POI经纬度（?分隔）
                "poi_geohash":  poi_geohash_str # 周边POI geohash（?分隔）
            }

            dataset_list.append(data_item)  # 添加到最终数据集列表

    # 保存 JSON
    print(f"处理完成，共生成 {len(dataset_list)} 条数据，正在保存...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n') # 以JSONL格式（每行一个JSON）写入文件，确保中文不被转义
    print(f"文件已保存至: {OUTPUT_JSON_PATH}")
def make_json_labels_consider_POI_distance_number_matchScore():
    """
        根据建筑周围POI的数量、功能类型进行划分
        此外需要考虑匹配得分
    :return:
    """
    # ================= 配置路径和参数区域 =================
    # 建筑Mask文件夹路径，存储建筑的掩码图像
    MASK_DIR = r"G:\Python_Project_02\paper-03\2_building_masks"

    # 建筑功能标签Shapefile路径，包含建筑的功能分类信息
    LABEL_SHP_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6-1_add_city_id\add_cls_change_id_hz.shp"

    # POI矢量数据路径 (建议包含待匹配相关的字段)
    POI_SHP_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\hangzhou-2023-add-info.shp"

    # 输出JSON保存路径
    OUTPUT_JSON_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60\hangzhou.jsonl"

    # 待匹配的城市字段
    city_name = "HZ"

    # 标签数据LABEL_SHP_PATH 中的字段名
    LABEL_ID_FIELD =     'build_id'      # 建筑ID字段名，用于与Mask文件名进行匹配
    LABEL_CLS_FIELD =    'Matched_Fu'    # 建筑功能类别字段名（值为1-7）
    LABEL_SCORE_FIELD =  'Match_Scor'    # 建筑功能类别字段名（值为1-7）

    # POI 字段映射，这些字段必须存在于 POI_SHP_PATH 对应的属性表中
    COL_LNG  = 'lng_wgs'   # 原始 WGS84 经度
    COL_LAT  = 'lat_wgs'   # 原始 WGS84 纬度
    COL_TYPE = 'type_en'   # POI 类别文本
    COL_HASH = 'geohash'   # Geohash 编码

    # 采样参数配置
    BUFFER_RADIUS = 300  # 缓冲半径（米），用于筛选出靠近建筑质心的POI点
    MAX_POI_COUNT = 31   # 每个样本最大保留的POI数量
    MIN_POI_COUNT = 3    # 每个样本最少要求的POI数量（少于此数则丢弃样本）
    # ================= 配置路径和参数区域 =================

    # 1. 加载POI矢量数据
    print("正在加载 POI 数据...")
    poi_gdf = gpd.read_file(POI_SHP_PATH)  # 使用GeoPandas加载POI数据
    poi_epsg = poi_gdf.crs.to_epsg()       # 获取并转换POI数据的EPSG代码
    assert poi_epsg == 3857, f"错误: POI 数据坐标系必须是 EPSG:3857，但检测到: EPSG:{poi_epsg}"

    # 构建空间索引以加速查询
    print("构建 POI 空间索引...")
    sindex = poi_gdf.sindex

    print("构建标签映射字典...")
    label_gdf = gpd.read_file(LABEL_SHP_PATH)  # 加载建筑标签数据
    # label_map = dict(zip(
    #     label_gdf[LABEL_ID_FIELD].astype(str),  # 将建筑ID强制转为字符串作为key
    #     label_gdf[LABEL_CLS_FIELD]  # 对应的功能标签作为value
    # ))  # 构建字典: { "build_id_str": label_value }
    label_map = {
        str(row[LABEL_ID_FIELD]): {
            "cls": row[LABEL_CLS_FIELD],
            "score": row[LABEL_SCORE_FIELD]
        }
        for _, row in label_gdf.iterrows()
    } # 构建以id为唯一的字典存储结构
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
        matched_score = float(label_map[image_id]["score"]) # 从字典中获取该建筑匹配功能时的得分，在分层采样时需要使用

        # 打开 Mask 影像
        with rasterio.open(mask_path) as src:
            # 断言 Mask 影像必须是 EPSG:3857
            img_epsg = src.crs.to_epsg()
            assert img_epsg == 3857, f"错误: 影像 {file_name_full} 坐标系必须是 EPSG:3857，但检测到: EPSG:{img_epsg}"

            mask_data = src.read(1)  # 读取第一个波段（通常Mask只有一波段，值为0/255或0/1）
            transform = src.transform  # 获取仿射变换参数（像素↔投影坐标转换用）

            # 连通性分析，使用OpenCV进行8连通域分析，返回标签数、标签图、统计信息、质心
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                mask_data.astype(np.uint8), connectivity=8)

            # 过滤背景 (label 0)
            valid_indices = []
            for i in range(1, num_labels):  # 从1开始，跳过背景（label=0）
                if stats[i, cv2.CC_STAT_AREA] > 0:  # 面积大于0（理论上都大于0，但保险）
                    valid_indices.append(i)

            # 若包含多个建筑或无建筑，跳过该样本以保证单体建筑的纯净性
            if len(valid_indices) != 1:
                raise ValueError(
                    f"数据错误: 影像 {image_id} 包含 {len(valid_indices)} 个连通区域（必须有且仅有 1 个），程序已终止。")

            # 获取质心并转为投影坐标
            cx, cy = centroids[valid_indices[0]]  # 获取唯一连通域的质心（列col, 行row）
            proj_x, proj_y = rasterio.transform.xy(transform, cy, cx)  # 将像素(row, col)转为投影坐标(x, y)
            center_point = Point(proj_x, proj_y)  # 创建Shapely Point对象（3857坐标）

            # 空间匹配
            buffer_geom = center_point.buffer(BUFFER_RADIUS)  # 直接基于3857投影坐标系生成指定距离的Buffer

            # 空间索引粗筛
            possible_matches_idx = list(sindex.intersection(buffer_geom.bounds))  # 用空间索引快速获取可能相交的POI索引
            if not possible_matches_idx:
                continue

            # 精确相交筛选并创建副本
            possible_matches = poi_gdf.iloc[possible_matches_idx].copy()
            exact_matches = possible_matches[possible_matches.intersects(buffer_geom)].copy()

            # 排序与切片逻辑
            if not exact_matches.empty:
                # 计算该范围内所有 POI 到建筑质心的欧氏距离（单位：米）
                exact_matches['dist'] = exact_matches.geometry.distance(center_point)

                # 按距离由近到远排序
                exact_matches = exact_matches.sort_values(by='dist', ascending=True)

                # 筛选前 K 个最近的 POI (K=50) # 若总数不足 50，head(50) 会返回所有匹配到的点
                exact_matches = exact_matches.head(MAX_POI_COUNT)

            # 字段提取与拼接
            # 提取并转为字符串列表
            texts = exact_matches[COL_TYPE].fillna('').astype(str).str.replace(';', ' ', regex=False).tolist()
            lngs = exact_matches[COL_LNG].astype(str).tolist()
            lats = exact_matches[COL_LAT].astype(str).tolist()
            hashes = exact_matches[COL_HASH].fillna('').astype(str).tolist()

            # 数量校验：少于设定阈值则舍弃样本，确保语义足够丰富
            if len(texts) < MIN_POI_COUNT:
                continue

            # 格式化经纬度串
            locs = [f"{lng},{lat}" for lng, lat in zip(lngs, lats)]

            # 拼接最终字符串
            poi_text_str = "?".join(texts)
            poi_loc_str = "?".join(locs)
            poi_geohash_str = "?".join(hashes)

            # 数据对齐完整性校验
            if not (len(texts) == len(locs) == len(hashes)):
                raise ValueError(f"ID {image_id} 字段对齐失败！")

            # 构建数据
            data_item = {
                "split": "",                    # 预留的划分字段，目前为空
                "image_id": image_id,           # 建筑ID
                "file_name": file_name_full,    # Mask文件名（含后缀）
                "fun_cls": current_label,       # 建筑功能标签（1-7）
                "fun_score": matched_score,     # 建筑功能匹配的得分
                "poi_text": poi_text_str,       # 周边POI类别文本（?分隔）
                "poi_location": poi_loc_str,    # 周边POI经纬度（?分隔）
                "poi_geohash": poi_geohash_str  # 周边POI geohash（?分隔）
            }

            dataset_list.append(data_item)  # 添加到最终数据集列表
    # 保存 JSON
    print(f"处理完成，共生成 {len(dataset_list)} 条数据，正在保存...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  # 以JSONL格式（每行一个JSON）写入文件，确保中文不被转义
    print(f"文件已保存至: {OUTPUT_JSON_PATH}")
def make_json_labels_consider_POI_distance_number_matchScore_filter():
    """
        根据建筑周围POI的数量、功能类型进行划分
        此外需要考虑匹配得分
        设计空间概率采样，将距离划分为0-100,100-200,200-300之间进行，保证POI点数量位于20之内
    :return:
    """
    # ================= 配置路径和参数区域 =================
    # 建筑Mask文件夹路径，存储建筑的掩码图像
    MASK_DIR         = r"G:\Python_Project_02\paper-03\2_building_masks"
    # 建筑功能标签Shapefile路径，包含建筑的功能分类信息
    LABEL_SHP_PATH   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6-1_add_city_id\add_cls_change_id_hz.shp"
    # POI矢量数据路径 (建议包含待匹配相关的字段)
    POI_SHP_PATH     = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\hangzhou-2023-add-info.shp"
    # 输出JSON保存路径
    OUTPUT_JSON_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60\hangzhou.jsonl"

    # 待匹配的城市字段
    city_name = "HZ"

    # 标签数据LABEL_SHP_PATH 中的字段名
    LABEL_ID_FIELD = 'build_id'       # 建筑ID字段名，用于与Mask文件名进行匹配
    LABEL_CLS_FIELD = 'Matched_Fu'    # 建筑功能类别字段名（值为1-7）
    LABEL_SCORE_FIELD = 'Match_Scor'  # 建筑功能类别字段名（值为1-7）

    # POI 字段映射，这些字段必须存在于 POI_SHP_PATH 对应的属性表中
    COL_LNG = 'lng_wgs'   # 原始 WGS84 经度
    COL_LAT = 'lat_wgs'   # 原始 WGS84 纬度
    COL_TYPE = 'type_en'  # POI 类别文本
    COL_HASH = 'geohash'  # Geohash 编码

    # 采样参数配置
    BUFFER_RADIUS = 300  # 缓冲半径（米），用于筛选出靠近建筑质心的POI点
    MAX_POI_COUNT = 20   # 每个样本最大保留的POI数量
    MIN_POI_COUNT = 3    # 每个样本最少要求的POI数量（少于此数则丢弃样本）
    RANDOM_SEED   = 42   # 固定随机种子保证实验可重复
    # ================= 配置路径和参数区域 =================

    # 1. 加载POI矢量数据
    print("正在加载 POI 数据...")
    poi_gdf = gpd.read_file(POI_SHP_PATH)  # 使用GeoPandas加载POI数据
    assert poi_gdf.crs.to_epsg() == 3857, f"错误: POI 数据坐标系必须是 EPSG:3857，但检测到: EPSG:{poi_epsg}"

    # 构建空间索引以加速查询
    print("构建 POI 空间索引...")
    sindex = poi_gdf.sindex

    print("构建标签映射字典...")
    label_gdf = gpd.read_file(LABEL_SHP_PATH)  # 加载建筑标签数据
    label_map = {
        str(row[LABEL_ID_FIELD]): {
            "cls": row[LABEL_CLS_FIELD],
            "score": row[LABEL_SCORE_FIELD]
        }
        for _, row in label_gdf.iterrows()
    } # 构建以id为唯一的字典存储结构
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

            # 过滤背景 (label 0)
            valid_indices = []
            for i in range(1, num_labels):  # 从1开始，跳过背景（label=0）
                if stats[i, cv2.CC_STAT_AREA] > 0:  # 面积大于0（理论上都大于0，但保险）
                    valid_indices.append(i)

            # 若包含多个建筑或无建筑，跳过该样本以保证单体建筑的纯净性
            if len(valid_indices) != 1:
                raise ValueError( f"数据错误: 影像 {image_id} 包含 {len(valid_indices)} 个连通区域（必须有且仅有 1 个），程序已终止。")

            # 获取质心并转为投影坐标
            cx, cy = centroids[valid_indices[0]]                       # 获取唯一连通域的质心（列col, 行row）
            proj_x, proj_y = rasterio.transform.xy(transform, cy, cx)  # 将像素(row, col)转为投影坐标(x, y)
            center_point = Point(proj_x, proj_y)                       # 创建Shapely Point对象（3857坐标）

            # 空间匹配
            buffer_geom = center_point.buffer(BUFFER_RADIUS)  # 直接基于3857投影坐标系生成指定距离的Buffer

            # 空间索引粗筛
            possible_matches_idx = list(sindex.intersection(buffer_geom.bounds))  # 用空间索引快速获取可能相交的POI索引
            if not possible_matches_idx:
                continue

            # 精确相交筛选并创建副本
            possible_matches = poi_gdf.iloc[possible_matches_idx].copy()
            exact_matches = possible_matches[possible_matches.intersects(buffer_geom)].copy()

            if exact_matches.empty: continue
            # 计算距离
            exact_matches['dist'] = exact_matches.geometry.distance(center_point)

            # 自适应分层采样逻辑
            if len(exact_matches) <= MAX_POI_COUNT: # 短路逻辑：如果不满 20 个，全部保留
                # 如果总数本身就不够 20 个，那么无需任何复杂的采样，直接将这些 POI 按距离（dist）从近到远排序，全部作为该建筑的特征点。这保护了稀疏地区的数据完整性。
                final_matches = exact_matches.sort_values('dist')

            else:
                # 划分三个距离桶
                tier_a = exact_matches[exact_matches['dist'] < 100].sort_values('dist') # 筛选出距离建筑 0-100 米内的 POI
                tier_b = exact_matches[(exact_matches['dist'] >= 100) & (exact_matches['dist'] < 200)] # 筛选出 100-200 米内的 POI
                tier_c = exact_matches[(exact_matches['dist'] >= 200) & (exact_matches['dist'] < 300)] # 筛选出 200-300 米内的 POI

                selected_parts = [] # 初始化一个列表，用来临时存放三个区间筛选出来的点。

                # 1. (0-100m): 确定性排序取前 10
                take_a = tier_a.head(10) # 从 0-100 米的 POI 中，取距离最近的前 10 个。这保证了模型能捕捉到最强、最直接的语义信号。
                selected_parts.append(take_a) # 将选出的这部分点存入结果列表

                # 2. (100-200m): 基础 5 个 + 借调 A 剩下的名额
                quota_b = 5 + (10 - len(take_a)) # 该区间的标准名额是 5 个 # 如果内圈（Tier A）不足 10 个，那么差额 (10 - len(take_a)) 将补偿给中圈，增加中圈的采样名额。
                if len(tier_b) > quota_b: # 检查中圈拥有的 POI 数量是否超过了计算出的名额
                    take_b = tier_b.sample(n=quota_b, random_state=RANDOM_SEED) # 如果超过了，就进行随机采样
                else:
                    take_b = tier_b # 如果中圈本身就不足名额数，则全部保留
                selected_parts.append(take_b)

                # 3. 背景层 (200-300m): 补齐总数到 20 个
                current_count = sum(len(df) for df in selected_parts) # 计算到目前为止（A 层 + B 层）一共选出了多少个点
                quota_c = MAX_POI_COUNT - current_count # 为了让总数达到 20，计算 C 层需要补充多少个名额。
                if len(tier_c) > quota_c: # 如果 C 层点多，就随机抽；如果点少，就全拿走。
                    take_c = tier_c.sample(n=quota_c, random_state=RANDOM_SEED)
                else:
                    take_c = tier_c
                selected_parts.append(take_c)

                final_matches = pd.concat(selected_parts).sort_values('dist')

            # 字段提取与拼接
            # 提取并转为字符串列表
            texts  = final_matches[COL_TYPE].fillna('').astype(str).str.replace(';', ' ', regex=False).tolist()
            lngs   = final_matches[COL_LNG].astype(str).tolist()
            lats   = final_matches[COL_LAT].astype(str).tolist()
            hashes = final_matches[COL_HASH].fillna('').astype(str).tolist()

            # 数量校验：少于设定阈值则舍弃样本，确保语义足够丰富
            if len(texts) < MIN_POI_COUNT:
                continue

            # 格式化经纬度串
            locs = [f"{lng},{lat}" for lng, lat in zip(lngs, lats)]
            # 拼接最终字符串
            poi_text_str = "?".join(texts)
            poi_loc_str = "?".join(locs)
            poi_geohash_str = "?".join(hashes)

            # 数据对齐完整性校验
            if not (len(texts) == len(locs) == len(hashes)):
                raise ValueError(f"ID {image_id} 字段对齐失败！")

            # 构建数据
            data_item = {
                "split": "",  # 预留的划分字段，目前为空
                "image_id": image_id,  # 建筑ID
                "file_name": file_name_full,  # Mask文件名（含后缀）
                "fun_cls": current_label,  # 建筑功能标签（1-7）
                "fun_score": matched_score,  # 建筑功能匹配的得分
                "poi_text": poi_text_str,  # 周边POI类别文本（?分隔）
                "poi_location": poi_loc_str,  # 周边POI经纬度（?分隔）
                "poi_geohash": poi_geohash_str  # 周边POI geohash（?分隔）
            }

            dataset_list.append(data_item)  # 添加到最终数据集列表
    # 保存 JSON
    print(f"处理完成，共生成 {len(dataset_list)} 条数据，正在保存...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  # 以JSONL格式（每行一个JSON）写入文件，确保中文不被转义
    print(f"文件已保存至: {OUTPUT_JSON_PATH}")


def merge_jsonls():
    """
        合并多个jsonl文件
    :return:
    """
    jsonl_paths = [
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\hangzhou_prompt.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\nanjing_prompt.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\shanghai_prompt.jsonl",
    ]

    output_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\merged.jsonl"
    with open(output_path, 'w', encoding='utf-8') as fout:
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line:  # 跳过空行
                        fout.write(line + '\n')

    print(f"合并完成，共合并 {len(jsonl_paths)} 个文件 → {output_path}")


def cal_id_city_cls_index():
    """
        统计每条样本数据所属的城市信息、功能类别信息、poi的数量信息
        要求划分数据集的时候需要参考上述数据
    :return:
    """
    # ====================================================== Step 1: 读取与初步特征工程 ======================================================
    jsonl_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\merged.jsonl"

    # 检查 jsonl 文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"错误：找不到文件 {jsonl_path}")
        return pd.DataFrame()

    # 初始化用于存储样本统计信息的列表
    rows = []
    # 读取JSONL文件并提取特征
    with open(jsonl_path, "r", encoding="utf-8") as f: # 打开 jsonl 文件
        for line in f: # 逐行读取 jsonl
            data = json.loads(line.strip())

            # 基础信息提取
            file_name = data.get("file_name", "")
            city = file_name.split("_")[0] if "_" in file_name else "Unknown"
            fun_cls = data.get("fun_cls", -1)

            # POI 数量计算
            poi_geohash = data.get("poi_geohash", "")
            poi_num = len(poi_geohash.split("?")) if poi_geohash else 0

            # 将当前样本统计信息加入列表
            rows.append({
                "image_id": data.get("image_id"),
                "file_name": file_name,
                "city": city,
                "fun_cls": fun_cls,
                "poi_num": poi_num
            })

    # 转换为DataFrame并记录原始样本数
    df_result = pd.DataFrame(rows)
    # 全局随机打乱（在任何过滤和分层之前）
    df_result = df_result.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # 保存统计结果
    df_result.to_csv(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv\result_index.csv",
                     index=False)
    original_len = len(df_result) # 记录原始样本数量

    # 对POI数量进行分箱处理（10个分位数箱）
    est = KBinsDiscretizer(n_bins=8, strategy="quantile", encode="ordinal")
    df_result["poi_bin"] = est.fit_transform(df_result[["poi_num"]]).astype(int)

    print(f"Step 1 完成: 原始数据加载共 {len(df_result)} 条。")
    # ====================================================== Step 1: 读取与初步特征工程 ======================================================


    # ====================================================== Step 2: 稀缺样本过滤  ======================================================
    # 创建组合特征用于识别稀缺样本
    df_result['temp_combo'] = (
            df_result['city'] + "_" +
            df_result['fun_cls'].astype(str) + "_" +
            df_result['poi_bin'].astype(str)
    )
    # 统计每个组合的样本数量
    combo_counts = df_result['temp_combo'].value_counts()
    # 识别样本数小于3的稀缺组合
    rare_combos = combo_counts[combo_counts < 3].index
    # 过滤掉稀缺样本
    df_final = df_result[~df_result['temp_combo'].isin(rare_combos)].reset_index(drop=True)

    print(f"Step 1.5 完成: 过滤稀缺样本 {original_len - len(df_final)} 条，剩余有效样本 {len(df_final)} 条。")
    # ====================================================== Step 2: 稀缺样本过滤 ======================================================


    # ====================================================== Step 3: 构造分层标签 ======================================================
    # 对城市字段进行 one-hot 编码
    city_dummies = pd.get_dummies(df_final["city"], prefix="city")
    # # 对功能类别进行 one-hot 编码
    cls_dummies = pd.get_dummies(df_final["fun_cls"], prefix="cls")
    # 对 POI 分箱进行 one-hot 编码
    poi_bin_dummies = pd.get_dummies(df_final["poi_bin"], prefix="poi_bin") # 直接使用 df_final 中已有的 'poi_bin' 列，不需要重新 fit_transform
    # 拼接多标签矩阵
    Y = pd.concat([city_dummies, cls_dummies, poi_bin_dummies], axis=1).values

    print(f"Step 2 完成: 分层标签矩阵 Shape: {Y.shape}")
    # ====================================================== Step 3: 构造分层标签 ======================================================


    # ====================================================== Step 4: 分层切分 ======================================================
    # 定义划分比例和随机种子
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    random_seed = 42

    # 第一次划分：训练集 vs (验证集+测试集)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(val_ratio + test_ratio), # 测试集比例为验证集+测试集总和
        random_state=random_seed
    )
    # 执行第一次分层抽样
    train_idx, temp_idx = next(msss.split(df_final, Y))

    # 创建训练集和临时集
    train_df = df_final.iloc[train_idx].reset_index(drop=True) # 构造训练集
    temp_df = df_final.iloc[temp_idx].reset_index(drop=True)   # 构造训练集

    # 计算第二次划分时验证集的比例
    val_size = val_ratio / (val_ratio + test_ratio)  # 0.5

    # 第二次划分：验证集 vs 测试集
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - val_size),
        random_state=random_seed
    )
    # 执行第二次分层抽样
    val_idx, test_idx = next(msss2.split(temp_df, Y[temp_idx]))

    # 创建验证集和测试集
    val_df = temp_df.iloc[val_idx].reset_index(drop=True) # 构造验证集
    test_df = temp_df.iloc[test_idx].reset_index(drop=True) # 构造测试集
    # ====================================================== Step 4: 分层切分 ======================================================


    # ====================================================== Step 5: 验证分层效果  ======================================================
    print("\n>>> Step 5: 开始验证分层效果...")
    # 定义分布检查函数
    def check_distribution(df_name, df, total_len):
        if len(df) == 0: return {}
        stats = {
            "name": df_name,
            "count": len(df),
            "ratio_global": len(df) / total_len, # 全局比例
            "city_dist": df["city"].value_counts(normalize=True).to_dict(), # 城市分布
            "poi_dist_head": df["poi_num"].describe()[["mean", "50%", "max"]].to_dict() # POI统计
        }
        return stats

    # 检查所有数据集的分布
    total_len_final = len(df_final)
    stats_train = check_distribution("Train", train_df, total_len_final)
    stats_val = check_distribution("Val", val_df, total_len_final)
    stats_test = check_distribution("Test", test_df, total_len_final)

    # 打印城市分布对比
    print(f"{'City':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    all_cities = df_final['city'].unique()

    for city in all_cities:
        t_r = stats_train.get('city_dist', {}).get(city, 0)
        v_r = stats_val.get('city_dist', {}).get(city, 0)
        e_r = stats_test.get('city_dist', {}).get(city, 0)
        print(f"{city:<10} {t_r:.4f}     {v_r:.4f}     {e_r:.4f}")

    # 检查数据集大小是否合理
    if len(val_df) < 10 or len(test_df) < 10:
        print("警告：验证集或测试集样本过少！")

    print(">>> 验证结束\n")
    # ====================================================== Step 5: 验证分层效果 ======================================================


    # ====================================================== Step 6: 保存结果 ======================================================
    train_df.to_csv(os.path.join(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv",
                                 "train.csv"), index=False)
    val_df.to_csv(os.path.join(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv",
                               "val.csv"), index=False)
    test_df.to_csv(os.path.join(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv",
                                "test.csv"), index=False)
    print("数据集划分完成：")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)}")
    # ====================================================== Step 6: 保存结果 ======================================================


    # ====================================================== Step 7: 根据 image_id 划分 jsonl ======================================================
    print("开始根据 image_id 划分 merged.jsonl ...")

    # 先将原始数据全部加载到内存字典中，以 file_name 为 Key
    # 这样做是为了后续可以按照 df 的乱序顺序直接查找数据
    data_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            file_name = data.get("file_name")
            if file_name:
                data_map[file_name] = data

    print(f"原始数据已加载到内存，共 {len(data_map)} 条。")

    # 定义输出文件路径
    train_jsonl = os.path.join(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv",
                               "train.jsonl")
    val_jsonl = os.path.join(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv",
                             "val.jsonl")
    test_jsonl = os.path.join(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv",
                              "test.jsonl")

    # 定义一个写入函数，强制对写入顺序进行 shuffle
    def write_shuffled_jsonl(target_df, output_path, split_name):
        # 再次强制打乱 DataFrame 的顺序 (虽然之前 split 已经是乱序，但为了保险起见)
        target_df_shuffled = target_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            # 遍历乱序后的 DataFrame 中的 image_id
            for file_name in target_df_shuffled["file_name"]:
                # 从内存字典中取出对应的完整数据
                row_data = data_map.get(file_name)
                if row_data:
                    row_data["split"] = split_name
                    f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
                    count += 1
        return count

    # 2. 执行写入
    c_train = write_shuffled_jsonl(train_df, train_jsonl, "train")
    c_val = write_shuffled_jsonl(val_df, val_jsonl, "val")
    c_test = write_shuffled_jsonl(test_df, test_jsonl, "test")

    print("jsonl 划分并乱序写入完成：")
    print(f"  Train jsonl: {c_train} 条 -> {train_jsonl}")
    print(f"  Val   jsonl: {c_val} 条 -> {val_jsonl}")
    print(f"  Test  jsonl: {c_test} 条 -> {test_jsonl}")


def cal_id_city_cls_index_use_POI_number_filter_jsonls():
    """
        需要定量筛选指定的样本数量
        fun_cls = 0: 52827
        fun_cls = 1: 4589
        fun_cls = 2: 183
        fun_cls = 3: 658
        fun_cls = 4: 2023
        fun_cls = 5: 6506
        fun_cls = 6: 10590

        首先筛选0 1 5 6类别数量是2050条，其余类别的数据全部保留
        最后再进行数据集划分

        优化内容：
            1. 针对大类（0,1,5,6）按 city + poi_num 比例抽取 2050 条（保证多样性）。
            2. 针对筛选后的数据重新计算 poi_bin（8分箱）。
            3. 划分数据集时使用 poi_bin 进行分层（保证稳定性）。
    :return:
    """
    # ====================================================== Step 1: 读取与初步特征工程 ======================================================
    jsonl_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\merged.jsonl"

    # 检查 jsonl 文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"错误：找不到文件 {jsonl_path}")
        return pd.DataFrame()

    # 初始化用于存储样本统计信息的列表
    rows = []
    # 读取JSONL文件并提取特征
    with open(jsonl_path, "r", encoding="utf-8") as f:  # 打开 jsonl 文件
        for line in f:  # 逐行读取 jsonl
            data = json.loads(line.strip())

            # 基础信息提取
            file_name = data.get("file_name", "")
            city = file_name.split("_")[0] if "_" in file_name else "Unknown"
            fun_cls = data.get("fun_cls", -1)

            # POI 数量计算
            poi_geohash = data.get("poi_text", "")
            poi_num = len(poi_geohash.split("?")) if poi_geohash else 0

            # 将当前样本统计信息加入列表
            rows.append({
                "image_id": data.get("image_id"),
                "file_name": file_name,
                "city": city,
                "fun_cls": fun_cls,
                "poi_num": poi_num
            })

    # 转换为DataFrame并记录原始样本数
    df_result = pd.DataFrame(rows)
    # 保存统计结果
    df_result.to_csv(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\result_index.csv",
        index=False)
    original_len = len(df_result)  # 记录原始样本数量
    print(f"Step 1 完成: 原始数据共 {original_len} 条。")
    # ====================================================== Step 1: 读取与初步特征工程 ======================================================


    # ====================================================== Step 1.5: 开始按照城市和POI分布进行目标重采样 ======================================================
    '''首先筛选0/1/5/6类别数量是2500条，其余类别的数据全部保留'''
    # 1. 定义需要筛选的目标类别及数量
    # target_config = {0: 2000, 1: 2000, 5: 2000, 6: 2000} # 目的：确保抽出来的 2050 条样本里，poi_num 从 3-50 的分布比例与原数据一致
    target_config = {0: 8000, 5: 2000, 6: 2000} # 目的：确保抽出来的 2050 条样本里，poi_num 从 3-50 的分布比例与原数据一致
    # 2. 准备分层组合键 (内部使用)
    # 创建细粒度分层键用于抽样
    df_result['stratify_key_fine'] = df_result['city'] + "_" + df_result['poi_num'].astype(str)
    resampled_list = []
    # 3. 按类别进行处理
    for cls in df_result['fun_cls'].unique():
        cls_data = df_result[df_result['fun_cls'] == cls]
        current_count = len(cls_data)
        # 检查是否在目标筛选名单中
        if cls in target_config and current_count > target_config[cls]:
            target_n = target_config[cls]
            print(f"  类别 {cls}: 正在按 poi_num 比例从 {current_count} 抽取 {target_n}...")
            # 核心修复：统计组合数，确保 stratify 健壮
            counts = cls_data['stratify_key_fine'].value_counts()
            # 只有样本数 >= 2 的组合才能参与分层抽样
            valid_cls_data = cls_data[cls_data['stratify_key_fine'].isin(counts[counts >= 2].index)]
            # 分层抽样
            sampled_df, _ = train_test_split(
                valid_cls_data,
                train_size=target_n,
                stratify=valid_cls_data['stratify_key_fine'],
                random_state=42
            )
            resampled_list.append(sampled_df)
        else:
            resampled_list.append(cls_data)
    # 4. 更新 df_final 供后续步骤使用
    df_filtered = pd.concat(resampled_list).reset_index(drop=True)
    print(f"Step 1.5 完成: 采样后数据规模为 {len(df_filtered)} 条。")
    # ====================================================== Step 1.5: 开始按照城市和POI分布进行目标重采样 ======================================================


    # 目的：为了 Step 4 的切分稳定性，将 3-50 的 poi_num 转化为 8 个分箱标签
    est = KBinsDiscretizer(n_bins=10, strategy="kmeans", encode="ordinal")
    # 注意：如果数据量太小导致 bin 无法区分，这里会自动处理
    df_filtered["poi_bin"] = est.fit_transform(df_filtered[["poi_num"]]).astype(int)


    # ====================================================== Step 2: 稀缺样本过滤  ======================================================
    # 创建组合特征
    df_filtered['temp_combo'] = (
            df_filtered['city'] + "_" +
            df_filtered['fun_cls'].astype(str) + "_" +
            df_filtered['poi_bin'].astype(str)
    )
    combo_counts = df_filtered['temp_combo'].value_counts()
    rare_combos = combo_counts[combo_counts < 3].index  # 样本数小于3的组合无法分给 Train/Val/Test
    df_final = df_filtered[~df_filtered['temp_combo'].isin(rare_combos)].reset_index(drop=True)
    print(f"Step 2 完成: 过滤稀缺组合后剩余 {len(df_final)} 条。")
    # ====================================================== Step 2: 稀缺样本过滤 ======================================================

    # ====================================================== Step 3: 构造分层标签 ======================================================
    city_dummies = pd.get_dummies(df_final["city"], prefix="city")
    cls_dummies = pd.get_dummies(df_final["fun_cls"], prefix="cls")
    # 使用 poi_bin 而非 poi_num，保证标签矩阵 Y 的每一列都有足够的 1
    poi_bin_dummies = pd.get_dummies(df_final["poi_bin"], prefix="poi_bin")
    Y = pd.concat([city_dummies, cls_dummies, poi_bin_dummies], axis=1).values
    print(f"Step 3 完成: 标签矩阵 Shape {Y.shape}")
    # ====================================================== Step 3: 构造分层标签 ======================================================

    # ====================================================== Step 4: 分层切分 ======================================================
    # 定义划分比例和随机种子
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    random_seed = 42

    # 第一次划分：训练集 vs (验证集+测试集)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(val_ratio + test_ratio),  # 测试集比例为验证集+测试集总和
        random_state=random_seed
    )
    # 执行第一次分层抽样
    train_idx, temp_idx = next(msss.split(df_final, Y))

    # 创建训练集和临时集
    train_df = df_final.iloc[train_idx].reset_index(drop=True)  # 构造训练集
    temp_df = df_final.iloc[temp_idx].reset_index(drop=True)  # 构造训练集

    # 计算第二次划分时验证集的比例
    val_size = val_ratio / (val_ratio + test_ratio)  # 0.5

    # 第二次划分：验证集 vs 测试集
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - val_size),
        random_state=random_seed
    )
    # 执行第二次分层抽样
    val_idx, test_idx = next(msss2.split(temp_df, Y[temp_idx]))

    # 创建验证集和测试集
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)  # 构造验证集
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)  # 构造测试集
    # ====================================================== Step 4: 分层切分 ======================================================

    # ====================================================== Step 5: 验证分层效果  ======================================================
    print("\n>>> Step 5: 开始验证分层效果...")

    # 定义分布检查函数
    def check_distribution(df_name, df, total_len):
        if len(df) == 0: return {}
        stats = {
            "name": df_name,
            "count": len(df),
            "ratio_global": len(df) / total_len,  # 全局比例
            "city_dist": df["city"].value_counts(normalize=True).to_dict(),  # 城市分布
            "poi_dist_head": df["poi_num"].describe()[["mean", "50%", "max"]].to_dict()  # POI统计
        }
        return stats

    # 检查所有数据集的分布
    total_len_final = len(df_final)
    stats_train = check_distribution("Train", train_df, total_len_final)
    stats_val = check_distribution("Val", val_df, total_len_final)
    stats_test = check_distribution("Test", test_df, total_len_final)

    # 打印城市分布对比
    print(f"{'City':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    all_cities = df_final['city'].unique()

    for city in all_cities:
        t_r = stats_train.get('city_dist', {}).get(city, 0)
        v_r = stats_val.get('city_dist', {}).get(city, 0)
        e_r = stats_test.get('city_dist', {}).get(city, 0)
        print(f"{city:<10} {t_r:.4f}     {v_r:.4f}     {e_r:.4f}")

    # 检查数据集大小是否合理
    if len(val_df) < 10 or len(test_df) < 10:
        print("警告：验证集或测试集样本过少！")

    print(">>> 验证结束\n")
    # ====================================================== Step 5: 验证分层效果 ======================================================

    # ====================================================== Step 6: 保存结果 ======================================================
    train_df.to_csv(os.path.join(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv",
        "train.csv"), index=False)
    val_df.to_csv(os.path.join(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv",
        "val.csv"), index=False)
    test_df.to_csv(os.path.join(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv",
        "test.csv"), index=False)
    print("数据集划分完成：")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)}")
    # ====================================================== Step 6: 保存结果 ======================================================

    # ====================================================== Step 7: 根据 image_id 划分 jsonl ======================================================
    print("开始根据 image_id 划分 merged.jsonl ...")

    # 先将原始数据全部加载到内存字典中，以 file_name 为 Key
    # 这样做是为了后续可以按照 df 的乱序顺序直接查找数据
    data_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            file_name = data.get("file_name")
            if file_name:
                data_map[file_name] = data

    print(f"原始数据已加载到内存，共 {len(data_map)} 条。")

    # 定义输出文件路径
    train_jsonl = os.path.join(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv",
        "train.jsonl")
    val_jsonl = os.path.join(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv",
        "val.jsonl")
    test_jsonl = os.path.join(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv",
        "test.jsonl")

    # 定义一个写入函数，强制对写入顺序进行 shuffle
    def write_shuffled_jsonl(target_df, output_path, split_name):
        # 再次强制打乱 DataFrame 的顺序 (虽然之前 split 已经是乱序，但为了保险起见)
        target_df_shuffled = target_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            # 遍历乱序后的 DataFrame 中的 image_id
            for file_name in target_df_shuffled["file_name"]:
                # 从内存字典中取出对应的完整数据
                row_data = data_map.get(file_name)
                if row_data:
                    row_data["split"] = split_name
                    f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
                    count += 1
        return count

    # 2. 执行写入
    c_train = write_shuffled_jsonl(train_df, train_jsonl, "train")
    c_val = write_shuffled_jsonl(val_df, val_jsonl, "val")
    c_test = write_shuffled_jsonl(test_df, test_jsonl, "test")

    print("jsonl 划分并乱序写入完成：")
    print(f"  Train jsonl: {c_train} 条 -> {train_jsonl}")
    print(f"  Val   jsonl: {c_val} 条 -> {val_jsonl}")
    print(f"  Test  jsonl: {c_test} 条 -> {test_jsonl}")



def use_jsonls_to_filter_rs_and_masks():
    """
        使用生成的jsonl文件去过滤rs文件和mask文件
    :return:
    """
    jsonl_files = [
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\train.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\val.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\test.jsonl"
    ]

    # 原始遥感影像文件夹
    rs_folder = r"G:\Python_Project_02\paper-03\1_rs_images"
    # 原始 Mask 文件夹
    mask_folder = r"G:\Python_Project_02\paper-03\2_building_masks"

    # 待复制文件的路径
    output_rs_folder = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\1_rs_images"
    output_mask_folder = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\2_building_masks"

    # -------------------- 创建输出文件夹 --------------------
    os.makedirs(output_rs_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # -------------------- 步骤1: 提取 image_id --------------------
    file_name_list = set()  # 用 set 避免重复

    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"警告：文件不存在 {jsonl_file}")
            continue

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                file_name = data.get("file_name").split(".")[0] # 获取“HZ_10004.tif”中的“HZ_10004”
                if file_name:
                    file_name_list.add(str(file_name))  # 转为字符串方便匹配
    print(f"总共提取 {len(file_name_list)} 个 file names")

    # -------------------- 步骤2: 筛选 RS 图像 --------------------
    for fname in os.listdir(rs_folder): # 遍历 RS 图像文件夹中的所有文件名
        name, ext = os.path.splitext(fname) # 将文件名拆分为 文件名 和 扩展名
        if ext.lower() != ".tif": # 判断文件扩展名是否为指定的 RS 图像后缀
            continue
        if name in file_name_list: # 检查文件名是否在之前提取的 image_id 列表中，如果存在，说明这是需要的样本。
            shutil.copy(os.path.join(rs_folder, fname),
                        os.path.join(output_rs_folder, fname))

    # -------------------- 步骤3: 筛选 Mask 文件 --------------------
    for fname in os.listdir(mask_folder):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".tif":
            continue
        if name in file_name_list:
            shutil.copy(os.path.join(mask_folder, fname),
                        os.path.join(output_mask_folder, fname))

    print("数据复制完成！")


def copy_file_name_to_image_id():
    """
        将file_name的内容复制到image_id上面
    :return:
    """
    jsonl_files = [
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv\train.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv\val.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\jsonls\index_csv\test.jsonl"
    ]

    for file_path in jsonl_files:
        updated_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 去掉文件名后缀
                image_id_from_file = os.path.splitext(data['file_name'])[0]
                data['image_id'] = image_id_from_file
                updated_lines.append(json.dumps(data, ensure_ascii=False))

        # 保存回原文件（或者可以保存为新的文件）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(updated_lines) + "\n")

    print("所有 JSONL 文件的 image_id 已更新完成！")

def check_if_file_and_copy_it():
    """
        检查新构建的数据集中tif图像数据是否存在于一个文件夹中，如果没有就进行复制
        功能:
            1. 遍历 jsonl_dir 下所有 jsonl 文件
            2. 逐行读取 json，提取 key 指定的字段（默认 file_name）
            3. 对 file_name 去重（保持顺序）
            4. 判断 file_name 是否存在于 folder_b
            5. 若不存在，则从 folder_a 复制到 folder_c
    :return:
    """
    # jsonl 文件所在目录
    jsonl_files = [
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\train.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\val.jsonl",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\7-class\index_csv\test.jsonl"
    ]

    file_names = []
    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "file_name" in data:
                    file_names.append(data["file_name"])
    raw_count = len(file_names)
    file_names = list(dict.fromkeys(file_names))
    unique_count = len(file_names)

    # folder_b(str): 已存在的子集目录
    folder_b = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\dataset\JL_Images"
    files_in_b = set(os.listdir(folder_b))

    # folder_c (str): 目标复制目录
    folder_c = r"G:\Python_Project_02\paper-03\copy_rs"
    os.makedirs(folder_c, exist_ok=True)

    # 4️⃣ 执行复制逻辑
    os.makedirs(folder_c, exist_ok=True)

    copied = 0
    skipped = 0
    missing_in_a = []

    # folder_a (str): 原始 tif 文件目录（全量）
    folder_a = r"G:\Python_Project_02\paper-03\1_rs_images"
    for file_name in file_names:
        if file_name in files_in_b:
            skipped += 1
            continue

        src_path = os.path.join(folder_a, file_name)
        dst_path = os.path.join(folder_c, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            missing_in_a.append(file_name)

    # 5️⃣ 返回统计信息
    print("jsonl_files", len(jsonl_files))
    print("raw_file_names", raw_count)
    print("unique_file_names", unique_count)
    print("copied_to_c", copied)
    print("already_in_b", skipped)
    print("missing_in_folder_a", missing_in_a)







def count_train_cls_label_ratio():
    """
        统计训练数据集中分类标签类别的数量比例
    :return:
    """
    import json
    from collections import Counter
    import torch
    import math

    jsonl_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\test.jsonl"

    fun_cls_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            fun_cls = data.get("fun_cls", None)
            if fun_cls is not None:
                fun_cls_counter[fun_cls] += 1

    print("fun_cls 统计结果：")
    for k, v in sorted(fun_cls_counter.items()):
        print(f"fun_cls = {k}: {v}")

    num_classes = len(fun_cls_counter)
    total_samples = sum(fun_cls_counter.values())

    alpha_list = []

    print("\n平方根倒数权重（未归一化）：")
    for c in sorted(fun_cls_counter.keys()):
        n_c = fun_cls_counter[c]
        freq = n_c / total_samples
        alpha_c = 1.0 / math.sqrt(freq)
        alpha_list.append(alpha_c)
        print(f"fun_cls = {c}: alpha = {alpha_c:.4f}")

    # 转为 Tensor
    alpha = torch.tensor(alpha_list, dtype=torch.float32)

    # =====================================================
    # Step 3. 归一化（推荐）
    # =====================================================
    alpha = alpha / alpha.sum() * num_classes

    print("\n归一化后的 alpha（用于 Focal Loss）：")
    for idx, val in enumerate(alpha):
        print(f"class {idx}: alpha = {val.item():.4f}")

    print("\n最终 alpha Tensor：")
    print(alpha)




if __name__ == "__main__":
    # make_json_labels() # 生成标签数据
    # make_json_labels_consider_POI_distance_number() # 考虑buffer的距离和POI的数量
    # make_json_labels_consider_POI_distance_number_matchScore() # 考虑buffer的距离、POI的数量、匹配属性得分
    # make_json_labels_consider_POI_distance_number_matchScore_filter() # 在上述基础上进行动态筛选

    # merge_jsonls() # 合并jsonl文件

    # cal_id_city_cls_index() # 统计每条样本数据所属的城市信息、功能类别信息、poi的数量信息
    cal_id_city_cls_index_use_POI_number_filter_jsonls() # 在cal_id_city_cls_index先筛选指定数量的样本

    # use_jsonls_to_filter_rs_and_masks() # 使用生成的jsonl文件去过滤rs文件和mask文件
    # check_if_file_and_copy_it()   # 检查文件夹中是否存在该文件，如果没有将其从总文件夹中复制到一个子文件夹中

    # copy_file_name_to_image_id() # 复制file name字段到image id字段

    # count_train_cls_label_ratio() # 统计训练数据集中分类标签类别的数量比例
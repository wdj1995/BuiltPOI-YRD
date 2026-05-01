
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

"""
    1. 基于之前的采样规则（建筑中心半径300m范围设置buffer，获取最多不超过100个poi，采样按照距离由近到远）
    2. 在上述采样过程中，同时将poi的经纬度坐标信息进行添加
    3. 与之前的数据进行对比，验证采样结果的poi_type字段，如果匹配，则将poi_location添加到原始文件中
"""

def turn_city_shp_into_jsonls():
    """
        将城市对应的shapefile文件汇总为jsonl文件
        逻辑更新：
            1. 贪婪采样：优先取0-100m，不足则取100-200m，再不足取200-300m，上限100个。
            2. 新增 poi_distance 字段，记录 POI 所属的距离区间（100/200/300）。
            3. 移除了经纬度和 Geohash 字段，仅保留文本和距离标签。

        更新：
            将poi的经纬度坐标也添加到数据中
    :return:
    """
    # ================= 配置路径和参数区域 =================
    # 建筑Mask文件夹路径，存储建筑的掩码图像
    MASK_DIR         = r"G:\Python_Project_02\paper-03\2_building_masks"
    # 建筑功能标签Shapefile路径，包含建筑的功能分类信息
    LABEL_SHP_PATH   = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6-1_add_city_id\add_cls_change_id_hz.shp"
    # POI矢量数据路径 (建议包含待匹配相关的字段)
    POI_SHP_PATH     = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\new_mid_class\hangzhou-2023-add-mid-info.shp"
    # 输出JSON保存路径
    OUTPUT_JSON_PATH = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\add_poi_location\hangzhou.jsonl"

    # 待匹配的城市字段
    city_name = "HZ"

    # 标签数据LABEL_SHP_PATH 中的字段名
    LABEL_ID_FIELD    = 'build_id'    # 建筑ID字段名，用于与Mask文件名进行匹配
    LABEL_CLS_FIELD   = 'Matched_Fu'  # 建筑功能类别字段名（值为1-7）
    LABEL_SCORE_FIELD = 'Match_Scor'  # 建筑功能类别字段名（值为1-7）
    # POI 字段映射，这些字段必须存在于 POI_SHP_PATH 对应的属性表中
    COL_TYPE          = 'type_en'  # POI 类别文本

    # 采样参数配置
    BUFFER_RADIUS = 300  # 缓冲半径（米），用于筛选出靠近建筑质心的POI点
    MAX_POI_COUNT = 100  # 每个样本最大保留的POI数量
    MIN_POI_COUNT = 3    # 每个样本最少要求的POI数量（少于此数则丢弃样本）
    # ================= 配置路径和参数区域 =================

    # 1. 加载POI矢量数据
    print("正在加载 POI 数据...")
    poi_gdf = gpd.read_file(POI_SHP_PATH)  # 使用GeoPandas加载POI数据
    assert poi_gdf.crs.to_epsg() == 3857, f"错误: POI 数据坐标系必须是 EPSG:3857"


    # 预处理POI数据
    # 新增一个字段poi_location，该字段的值是使用lng_wgs字段和lat_wgs字段进行拼接，即poi_location=“lng_wgs,lat_wgs”
    poi_gdf['poi_location'] = (poi_gdf['lng_wgs'].astype(str) + "," + poi_gdf['lat_wgs'].astype(str))
    # 对读取的poi数据进行去重，根据的字段就是poi_location
    # 根据拼接后的坐标字段去重，保留第一条记录
    initial_count = len(poi_gdf)
    poi_gdf = poi_gdf.drop_duplicates(subset=['poi_location'], keep='first').copy()
    print(f"去重完成：原始 {initial_count} 条，去重后 {len(poi_gdf)} 条。")


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
            texts       = final_matches[COL_TYPE].astype(str).tolist()
            dist_labels = final_matches['dist_label'].tolist()
            # 新增提取poi_location信息
            locations   = final_matches['poi_location'].tolist()

            if len(texts) < MIN_POI_COUNT:
                continue

            # 构建最终输出项
            data_item = {
                "split"        : "",
                "image_id"     : image_id,
                "file_name"    : file_name_full,
                "fun_cls"      : current_label,
                "fun_score"    : matched_score,
                "poi_text"     : "?".join(texts),
                "poi_distance" : "?".join(dist_labels),
                # 新增poi_location字段
                "poi_location" : "?".join(locations)
            }
            dataset_list.append(data_item)

    # 保存 JSON
    print(f"处理完成，共生成 {len(dataset_list)} 条数据，正在保存...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  # 以JSONL格式（每行一个JSON）写入文件，确保中文不被转义
    print(f"文件已保存至: {OUTPUT_JSON_PATH}")


def check_json_if_true():
    """
        检查上述生成的城市数据与原来的train val test数据是否一致
    :return:
    """

    val_jsonl      = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\20260109\test.jsonl"
    shanghai_jsonl = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\add_poi_location\shanghai.jsonl"
    city_str = "SH"

    # 1. 一次性加载 shanghai.jsonl
    shanghai_index = {}
    with open(shanghai_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] shanghai.jsonl JSON解析失败 (line {line_num})")
                continue
            file_name = item.get("file_name")
            if file_name is None:
                continue
            shanghai_index[file_name] = (
                item.get("poi_text"),
                item.get("poi_distance")
            )

    # 2. 遍历 val.jsonl 并对比
    diff_file_names = []
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                val_item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] val.jsonl JSON解析失败 (line {line_num})")
                continue
            file_name = val_item.get("file_name", "")
            if city_str not in file_name:
                continue
            if file_name not in shanghai_index:
                continue
            val_poi_text                 = val_item.get("poi_text")
            val_poi_distance             = val_item.get("poi_distance")
            sh_poi_text, sh_poi_distance = shanghai_index[file_name]
            if (val_poi_text != sh_poi_text or val_poi_distance != sh_poi_distance):
                diff_file_names.append(file_name)

    # 3. 打印结果
    print("========== 对比结果 ==========")
    print(f"不一致的 file_name 数量: {len(diff_file_names)}")
    if diff_file_names:
        print("不一致的 file_name 列表：")
        for name in diff_file_names:
            print(name)
    else:
        print("未发现不一致的 file_name")


def add_poi_location_info():
    """
        将城市数据中的poi_location添加到train val test数据集中
    :return:
    """
    # 存放城市jsonl文件的路径
    city_jsonl_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\3_jsonls\greater_60_mid_class\add_poi_location"
    city_index = {}
    for file in os.listdir(city_jsonl_dir):
        file_path = os.path.join(city_jsonl_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                item = json.loads(line)
                file_name = item.get("file_name")
                city_index[file_name] = {
                    "poi_text"     : item.get("poi_text"),
                    "poi_distance" : item.get("poi_distance"),
                    "poi_location" : item.get("poi_location")
                }
    print(f"[INFO] 已加载城市数据条数: {len(city_index)}")

    val_jsonl_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\datainfo\20260109\test.jsonl"
    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        val_lines = f.readlines()

    updated_lines = []
    total         = 0
    matched       = 0
    mismatch      = 0

    for line_num, line in enumerate(val_lines, 1):
        line      = line.strip()
        total += 1
        val_item  = json.loads(line)
        file_name = val_item.get("file_name")
        if file_name in city_index:
            city_item         = city_index[file_name]
            val_poi_text      = val_item.get("poi_text")
            val_poi_distance  = val_item.get("poi_distance")
            city_poi_text     = city_item["poi_text"]
            city_poi_distance = city_item["poi_distance"]
            # 完全一致，添加 poi_location
            if val_poi_text == city_poi_text and val_poi_distance == city_poi_distance:
                val_item["poi_location"] = city_item["poi_location"]
                matched += 1
            else:
                mismatch += 1
                print(
                    "[MISMATCH]",
                    file_name,
                    "| poi_text_same:",
                    val_poi_text == city_poi_text,
                    "| poi_distance_same:",
                    val_poi_distance == city_poi_distance
                )
        updated_lines.append(json.dumps(val_item, ensure_ascii=False) + "\n")

    # 4. 覆盖写回原 val.jsonl
    with open(val_jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    # 5. 打印统计
    print("========== 更新完成 ==========")
    print(f"val 总行数: {total}")
    print(f"成功添加 poi_location: {matched}")
    print(f"匹配但 poi 信息不一致: {mismatch}")
    print(f"已原地更新文件: {val_jsonl_path}")




if __name__ == "__main__":

    # turn_city_shp_into_jsonls() # 将城市对应的shapefile文件汇总为jsonl文件

    # check_json_if_true()          # 检查上述生成的城市数据与原来的train val test数据是否一致

    add_poi_location_info() # 为train val test添加poi_location数据
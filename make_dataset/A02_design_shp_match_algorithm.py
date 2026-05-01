"""
    设计一种匹配算法，能够将开源建筑功能类别数据赋给我们识别的建筑分割区域上
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape, Polygon
from tqdm import tqdm


def cal_overlap_ratio(geom_a, geom_b):
    """ 计算 A 被 B 覆盖的比例: Area(A ∩ B) / Area(A) """
    if not geom_a.is_valid or not geom_b.is_valid: return 0.0 # 检查几何对象是否合法 (防止拓扑错误导致程序崩溃)
    if not geom_a.intersects(geom_b): return 0.0              # 预先快速检查是否相交，如果不相交直接返回0，节省计算资源
    return geom_a.intersection(geom_b).area / geom_a.area     # 计算交集面积 / A的面积

def cal_iou(geom_a, geom_b):
    """ 计算交并比 IoU """
    if not geom_a.is_valid or not geom_b.is_valid: return 0.0
    if not geom_a.intersects(geom_b): return 0.0
    intersection = geom_a.intersection(geom_b).area # 计算交集面积
    union = geom_a.union(geom_b).area # 计算并集面积
    if union == 0: return 0.0 # 防止除以0
    return intersection / union

def cal_centroid_distance(geom_a, geom_b):
    """ 计算质心欧氏距离 """
    return geom_a.centroid.distance(geom_b.centroid)


def main(path_ours,
         path_others,
         output_path):
    """
        主函数：执行空间数据的加载、预处理、统一加权匹配与保存
    """
    # --- A. 参数配置区 (在此调整权重) ---
    # 定义四个维度的权重，总和建议为 1.0
    W_IOU     = 0.65      # IoU权重：形状匹配最重要
    W_OVERLAP = 0.15  # 重叠率权重：辅助判断包含关系
    W_AREA    = 0.1      # 面积比权重：防止尺度差异过大 (如保安亭匹配整个小区)
    W_DIST    = 0.1      # 距离权重：位置偏离惩罚

    # 距离衰减常数 (米)。
    # 作用：将距离(0~无穷大)映射为分数(1~0)。
    # 当距离 = 20米时，得分衰减为 1/e (约0.37)。距离越远，得分越低。
    DIST_DECAY_ALPHA = 20.0

    # 自动采纳的最低分数线 (Score < 0.60 的会被标记为 Low_Score)
    ACCEPT_THRESHOLD = 0.60

    # 歧义判定阈值 (如果第一名和第二名分差小于 0.05，认为有歧义)
    AMBIGUITY_DIFF   = 0.05


    # ==========================
    # 1. 加载数据与预处理
    # ==========================
    print("正在加载数据...")
    # 读取我们的分割结果 Shapefile (数据 A)
    shp_ours = gpd.read_file(path_ours)
    # 读取开源的功能区 Shapefile (数据 B)
    shp_others = gpd.read_file(path_others)


    # ==========================
    # 2. 坐标系统一
    # ==========================
    # 检查两个数据的坐标系是否一致。如果不一致，空间查询和距离计算会出错。
    if not shp_others.crs.equals(shp_ours.crs):
        print(f"转换 shp_others 坐标系到 {shp_ours.crs}")  # 输出转换提示
        # 将数据 B 投影转换到 数据 A 的坐标系
        shp_others = shp_others.to_crs(shp_ours.crs)


    # ==========================
    # 3. 初始化字段
    # ==========================
    # target_field = 'class'  # 设定我们需要从 B 数据中获取的字段名
    target_field = 'Function'  # 设定我们需要从 B 数据中获取的字段名
    # 检查 B 数据中是否有这个字段
    if target_field not in shp_others.columns:
        raise ValueError(f"在数据B中未找到字段: {target_field}，请检查列名。")

    # 在 A 数据中初始化新列，用于存储匹配结果
    shp_ours['Matched_Func'] = None         # 匹配到的功能类别
    shp_ours['Match_Status'] = 'Unchecked'  # 匹配状态 (如 Auto_Match, No_Match)
    shp_ours['Match_Score']  = 0.0          # 最终匹配得分
    shp_ours['Match_Info']   = ''           # 详细信息 (如 IoU具体数值，方便Debug)


    # ==========================
    # 4. 构建空间索引 (核心优化)
    # ==========================
    print("正在为海量数据构建空间索引 (R-tree)...")
    # 构建 R-tree 空间索引。这一步非常关键，它能将查询复杂度从 O(N*M) 降低到 O(N*logM)
    # 没有索引的话，几万条数据可能要跑几天；有了索引只要几分钟。
    sindex_others = shp_others.sindex


    # ==========================
    # 5. 遍历匹配
    # ==========================
    print(f"开始遍历 A 数据 (共 {len(shp_ours)} 条)...")

    # 使用 tqdm 显示进度条，如果没安装可以直接用 range(len(shp_ours))
    for idx_a, row_a in tqdm(shp_ours.iterrows(), total=shp_ours.shape[0]): # 遍历A数据
        geom_a = row_a.geometry # 获取A数据几何对象

        # --- 几何合法性检查 ---
        # 如果几何为空或无效（如自相交），跳过并在属性表中记录状态
        if geom_a is None or not geom_a.is_valid:
            shp_ours.at[idx_a, 'Match_Status'] = 'Invalid_Geometry'
            continue

        # --- 步骤 5.1: 利用空间索引快速筛选候选者 (粗筛) ---
        # intersection(bounds) 只判断 Bounding Box (外接矩形) 是否相交
        # 这一步非常快，返回的是 B 数据中可能的行索引列表
        possible_indexes = list(sindex_others.intersection(geom_a.bounds))

        # 如果连外接矩形都没碰到，说明附近没东西，标记无匹配
        if not possible_indexes:
            shp_ours.at[idx_a, 'Match_Status'] = 'No_Match'
            continue

        # 根据索引取出实际的候选 B 数据行
        candidates = shp_others.iloc[possible_indexes]

        # --- 步骤 5.2: 精确几何相交判断 (精筛) ---
        # 外接矩形相交不代表实际形状相交，所以需要用 .intersects() 做精确计算
        precise_matches = candidates[candidates.intersects(geom_a)]

        # 如果精确判断后没有相交的
        if len(precise_matches) == 0:
            shp_ours.at[idx_a, 'Match_Status'] = 'No_Intersect_Precise'
            continue

        # 准备一个列表，存储所有候选对象的评分结果
        candidate_scores = []

        # 遍历所有相交的候选对象 B
        for idx_b, row_b in precise_matches.iterrows():
            geom_b = row_b.geometry

            # A. 计算各项基础指标
            val_iou = cal_iou(geom_a, geom_b)  # 交并比
            val_overlap = cal_overlap_ratio(geom_a, geom_b)  # 覆盖率
            val_dist = cal_centroid_distance(geom_a, geom_b)  # 距离(米)

            # 计算面积比：min(A,B) / max(A,B)，结果在 0~1 之间
            # 用于惩罚尺度差异巨大的匹配 (例如小房子匹配大地块)
            area_a = geom_a.area
            area_b = geom_b.area
            val_area_ratio = min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) > 0 else 0

            # B. 距离归一化 (核心算法)
            # 使用指数衰减函数：e^(-x / alpha)
            # 距离越近，val_dist越小，score_dist 越接近 1；距离越远，分数迅速趋近 0
            score_dist = np.exp(-val_dist / DIST_DECAY_ALPHA)

            # C. 加权总分计算 (结果在 0 ~ 1 之间)
            final_score = (
                    W_IOU * val_iou +
                    W_OVERLAP * val_overlap +
                    W_AREA * val_area_ratio +
                    W_DIST * score_dist
            )

            # 将该候选的结果存入列表
            candidate_scores.append({
                'func': row_b[target_field],  # 候选的功能类别
                'score': final_score,  # 候选的得分
                'details': f"IoU:{val_iou:.2f},Ov:{val_overlap:.2f},Dist:{val_dist:.1f}m"  # 记录详情字符串
            })


        # --- 步骤 5.3: 决策 ---
        # 对候选列表按分数从高到低排序 (reverse=True)
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        # 取出第一名 (Top 1)
        top_match = candidate_scores[0]
        top_score = top_match['score']

        # 在表中记录最高分
        shp_ours.at[idx_a, 'Match_Score'] = round(top_score, 3)

        # 判定逻辑：首先分数必须及格
        if top_score > ACCEPT_THRESHOLD:
            # 检查是否有歧义 (Ambiguity Check)
            is_ambiguous = False
            # 如果存在至少两个候选
            if len(candidate_scores) > 1:
                second_score = candidate_scores[1]['score']  # 取第二名分数
                # 如果第一名和第二名分差太小 (比如 0.82 vs 0.81)
                if (top_score - second_score) < AMBIGUITY_DIFF:
                    is_ambiguous = True
                    shp_ours.at[idx_a, 'Match_Info'] = f"Ambiguous: Top1({top_score:.2f}) vs Top2({second_score:.2f})"

            # 如果判定为有歧义，标记冲突
            if is_ambiguous:
                shp_ours.at[idx_a, 'Match_Status'] = 'Conflict_Ambiguous'
            else:
                # 完美匹配 (分数高且无歧义) -> 自动赋值
                shp_ours.at[idx_a, 'Matched_Func'] = top_match['func']
                shp_ours.at[idx_a, 'Match_Status'] = 'Auto_Match'
                shp_ours.at[idx_a, 'Match_Info'] = top_match['details']
        else:
            # 分数过低 (即使是第一名也不及格)，标记为低分待查
            shp_ours.at[idx_a, 'Match_Status'] = 'Low_Score_Check'
            shp_ours.at[idx_a, 'Match_Info'] = f"TopScore:{top_score:.2f} ({top_match['details']})"

    # ==========================
    # 6. 保存结果
    # ==========================
    # 注意：这里的缩进必须与 for 循环平级，表示循环结束后执行
    print(f"匹配完成，正在保存结果至: {output_path}\n")
    # 将处理后的 shp_ours (包含新添加的属性列) 保存到硬盘
    shp_ours.to_file(output_path, encoding='utf-8')


def add_city_id():
    """
        当前的build_id字段没有city属性，将其修改为city_id的形式
    :return:
    """
    # 读取shapefile文件
    # ================== 配置 ==================
    shp_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\6_add_fun_cls\add_cls_nb.shp"
    city_prefix = "NB"  # 城市缩写
    # ================== 读取 Shapefile ==================
    gdf = gpd.read_file(shp_path)
    # ================== build_id 前缀拼接 ==================
    # 确保 build_id 为字符串，避免数字类型拼接报错
    gdf["build_id"] = (city_prefix + "_" + gdf["build_id"].astype(str))
    # ================== 保存结果（可选） ==================
    out_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\6-1_add_city_id\add_cls_change_id_nb.shp"
    gdf.to_file(out_path, encoding="utf-8")
    print("build_id 前缀添加完成！")


def merge_path_all_shp():
    """
        合并文件夹下全部的shp文件
    :return:
    """
    import geopandas as gpd
    import glob
    import os
    import pandas as pd

    input_folder = r'G:\Experiment_files\Building_Dataset\Building-function-dataset\CMAB\shanghai\上海市'  # 替换为你的 shp 文件夹路径
    output_path = r'F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\test\merged_cleaned_sh.shp'  # 替换为你想保存的文件名

    shp_files = glob.glob(os.path.join(input_folder, "*.shp"))
    if not shp_files:
        print("文件夹中未找到任何 .shp 文件。")
        return
    print(f"共发现 {len(shp_files)} 个分幅文件，正在合并...")
    # 2. 读取并合并所有文件
    # 使用列表推导式读取所有文件，然后一次性 concat（比循环 append 快）
    gdfs = [gpd.read_file(f) for f in shp_files]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    print(f"合并完成。当前要素总数: {len(merged_gdf)}")
    # 3. 几何一致性过滤
    print("正在进行几何一致性去重...")
    # 步骤 A: 几何标准化
    # 解决相同形状但节点顺序、起点不同的情况
    merged_gdf['geometry'] = merged_gdf.geometry.normalize()
    # 步骤 B: 去重
    # subset=['geometry'] 确保只根据地理形状判断重复
    # 如果你想连属性也一起考虑，可以去掉 subset 或者增加属性列名
    final_gdf = merged_gdf.drop_duplicates(subset=['geometry'], keep='first')
    print(f"去重完成。保留要素总数: {len(final_gdf)}")
    print(f"共移除了 {len(merged_gdf) - len(final_gdf)} 个重复项。")
    # 4. 保存结果
    final_gdf.to_file(output_path, encoding='utf-8')
    print(f"过滤后的文件已保存至: {output_path}")


if __name__ == "__main__":

    # # nj数据处理
    # main(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\5_merge_512_shp_less_100\merge_512_nj_UTM50N.shp",
    #      r"G:\Experiment_files\Building_Dataset\Building-function-dataset\Building-level-functional-109-cities\East_Jiangsu_Nanjing\East_Jiangsu_Nanjing.shp",
    #      r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls\add_cls_nj.shp"
    #      )
    #
    # # sh数据处理
    # main(
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\5_merge_512_shp_less_100\merge_512_sh_UTM51N.shp",
    #     r"G:\Experiment_files\Building_Dataset\Building-function-dataset\Building-level-functional-109-cities\East_Shanghai_Shanghai\East_Shanghai_Shanghai.shp",
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls\add_cls_sh.shp"
    # )
    #
    # # hz数据处理
    # main(
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\5_merge_512_shp_less_100\merge_512_hz_UTM51N.shp",
    #     r"G:\Experiment_files\Building_Dataset\Building-function-dataset\Building-level-functional-109-cities\East_Zhejiang_Hangzhou\East_Zhejiang_Hangzhou.shp",
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls\add_cls_hz.shp"
    # )

    add_city_id() # 修改原始的build_id为city_id的形式

    # # sz数据处理
    # main(
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\5_merge_512_shp_less_100\merge_512_sz_UTM_50N.shp",
    #     r"G:\Experiment_files\Building_Dataset\Building-function-dataset\Building-level-functional-109-cities\East_Jiangsu_Suzhou\East_Jiangsu_Suzhou.shp",
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\6_add_fun_cls\add_cls_sz.shp"
    # )
    #
    # # nb数据处理
    # main(
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\5_merge_512_shp_less_100\merge_512_nb_UTM_51N.shp",
    #     r"G:\Experiment_files\Building_Dataset\Building-function-dataset\Building-level-functional-109-cities\East_Zhejiang_Ningbo\East_Zhejiang_Ningbo.shp",
    #     r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\6_add_fun_cls\add_cls_nb.shp"
    # )


    ########################################################## TEST ##########################################################
    # 合并文件夹中的所有shp文件，并基于要素进行去重
    # merge_path_all_shp()

    # sh-CMAB数据处理
    main(
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\5_merge_512_shp_less_100\merge_512_sh_UTM51N.shp",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\test\merged_cleaned_sh.shp",
        r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\test\sh_test.shp"
    )


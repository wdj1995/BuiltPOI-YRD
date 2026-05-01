"""
    将POI数据批量转换为矢量数据
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from convert_coordinate import gcj02_to_wgs84 # 导入坐标转换函数


def filter_city():
    """
        根据字段名将对应城市的数据筛选出来
    :return:
    """
    # 1. 读取 CSV 文件
    input_path = r"G:\Experiment_files\POIs_dataset\2023\浙江省.csv"  # 原始数据
    # df = pd.read_csv(input_path, encoding="utf-8", header=None)
    df = pd.read_csv(input_path, encoding="utf-8")
    # df.columns = [f"col_{i}" for i in range(df.shape[1])]
    # 打印字段名
    print(df.columns)
    print(len(df))

    # 2. 对某个字段进行筛选（示例：筛选字段 "category" 中值为 "学校" 的行）
    # filtered_df = df[df["col_10"] == "南京市"]
    filtered_df = df[df["cityname"] == "杭州市"]
    print(len(filtered_df))

    # # 3. 保存筛选后的数据为新的 CSV 文件
    output_path = r"G:\Experiment_files\POIs_dataset\hangzhou-2023.csv"
    filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # print("筛选完成！结果已保存到：", filtered_df)


def turn_csv_to_shapefile():
    """
        将csv文件转换为shapefile文件
    :return:
    """
    # 读取 CSV
    csv_path = r"G:\Experiment_files\POIs_dataset\南京市POI数据.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")
    # 检查 location 字段格式示例：
    # "118.88987,32.09888"
    # 按逗号拆分为 经度 lng、纬度 lat
    # df[['lng_gcj', 'lat_gcj']] = df['location'].str.split(',', expand=True)

    # 转换为 float 类型
    df['经度'] = df['经度'].astype(float)
    df['纬度'] = df['纬度'].astype(float)

    # 对每一行执行 GCJ02 → WGS84
    def convert_row(row):
        lng, lat = gcj02_to_wgs84(row['经度'], row['纬度'])
        return pd.Series({'lng_wgs': lng, 'lat_wgs': lat})

    df[['lng_wgs', 'lat_wgs']] = df.apply(convert_row, axis=1)

    # 生成点 geometry（WGS84）
    geometry = [Point(xy) for xy in zip(df['lng_wgs'], df['lat_wgs'])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # 删除临时字段（如果你不想要）
    # gdf = gdf.drop(columns=['lng_gcj', 'lat_gcj', 'lng_wgs', 'lat_wgs'])

    # 输出为 GeoJSON / Shapefile / GPKG 等都可以
    output_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\shp\南京市POI数据.shp"
    gdf.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")

    print("矢量点数据生成完成！保存于：", output_path)


def use_rs_range_filter_poi():
    """
        使用遥感影像的范围去筛选POI数据
    :return:
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box
    # 输入文件路径
    raster_path = r"G:\Experiment_files\RS_Image\Jilin_rs\merge\JL1_SH_merge_3857.tif"  # 遥感影像路径
    poi_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\1-shp-city\shanghai-2023.shp"  # POI 矢量点数据
    output_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\2-shp-rs-filter\shanghai-within-rs.shp"  # 输出的 shapefile 文件

    # 1. 读取遥感影像，获取 CRS 和范围
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        # 创建影像范围 Polygon
        raster_polygon = box(raster_bounds.left, raster_bounds.bottom,
                             raster_bounds.right, raster_bounds.top)
    print(f"Raster CRS: {raster_crs}")

    # 2. 读取 POI 矢量数据
    poi_src_crs = gpd.read_file(poi_path, rows=0).crs
    print("POI CRS:", poi_src_crs)

    # 3. 若 POI CRS 与 Raster CRS 不同，则转换
    if poi_src_crs != raster_crs:
        print("CRS 不一致，先全量读取再转换...")
        # 场景 A: 坐标系不同，必须全量读取后转换，再过滤
        poi_gdf = gpd.read_file(poi_path)
        poi_gdf = poi_gdf.to_crs(raster_crs)
        # 使用空间索引过滤 (sindex) 或 clip，比直接 within 更快
        filtered_poi = gpd.clip(poi_gdf, raster_polygon)
    else:
        print("CRS 一致，使用 bbox 参数加速读取...")
        # 场景 B: 坐标系一致，直接只读取框内的点 (极大节省内存)
        filtered_poi = gpd.read_file(poi_path, bbox=raster_polygon)

    # 3. 结果检查与保存
    if not filtered_poi.empty:
        print(f"过滤后 POI 数量: {len(filtered_poi)}")
        filtered_poi.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"已保存: {output_path}")
    else:
        print("警告: 当前影像范围内没有包含任何 POI 点，未生成输出文件。")


def add_en_type_geohash():
    import geohash
    """
        基于高德地图官方发布的中英文类别名对照表，将矢量数据中的中文类别修改为英文类别
    """
    # 读取高德地图分类标注库（中英文对照表），将中文类别转换为英文类别
    # 1. 读取 Excel
    print("正在加载对照表...")
    # gaode_df = pd.read_excel(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\search_POI\AMap_poi_code.xlsx") # 替换为你的路径
    # 高德地图原始类别对照表已经进行了人工筛选，删除了与功能分类无关的POI类
    gaode_df = pd.read_excel(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\search_POI\AMap_poi_code_filter.xlsx") # 替换为你的路径

    # 2. 新建 new_cn 字段 ："大类;中类;小类"
    # gaode_df["new_cn"] = gaode_df["大类"].str.strip() + ";" + gaode_df["中类"].str.strip() + ";" + gaode_df["小类"].str.strip()
    gaode_df["new_cn"] = gaode_df["大类"].astype(str).str.strip() + ";" + gaode_df["中类"].astype(str).str.strip()

    # 3. 新建 new_en 字段 ："Big_Category;Mid_Category"，第三类别由于类别种类过于复杂，因此被删除，只保留大类和中类
    # gaode_df["new_en"] = gaode_df["Big_Category"].astype(str) + ";" +gaode_df["Mid_Category"].astype(str) + ";" + gaode_df["Sub_Category"].astype(str)
    gaode_df["new_en"] = gaode_df["Big_Category"].astype(str).str.strip() + ";" +gaode_df["Mid_Category"].astype(str).str.strip()

    # 构建中文字段名对照字典
    mapping_dict = gaode_df.set_index("new_cn")["new_en"].to_dict()
    print("高德地图POI 中英文类别对照表 构建完成")


    # 读取shapefile文件，开始遍历每一行数据，查询对应的new_en结果并赋值给type_en字段
    print("正在读取矢量数据...")
    gdf = gpd.read_file(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\2-shp-rs-filter\hangzhou-2023-within-rs.shp")

    # 根据wgs84经纬度坐标将相同位置的点删除
    print(f"去重前剩余 {len(gdf)} 条数据")
    gdf = gdf.drop_duplicates(subset=['lng_wgs', 'lat_wgs']).reset_index(drop=True)
    print(f"去重后剩余 {len(gdf)} 条数据")

    # 映射英文分类 (使用 map 替代 apply，速度更快)
    # 假设 gdf['type'] 的格式严格匹配 '大类;中类'
    gdf["type_2lvl"] = (gdf["type"].astype(str).str.strip().str.split(";").str[:2].str.join(";"))
    gdf["type_en"] = gdf["type_2lvl"].map(mapping_dict)

    # 标记匹配状态
    gdf["info"] = gdf["type_en"].apply(lambda x: "not found" if pd.isna(x) else "ok")

    # 计算geohash
    gdf['geohash'] = gdf.apply(lambda row: geohash.encode(row['lat_wgs'], row['lng_wgs'], 9), axis=1)
    print("正在保存...")

    # 只保留匹配成功的内容
    gdf_ok = gdf[gdf["info"] == "ok"].copy()
    print(f"原始 POI 数量: {len(gdf)}")
    print(f"成功匹配 POI 数量: {len(gdf_ok)}")

    gdf_ok.to_file(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\hangzhou-2023-add-info.shp",
                encoding="utf-8")
    print("处理完成！")

def add_mid_class_type():
    import geohash
    """
        已经将AMap的类别对照表进行了修改，已经修改了对应的英文类别（mid category）
    """
    # 读取高德地图分类标注库（中英文对照表），将中文类别转换为英文类别
    # 1. 读取 Excel
    print("正在加载对照表...")
    # 高德地图原始类别对照表已经进行了人工筛选，删除了与功能分类无关的POI类
    gaode_df = pd.read_excel(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\search_POI\AMap_poi_code_filter_1225.xlsx") # 替换为最新修改的类别对照表

    # 2. 新建 new_cn 字段 ："大类;中类;小类"
    gaode_df["new_cn"] = gaode_df["大类"].astype(str).str.strip() + ";" + gaode_df["中类"].astype(str).str.strip()

    # 3. 新建 new_en 字段 ："Mid_Category"，这一字段已经修改为可被siglip模型识别的简单poi属性描述词语
    gaode_df["new_en"] = gaode_df["Mid_Category"].astype(str).str.strip()

    # 构建中文字段名对照字典
    mapping_dict = gaode_df.set_index("new_cn")["new_en"].to_dict()
    print("高德地图POI 中英文类别对照表 构建完成")

    # 读取shapefile文件，开始遍历每一行数据，查询对应的new_en结果并赋值给type_en字段
    print("正在读取矢量数据...")
    gdf = gpd.read_file(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\2-shp-rs-filter\shanghai-2023-within-rs.shp")
    # 根据wgs84经纬度坐标将相同位置的点删除
    print(f"去重前剩余 {len(gdf)} 条数据")
    gdf = gdf.drop_duplicates(subset=['lng_wgs', 'lat_wgs']).reset_index(drop=True)
    print(f"去重后剩余 {len(gdf)} 条数据")

    # 映射英文分类 (使用 map 替代 apply，速度更快)
    # 假设 gdf['type'] 的格式严格匹配 '大类;中类'
    gdf["type_2lvl"] = (gdf["type"].astype(str).str.strip().str.split(";").str[:2].str.join(";"))
    gdf["type_en"] = gdf["type_2lvl"].map(mapping_dict)

    # 标记匹配状态
    gdf["info"] = gdf["type_en"].apply(lambda x: "not found" if pd.isna(x) else "ok")

    # 计算geohash
    gdf['geohash'] = gdf.apply(lambda row: geohash.encode(row['lat_wgs'], row['lng_wgs'], 9), axis=1)
    print("正在保存...")

    # 只保留匹配成功的内容
    gdf_ok = gdf[gdf["info"] == "ok"].copy()
    print(f"原始 POI 数量: {len(gdf)}")
    print(f"成功匹配 POI 数量: {len(gdf_ok)}")

    gdf_ok.to_file(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\poi_data\3-shp-add-en-type-geohash\new_mid_class\shanghai-2023-add-mid-info.shp",
                encoding="utf-8")
    print("处理完成！")



if __name__ == '__main__':
    # filter_city()             # 根据字段名将对应城市的数据筛选出来

    # turn_csv_to_shapefile()   # 将csv文件转换为shapefile文件

    # use_rs_range_filter_poi() # 使用遥感影像的范围去筛选POI数据

    # add_en_type_geohash()     # 基于高德地图官方发布的中英文类别名对照表，将矢量数据中的中文类别修改为英文类别
    add_mid_class_type()        # 根据最新的对应的英文类别（mid category）编码POI的类别信息
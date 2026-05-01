import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm


def make_0_1_mask_tif():
    """
        现在已经获取了建筑分割shapefile文件，且对应有建筑功能标注；
        依次遍历每一张1024*1024大小的遥感影像，将其与建筑分割shapefile文件进行空间匹配；
        使用遥感影像对shapefile文件进行裁剪，并将裁剪得到的shapefile转换为二值mask图像
    :return:
    """
    # 遥感影像存放路径
    rs_folder = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\1_origin_rs_1024"
    # 建筑shapefile文件路径
    building_shp_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls"
    # 建筑mask文件路径
    building_mask_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\mask_data\1_building_0_1_mask"

    shp_files = glob.glob(os.path.join(building_shp_path, "*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"在 {building_shp_path} 中未找到任何 .shp 文件")
    print(f"正在合并 {len(shp_files)} 个Shapefile文件...")

    gdf_list = []
    base_crs = "EPSG:3857"
    target_crs = "EPSG:3857"

    # --- 1. 读取并合并 ---
    for f in tqdm(shp_files, desc="Reading Shapefiles"):
        gdf = gpd.read_file(f)
        # 判断文件坐标系是否与预定义的坐标系相同，如果不相同则进行转换
        if gdf.crs != base_crs:
            gdf = gdf.to_crs(base_crs)
        gdf_list.append(gdf)
    if not gdf_list:
        raise ValueError("未能读取任何有效的Shapefile数据。")
    # 使用pandas合并
    merged_df = pd.concat(gdf_list, ignore_index=True)
    # 转回GeoDataFrame
    merged_building_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=base_crs)
    print(f"原始合并完成。当前坐标系: {merged_building_gdf.crs}")

    # --- 2. 全局统一转换为 EPSG:3857 ---
    if merged_building_gdf.crs.to_string() != target_crs:
        print(f"正在将合并后的矢量数据从 {merged_building_gdf.crs} 转换为 {target_crs} ...")
        merged_building_gdf = merged_building_gdf.to_crs(target_crs)
    else:
        print(f"数据已经是 {target_crs}，无需转换。")


    os.makedirs(building_mask_path, exist_ok=True)
    rs_files = glob.glob(os.path.join(rs_folder, "*.tif"))
    if not rs_files:
        raise FileNotFoundError(f"在 {rs_folder} 中未找到任何 .tif 文件")
    print("开始处理遥感影像生成Mask...")

    # 开始遍历处理
    for rs_path in tqdm(rs_files, desc="Rasterizing"):
        file_name = os.path.basename(rs_path)
        with rasterio.open(rs_path) as src:
            # 获取影像元数据
            rs_meta = src.meta.copy()
            rs_transform = src.transform
            rs_crs = src.crs
            height, width = src.shape
            bounds = src.bounds  # (left, bottom, right, top)

            # --- 安全检查 ---
            # 确保影像真的是 3857，否则后续裁剪会对不上
            # 注意：rasterio读取的crs对象比较复杂，这里用简单的字符串包含判断或直接比较
            if rs_crs.to_string() != target_crs:
                # 如果你的影像虽然是3857但描述字符串略有不同，这里可能误报，可视情况注释掉
                print(f"警告: 影像 {file_name} 的坐标系 ({rs_crs}) 可能不是 {target_crs}")
                # 如果影像不是3857，这里其实应该报错或者跳过，因为我们已经把shp定死为3857了

            # --- 步骤 3: 空间筛选 (Clipping) ---
            # 利用矢量数据的空间索引，快速找出落在影像范围内的建筑
            minx, miny, maxx, maxy = bounds

            # 使用 .cx 进行相交查询 (Intersection Query)
            # 这一步非常快，因为不用做坐标转换了
            candidate_gdf = merged_building_gdf.cx[minx:maxx, miny:maxy]

            # 初始化 mask (背景全为0)
            mask_arr = np.zeros((height, width), dtype=np.uint8)

            # --- 步骤 4: 栅格化 (Rasterization) ---
            if not candidate_gdf.empty:
                # 准备几何体，所有建筑像素值设为 1
                shapes = ((geom, 1) for geom in candidate_gdf.geometry)

                # 烧录到 mask_arr
                rasterize(
                    shapes=shapes,
                    out=mask_arr,  # 直接写入上面的numpy数组
                    transform=rs_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False  # 严格模式：中心点在多边形内才算
                )

            # --- 步骤 5: 保存结果 ---
            rs_meta.update({
                "driver": "GTiff",
                "count": 1,  # 单波段
                "dtype": rasterio.uint8,
                "compress": "lzw"  # 压缩
            })

            out_path = os.path.join(building_mask_path, file_name)
            with rasterio.open(out_path, "w", **rs_meta) as dst:
                dst.write(mask_arr, 1)



def use_center_building_to_make_0_1_mask_tif():
    import os
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    from rasterio.features import rasterize
    from tqdm import tqdm

    """
        以建筑为中心进行裁剪，分别获取1024*1024大小的遥感影像和建筑分割区域
    """

    # ================= 配置区域 =================
    # 1. 单张完整的遥感影像路径 (可以是 .tif 或 .vrt)
    rs_image_path = r"G:\Experiment_files\RS_Image\Jilin_rs\merge\JL1_HZ_merge_3857.tif"

    # 2. 单个完整的建筑 Shapefile 路径
    # building_shp_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6_add_fun_cls\add_cls_sh.shp"
    # 修改为60分以上的全部数据
    building_shp_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\6-1_add_city_id\add_cls_change_id_hz.shp"

    # 3. 输出保存路径
    output_root = r"G:\Python_Project_02\paper-03"
    out_img_dir = os.path.join(output_root,  "1_rs_images")
    out_mask_dir = os.path.join(output_root, "2_building_masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 4. 参数设置
    # SAVE_NAME = "SH" # 不需要了，因为build_id已经被修改为city_id的形式了
    CROP_SIZE = 1024  # 裁剪大小
    TARGET_CRS = "EPSG:3857"  # 统一坐标系 (建议使用投影坐标系，单位为米)
    # ===========================================


    # --- 第一步：直接读取单个 Shapefile ---
    print(f">>> 正在读取建筑矢量文件: {os.path.basename(building_shp_path)} ...")
    # 直接读取文件
    gdf = gpd.read_file(building_shp_path)
    # 检查并统一坐标系
    if gdf.crs is not None and gdf.crs.to_string() != TARGET_CRS:
        print(f"检测到坐标系为 {gdf.crs}，正在转换为 {TARGET_CRS} ...")
        gdf = gdf.to_crs(TARGET_CRS)
    # 建立空间索引 (虽然 GeoPandas 会自动建立，但显式调用确保就绪)
    _ = gdf.sindex
    print(f">>> 矢量读取完成，共包含 {len(gdf)} 个建筑目标。")


    # --- 第二步：打开遥感影像并遍历裁剪 ---
    print(">>> 开始执行以建筑 Bbox 为中心的裁剪任务...")
    with rasterio.open(rs_image_path) as src:
        # 检查影像坐标系是否一致
        if src.crs.to_string() != TARGET_CRS:
            raise ValueError(f"严重错误：影像坐标系 {src.crs} 与目标 {TARGET_CRS} 不一致，无法继续！请先进行重投影。")
        # 获取影像尺寸
        src_h, src_w = src.height, src.width

        # 遍历每个建筑要素，跳过无效几何
        # 使用 tqdm 显示进度
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing"):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # -------------------------------------------------------
            # A. 计算定位中心 (Bbox Center) # 计算建筑中心点并转换为像素坐标
            # -------------------------------------------------------
            minx, miny, maxx, maxy = geom.bounds
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            # 地理坐标 -> 像素坐标
            py, px = src.index(cx, cy)

            # -------------------------------------------------------
            # B. 计算裁剪窗口 & 边缘回退 (Shift Logic)
            # -------------------------------------------------------
            # 计算裁剪窗口左上角坐标
            win_col_off = int(px - CROP_SIZE / 2)
            win_row_off = int(py - CROP_SIZE / 2)

            # 左/上边界限制 (不能小于0)
            if win_col_off < 0: win_col_off = 0
            if win_row_off < 0: win_row_off = 0

            # 右/下边界限制 (不能超出图像尺寸)
            if (win_col_off + CROP_SIZE) > src_w:
                win_col_off = src_w - CROP_SIZE
            if (win_row_off + CROP_SIZE) > src_h:
                win_row_off = src_h - CROP_SIZE

            # 再次检查：防止图像本身小于 Crop Size 的极端情况 # 检查极端情况（影像尺寸小于裁剪尺寸）
            if win_col_off < 0 or win_row_off < 0:
                continue

            # 创建裁剪窗口对象
            window = Window(col_off=win_col_off, row_off=win_row_off, width=CROP_SIZE, height=CROP_SIZE)

            # -------------------------------------------------------
            # C. 读取影像
            # -------------------------------------------------------
            # 读取窗口影像数据
            img_data = src.read(window=window)
            # 检查影像尺寸
            if img_data.shape[1] != CROP_SIZE or img_data.shape[2] != CROP_SIZE:
                continue
            # 检测当前裁剪块中是否包含黑色区域 (NoData) # 逻辑：只要发现任意一个像素在所有波段上都为 0，就视为包含无效背景，直接跳过。
            if np.any(np.all(img_data == 0, axis=0)):
                # 包含黑色区域，跳过当前建筑，不保存
                continue
            # 获取局部窗口的 Transform (用于Mask对齐)
            window_transform = src.window_transform(window)

            # -------------------------------------------------------
            # D. 生成二值 Mask
            # -------------------------------------------------------
            # 初始化全0 mask
            mask_arr = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
            # 不再搜索周围建筑，只处理当前这一栋建筑
            target_geom = row.geometry
            # 栅格化当前建筑几何体到掩膜
            if target_geom is not None:
                # 1. 直接烧录当前目标建筑
                # 这里的逻辑是：只把 target_geom 设为 1，其他所有像素（包括邻居建筑和地面）都保持 0
                rasterize(
                    shapes=[(target_geom, 1)],  # 列表里只有一个几何体
                    out=mask_arr,
                    transform=window_transform,  # 使用局部窗口坐标系
                    fill=0,  # 背景为0
                    default_value=1,
                    dtype=np.uint8,
                    all_touched=False  # 建议 False，保证只标记建筑主体中心
                )

            # 检查 Mask 是否有效
            # 如果 Mask 全是 0，说明建筑太小或栅格化失败
            # 这种情况不仅没有训练价值，还是有害的“脏数据”
            if np.sum(mask_arr) == 0:
                # print(f"Skip ID {idx}: Mask is empty (building too small or alignment issue).")
                continue

            # -------------------------------------------------------
            # E. 保存文件
            # -------------------------------------------------------
            # 强制使用 Shapefile 中的唯一 ID 列
            if 'build_id' in row:
                file_id = str(row['build_id'])
            else:
                file_id = str(idx)  # 只有万不得已才用 idx

            # 保存 Image
            img_meta = src.meta.copy()
            img_meta.update({
                "driver": "GTiff",
                "height": CROP_SIZE, "width": CROP_SIZE,
                "transform": window_transform,
                "compress": "lzw"
            })
            # with rasterio.open(os.path.join(out_img_dir, f"{SAVE_NAME}_{file_id}.tif"), "w", **img_meta) as dst:
            with rasterio.open(os.path.join(out_img_dir, f"{file_id}.tif"), "w", **img_meta) as dst:
                dst.write(img_data)

            # 保存 Mask
            mask_meta = img_meta.copy()
            mask_meta.update({"count": 1, "dtype": rasterio.uint8})
            # with rasterio.open(os.path.join(out_mask_dir, f"{SAVE_NAME}_{file_id}.tif"), "w", **mask_meta) as dst:
            with rasterio.open(os.path.join(out_mask_dir, f"{file_id}.tif"), "w", **mask_meta) as dst:
                dst.write(mask_arr, 1)
    print(">>> 全部处理完成！")


def read_tif_test():
    """
        读取mask图像，打印结果查看数组信息
    :return:
    """
    from PIL import Image
    # np.set_printoptions(threshold=np.inf)
    mask = np.array(Image.open(r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\dataset\building_labels\HZ_286.tif"))
    print(mask)
    has_one = np.any(mask == 1)
    print("是否存在像素值为 1 :", has_one)


def turn_tif_into_png():
    """
        将tif图像转换为png图像
    :return:
    """
    import os
    import numpy as np
    from PIL import Image

    tif_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\7_POI_center_rs_mask_1024\2_building_masks"
    png_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\RSRefSeg-function-seg-1202\dataset\building_labels"

    for fname in os.listdir(tif_dir):
            if not fname.lower().endswith(('.tif')):
                continue

            tif_path = os.path.join(tif_dir, fname)
            png_path = os.path.join(
                png_dir,
                os.path.splitext(fname)[0] + ".png"
            )

            # 1. 读取 tif（单波段）
            img = Image.open(tif_path)
            arr = np.array(img)

            # 强制保证是 2D
            if arr.ndim != 2:
                raise ValueError(f"{fname} 不是单波段图像，shape={arr.shape}")

            # 3. 保存 PNG
            Image.fromarray(arr, mode="L").save(png_path)

            print(f"✔ Converted: {fname} -> {png_path}")



if __name__ == "__main__":
    # make_0_1_mask_tif() #

    # use_center_building_to_make_0_1_mask_tif() # 以建筑为中心裁剪1024*1024大小的遥感影像和mask

    # read_tif_test()

    turn_tif_into_png()



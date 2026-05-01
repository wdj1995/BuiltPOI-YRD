"""
    重新制作数据集，主要步骤如下：
        1. 遥感影像切块为1024*1024大小，测试发现512效果更好，因此1024用于后续模型训练，而512负责生成建筑功能shapefile
        2. 使用开源框架（SAMpolyBuild）识别建筑轮廓
        3. 基于rs数据的地理坐标系，将识别的建筑轮廓栅格数据转换为shapefile格式
        4. 与开源建筑功能数据（Building-level-functional-109-cities）进行空间关系匹配，设计空间匹配规则。
            保证我们为识别出的建筑轮廓信息添加准确的建筑功能属性（由于开源数据集和我们使用的遥感影像不匹配，因此需要结合上述处理操作删除不匹配的建筑轮廓，保证数据的一致性）
        5. 将我们的输出重新转换为栅格数据png，进行训练
"""

import os
import rasterio
from rasterio.windows import Window
import numpy as np

############################################################### 裁剪遥感影像 ###############################################################
def clip_rs_into_1024():
    """
        将获取的遥感影像数据裁剪为1024*1024大小，并且过滤包含黑边/有云等的区域的遥感影像
    :return:
    """
    raster_path = r"G:\Experiment_files\RS_Image\Jilin_rs\merge\JL1_SZ_merge_3857.tif"  # 替换为你的输入路径
    output_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\1_origin_rs_1024"  # 替换为你的输出目录
    os.makedirs(output_dir, exist_ok=True)
    tile_size = 1024
    prefix = 'sz'
    # 严格过滤阈值
    # 0.0 表示：只要图块里有 1 个像素是空洞/黑边，就直接丢弃。
    # 建议设置为 0.0 以保证数据的绝对纯净。
    max_nodata_ratio = 0.0
    with rasterio.open(raster_path) as src:
        img_width = src.width
        img_height = src.height
        # 1. 获取影像定义的 NoData 值
        nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = 0  # 常见默认为0
            print(f"提示：影像未定义 NoData 值，默认假定 {nodata_value} 为无效背景。")
        else:
            print(f"检测到影像 NoData 值为: {nodata_value}")
        print(f"开始裁剪... 影像尺寸: {img_width} x {img_height}")
        save_count = 0
        skip_edge_count = 0
        skip_nodata_count = 0
        # 2. 标准滑动窗口循环 (步长 = tile_size，无重叠)
        for top in range(0, img_height, tile_size):
            for left in range(0, img_width, tile_size):
                # A. 物理边界检查：丢弃不足 tile_size 的边缘
                if top + tile_size > img_height or left + tile_size > img_width:
                    skip_edge_count += 1
                    continue
                # 定义窗口
                window = Window(left, top, tile_size, tile_size)
                # 读取数据 (bands, height, width)
                tile_img = src.read(window=window)
                # --- B. NoData 内容检测 ---
                # 生成掩膜：True 为无效像素
                if np.isnan(nodata_value):
                    # 处理浮点数据的 NaN
                    nodata_mask = np.isnan(tile_img)
                else:
                    # 处理整数或特定值的 NoData
                    # 使用 isclose 防止浮点数精度误差，如果是整数其实直接 == 也可以
                    nodata_mask = np.isclose(tile_img, nodata_value)
                # 只要任意一个波段是 NoData，该像素即视为无效 (压缩波段维度)
                pixel_is_nodata = np.any(nodata_mask, axis=0)
                # 计算无效像素占比
                invalid_ratio = np.sum(pixel_is_nodata) / (tile_size * tile_size)
                if invalid_ratio > max_nodata_ratio:
                    # print(f"跳过空洞区域: ({left}, {top}) - 无效占比 {invalid_ratio:.2%}")
                    skip_nodata_count += 1
                    continue
                # -------------------------
                # C. 保存有效图块
                out_path = os.path.join(output_dir, f"{prefix}_{save_count}.tif")
                profile = src.profile.copy()
                profile.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, src.transform),
                    "compress": "lzw"  # 推荐压缩
                })
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(tile_img)
                save_count += 1
                if save_count % 10 == 0:
                    print(f"已保存 {save_count} 张...", end="\r")
    print(f"\n处理结束 summary:")
    print(f"1. 成功保存 (纯净数据): {save_count} 张")
    print(f"2. 丢弃 (物理边缘不足): {skip_edge_count} 张")
    print(f"3. 丢弃 (含空洞/NoData): {skip_nodata_count} 张")

def batch_clip_rs_into_512():
    """
        对文件夹中的每一个子文件进行批量裁剪
    :return:
    """
    input_dir  = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\1_origin_rs_1024"
    output_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\2_origin_rs_512"

    tile_size = 512
    max_nodata_ratio = 0.0
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 input_dir 中所有 tif 文件
    tif_list = [f for f in os.listdir(input_dir) if f.lower().endswith(".tif")]
    print(f"共找到 {len(tif_list)} 个TIF文件，开始处理...\n")

    for tif_name in tif_list:
        raster_path = os.path.join(input_dir, tif_name)
        base_name = os.path.splitext(tif_name)[0]  # 用于输出文件命名
        print(f"\n>>> 正在处理影像：{tif_name}")

        with rasterio.open(raster_path) as src:
            img_width, img_height = src.width, src.height
            # 获取 NoData 值
            nodata_value = src.nodata
            if nodata_value is None:
                nodata_value = 0
                print(f"提示：影像未定义 NoData，默认使用 0。")

            print(f"影像尺寸：{img_width} x {img_height}")
            print(f"NoData值：{nodata_value}")

            save_count = 0
            skip_edge_count = 0
            skip_nodata_count = 0

            # 滑动窗口裁剪
            for top in range(0, img_height, tile_size):
                for left in range(0, img_width, tile_size):
                    # 边缘丢弃
                    if top + tile_size > img_height or left + tile_size > img_width:
                        skip_edge_count += 1
                        continue
                    # 读取窗口
                    window = Window(left, top, tile_size, tile_size)
                    tile_img = src.read(window=window)  # shape: (bands, H, W)

                    # --- NoData 判断 ---
                    if np.isnan(nodata_value):
                        nodata_mask = np.isnan(tile_img)
                    else:
                        nodata_mask = np.isclose(tile_img, nodata_value)

                    pixel_is_nodata = np.any(nodata_mask, axis=0)
                    invalid_ratio = np.sum(pixel_is_nodata) / (tile_size * tile_size)

                    if invalid_ratio > max_nodata_ratio:
                        skip_nodata_count += 1
                        continue

                    # --- 保存 ---
                    out_name = f"{base_name}_{save_count}.tif"
                    out_path = os.path.join(output_dir, out_name)

                    profile = src.profile.copy()
                    profile.update({
                        "height": tile_size,
                        "width": tile_size,
                        "transform": rasterio.windows.transform(window, src.transform),
                        "compress": "lzw"
                    })

                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(tile_img)

                    save_count += 1
            # 每张影像的结果统计
            print(f"\n【{tif_name}】处理完成：")
            print(f"  保存有效切片：{save_count}")
            print(f"  丢弃边缘切片：{skip_edge_count}")
            print(f"  丢弃含NoData：{skip_nodata_count}")
############################################################### 裁剪遥感影像 ###############################################################



############################################################### mask栅格合并+几何处理 ###############################################################
def merge_all_build_seg_mask():
    """
        直接将我们识别的建筑mask全部拼接成一个完整的mask大图，后续在进行过滤
    :return:
    """
    import os
    import rasterio
    from rasterio.windows import from_bounds
    import numpy as np
    from tqdm import tqdm
    from rasterio.transform import from_origin
    input_folder = r"G:\Python_Project\SAMPolyBuild-master\work_dir\rs_512\whumix_auto\geospatial_results\building_masks_tif"
    output_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data\3_merge_512"
    os.makedirs(output_dir, exist_ok=True)
    # 定义分组关键词
    keywords = ['hz', 'nj', 'sh']
    # 分类文件
    tif_files_dict = {k: [] for k in keywords}
    for f in os.listdir(input_folder):
        if f.endswith('.tif') or f.endswith('.tiff'):
            for k in keywords:
                if k in f:
                    tif_files_dict[k].append(os.path.join(input_folder, f))
    # 对每个关键字分别拼接
    for k in keywords:
        tif_files = tif_files_dict[k]
        if not tif_files:
            print(f"关键字 '{k}' 未找到对应的 TIFF 文件，跳过。")
            continue
        print(f"关键字 '{k}' 共找到 {len(tif_files)} 个影像文件，开始拼接...")
        # 计算最终大图的地理边界 (Bounding Box)
        # 初始化边界坐标变量
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        # 打开第一个文件获取基本元数据（分辨率、坐标系等）
        with rasterio.open(tif_files[0]) as first:
            src_res = first.res
            src_crs = first.crs
            src_nodata = first.nodata
            src_count = first.count
            src_dtype = first.dtypes[0]
        # 遍历所有TIFF文件，计算整体的地理边界范围
        for fp in tqdm(tif_files, desc="扫描元数据"):
            with rasterio.open(fp) as src:
                bounds = src.bounds               # 获取该影像的四个角点坐标
                min_x = min(min_x, bounds.left)   # 更新最小经度
                min_y = min(min_y, bounds.bottom) # 更新最小纬度
                max_x = max(max_x, bounds.right)  # 更新最大经度
                max_y = max(max_y, bounds.top)    # 更新最大纬度
        # 根据整体边界和分辨率计算输出图像的尺寸 # 创建输出画布
        out_width = int(round((max_x - min_x) / src_res[0]))
        out_height = int(round((max_y - min_y) / src_res[1]))
        out_transform = from_origin(min_x, max_y, src_res[0], src_res[1])
        output_path = os.path.join(output_dir, f"merge_512_{k}.tif")
        print(f"[{k}] 输出图像尺寸: {out_width} x {out_height}")
        print(f"[{k}] 输出路径: {output_path}")
        # 配置输出文件的参数
        profile = {
            'driver': 'GTiff',
            'height': out_height,
            'width': out_width,
            'count': src_count,
            'dtype': src_dtype,
            'crs': src_crs,
            'transform': out_transform,
            'compress': 'lzw',
            'bigtiff': 'YES',
            'tiled': True,
            'nodata': 0
        }
        # 创建并打开目标文件
        with rasterio.open(output_path, 'w+', **profile) as dst:
            # 遍历每个输入文件进行拼接
            for fp in tqdm(tif_files, desc="正在拼接"):
                with rasterio.open(fp) as src:
                    # 读取当前文件的数据
                    src_data = src.read(1)
                    # 精确定位每幅小影像的位置/计算小影像在大影像中的精确像素位置
                    window = from_bounds(src.bounds.left, src.bounds.bottom,
                                         src.bounds.right, src.bounds.top,
                                         transform=out_transform)
                    # 对窗口偏移进行取整，确保窗口尺寸匹配
                    window = window.round_offsets()
                    # MAX合并算法详解
                    # 读取目标位置已有的数据
                    existing_data = dst.read(1, window=window, boundless=True, fill_value=0)
                    # 处理边缘误差 # 浮点数计算地理坐标→像素坐标时可能出现±1像素的舍入误差 # 统一裁剪到最小公共尺寸，确保数据对齐
                    if existing_data.shape != src_data.shape:
                        h_min = min(existing_data.shape[0], src_data.shape[0])
                        w_min = min(existing_data.shape[1], src_data.shape[1])
                        src_data = src_data[:h_min, :w_min]
                        existing_data = existing_data[:h_min, :w_min]
                    # 执行MAX操作：保留每个像素位置的最大值
                    combined_data = np.maximum(existing_data, src_data)
                    # 将合并后的数据写入输出文件
                    dst.write(combined_data, 1, window=window)
        print("拼接完成！")


def merge_all_build_seg_mask_add_closed():
    """
        直接将我们识别的建筑mask全部拼接成一个完整的mask大图，后续在进行过滤
            新增：使用形态学闭运算对拼接后的结果进行几何修正，消除缝隙。
    :return:
    """
    import os
    import rasterio
    from rasterio.windows import from_bounds
    import numpy as np
    from tqdm import tqdm
    from rasterio.transform import from_origin
    import cv2

    input_folder = r"G:\Python_Project\SAMPolyBuild-master\work_dir\rs_sz_nb_512\whumix_auto\geospatial_results\building_masks_tif"
    output_dir = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\3_merge_512_closed"  # 修改输出目录名以区分
    os.makedirs(output_dir, exist_ok=True)

    # --- 新增：形态学处理函数 ---
    def post_process_and_save(output_path, profile):
        print(f"\n开始对 [{k}] 进行形态学闭运算修正...")
        # 1. 以读写模式打开目标文件
        with rasterio.open(output_path, 'r+') as dst:
            # 2. 读取整个图像数据 (默认读取第一个波段)
            mask_data = dst.read(1)
            # 3. 定义形态学内核 (Structuring Element)
            # 核的大小决定了连接的距离。例如，5x5 核可以连接间隔小于 5 像素的缝隙。
            # 请根据您的缝隙宽度和数据分辨率进行调整。
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # 4. 执行形态学闭运算 (Closing Operation)
            # 闭运算 = 膨胀 (Dilation) + 腐蚀 (Erosion)
            # 膨胀：连接断裂的缝隙
            # 腐蚀：平滑边界，去除膨胀引入的过多扩张
            # 将mask数据转换为8位无符号整数 (OpenCV 要求)
            mask_8bit = mask_data.astype(np.uint8)
            # 只有前景（建筑）像素为1时，闭运算才有意义，确保数据为二值
            processed_mask = cv2.morphologyEx(mask_8bit, cv2.MORPH_CLOSE, kernel)
            print(f"修正完成，使用 {kernel_size}x{kernel_size} 核。")
            # 5. 将处理后的数据写入目标文件
            # 写入前将数据类型转换回原始类型 (如 bool 或 uint8)
            dst.write(processed_mask.astype(profile['dtype']), 1)
        print(f"[{k}] 几何修正后的栅格图像已保存至: {output_path}")
    # -----------------------------

    # 定义分组关键词
    # keywords = ['hz', 'nj', 'sh']
    keywords = ['nb', 'sz']
    # 分类文件
    tif_files_dict = {k: [] for k in keywords}
    for f in os.listdir(input_folder):
        if f.endswith('.tif') or f.endswith('.tiff'):
            for k in keywords:
                if k in f:
                    tif_files_dict[k].append(os.path.join(input_folder, f))

    # 对每个关键字分别拼接
    for k in keywords:
        tif_files = tif_files_dict[k]
        if not tif_files:
            print(f"关键字 '{k}' 未找到对应的 TIFF 文件，跳过。")
            continue
        print(f"关键字 '{k}' 共找到 {len(tif_files)} 个影像文件，开始拼接...")

        # 计算最终大图的地理边界 (Bounding Box)
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        with rasterio.open(tif_files[0]) as first:
            src_res = first.res
            src_crs = first.crs
            src_nodata = first.nodata
            src_count = first.count
            src_dtype = first.dtypes[0]
        for fp in tqdm(tif_files, desc="扫描元数据"):
            with rasterio.open(fp) as src:
                bounds = src.bounds
                min_x = min(min_x, bounds.left)
                min_y = min(min_y, bounds.bottom)
                max_x = max(max_x, bounds.right)
                max_y = max(max_y, bounds.top)
        out_width = int(round((max_x - min_x) / src_res[0]))
        out_height = int(round((max_y - min_y) / src_res[1]))
        out_transform = from_origin(min_x, max_y, src_res[0], src_res[1])
        output_path = os.path.join(output_dir, f"merge_512_{k}.tif")
        print(f"[{k}] 输出图像尺寸: {out_width} x {out_height}")
        print(f"[{k}] 输出路径: {output_path}")
        profile = {
            'driver': 'GTiff',
            'height': out_height,
            'width': out_width,
            'count': src_count,
            'dtype': src_dtype,
            'crs': src_crs,
            'transform': out_transform,
            'compress': 'lzw',
            'bigtiff': 'YES',
            'tiled': True,
            'nodata': 0
        }
        # 1. 创建并打开目标文件，进行拼接写入
        with rasterio.open(output_path, 'w+', **profile) as dst:
            for fp in tqdm(tif_files, desc="正在拼接"):
                with rasterio.open(fp) as src:
                    src_data = src.read(1)
                    window = from_bounds(src.bounds.left, src.bounds.bottom,
                                         src.bounds.right, src.bounds.top,
                                         transform=out_transform)
                    window = window.round_offsets()
                    existing_data = dst.read(1, window=window, boundless=True, fill_value=0)
                    if existing_data.shape != src_data.shape:
                        h_min = min(existing_data.shape[0], src_data.shape[0])
                        w_min = min(existing_data.shape[1], src_data.shape[1])
                        src_data = src_data[:h_min, :w_min]
                        existing_data = existing_data[:h_min, :w_min]
                    combined_data = np.maximum(existing_data, src_data)
                    dst.write(combined_data, 1, window=window)
        print("拼接完成！")
        # 2. 调用后处理函数：在拼接完成后进行几何修正
        try:
            post_process_and_save(output_path, profile)
        except Exception as e:
            print(f"形态学处理失败：{e}")
            print(f"未修正的栅格图像仍保存在 {output_path}")
############################################################### mask栅格合并+几何处理 ###############################################################


############################################################### mask转shapefile ###############################################################
def turn_mask_into_shp():
    """
        将mask栅格图像转换为矢量数据
    :return:
    """
    import rasterio
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import shape
    import os

    mask_path       = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\3_merge_512_closed\merge_512_sz.tif"
    output_shp_path = r"F:\Python_Files\Python_Project_02\Text_Image_Learning\Data_source\building_attributes\merged_NJ_SH_HZ_1204\rs_data_SZ_NB\4_merge_512_shp\merge_512_sz.shp"

    # 1. 读取栅格数据
    with rasterio.open(mask_path) as src:
        # 读取第一波段
        # 注意：为了节省内存，如果你的mask只有0和1，强制指定为uint8类型
        image = src.read(1, out_dtype='uint8')
        # 获取坐标变换矩阵 (将像素坐标转换为地理坐标)
        transform = src.transform
        crs = src.crs
        print(f"影像尺寸: {src.width} x {src.height}")
        print("开始进行矢量化 (Raster to Vector)... 这可能需要一些时间")

        # 2. 生成形状生成器
        # mask参数指定了只处理哪些区域（这里我们只关心非0区域，或者你可以显式指定 image==1）
        # connectivity=4 表示4连通（上下左右），connectivity=8 表示8连通（包含对角线）
        # 通常建筑提取使用 4连通 边缘更规整，8连通 可能会让对角线像素连在一起
        mask_bool = (image == 1)

        # rasterio.features.shapes 返回一个生成器，生成 (geojson_geometry, value) 元组
        # 我们只提取 mask 为 True 的部分
        results = shapes(image, mask=mask_bool, transform=transform, connectivity=4)

        # 3. 构建几何对象列表
        geometries = []
        for geom, val in results:
            # 这里的 val 就是栅格的值（应该是1.0），我们只需要几何体
            if val == 1:
                geometries.append(shape(geom))

    print(f"共提取出 {len(geometries)} 个建筑图斑")

    # 4. 创建 GeoDataFrame
    if len(geometries) > 0:
        gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=crs)

        # (可选) 添加一个ID字段
        gdf['build_id'] = range(len(gdf))

        # 5. 保存为 Shapefile
        # 确保输出目录存在
        output_dir = os.path.dirname(output_shp_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"正在保存 Shapefile 至: {output_shp_path}")
        # encoding='utf-8' 防止中文路径或字段乱码
        gdf.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print("转换完成！")
    else:
        print("警告：未检测到任何建筑像素（值为1的区域），未生成文件。")





if __name__ == '__main__':
    # clip_rs_into_1024() # 1. 将获取的遥感影像数据裁剪为1024*1024大小，并且过滤包含黑边/有云等的区域的遥感影像
    # batch_clip_rs_into_512() # 批量将文件裁剪为512*512大小

    # merge_all_build_seg_mask() # 2. 直接将我们识别的建筑mask全部拼接成一个完整的mask大图，后续在进行过滤
    # merge_all_build_seg_mask() # 每个城市单独合并建筑mask
    merge_all_build_seg_mask_add_closed() # 在合并同类数据的基础上，添加几何修复的功能

    turn_mask_into_shp() # 3. 将mask栅格图像转换为矢量数据

import sys
sys.path.append(sys.path[0] + '/..')

import argparse
import os
import os.path as osp
from typing import List, Tuple
import mmengine
import numpy as np
from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
import torch
from mmengine.structures import PixelData
import geopandas as gpd
from shapely.geometry import box
import rasterio
import shutil


os.environ["MODELSCOPE_CACHE"] = "model_cache"
# torch.set_printoptions(
#     threshold=float('inf'),  # 打印完整内容
#     linewidth=5,             # 单行最大字符数
#     sci_mode=False           # 不使用科学计数法
# )

# =====================================================
# 自定义类别与调色板
# =====================================================
CUSTOM_CLASSES = ['Residential', 'Commercial-Industrial', 'Public service']
CUSTOM_PALETTE = [
    [255, 255, 153],    # industrial  - 橙色  # #FFFF99
    [56, 108, 176],     # residential - 蓝色  # #386CB0
    [127, 201, 127],    # public      - 绿色  # #7FC97F
]
# =====================================================


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg test with visualization and unified output')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--work-dir', help='Directory to save evaluation results')
    parser.add_argument('--out-dir', type=str, help='Root directory to save predictions and visualizations')
    parser.add_argument('--show-dir', help='Directory to save visualization images (override out/vis)')
    parser.add_argument('--wait-time', type=float, default=2, help='Display interval when showing images')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--tta', action='store_true', help='Enable test-time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--opacity', type=float, default=0.7, help='Opacity of segmentation overlay (0,1]')
    parser.add_argument('--with-labels', action='store_true', default=False, help='Show class labels on visualization')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def save_bboxes_to_shp(bboxes, labels, classes, save_path, crs, transform):
    """将 BBox 坐标从像素空间转换到地理空间，并保存为 Shapefile"""
    polygons    = []
    attr_labels = []
    attr_names  = []
    for bbox, label in zip(bboxes, labels):
        # 1. 提取像素坐标
        px_x1, px_y1, px_x2, px_y2 = bbox
        # 2. 使用仿射变换将像素坐标转换为地理坐标
        # transform * (pixel_column, pixel_row) -> (geo_x, geo_y)
        geo_x1, geo_y1 = transform * (px_x1, px_y1)
        geo_x2, geo_y2 = transform * (px_x2, px_y2)
        # 3. 创建矩形几何体 (注意：在地理空间中，y1 和 y2 的大小关系可能与像素坐标相反)
        # shapely box 参数: (minx, miny, maxx, maxy)
        minx, maxx = min(geo_x1, geo_x2), max(geo_x1, geo_x2)
        miny, maxy = min(geo_y1, geo_y2), max(geo_y1, geo_y2)
        polygons.append(box(minx, miny, maxx, maxy))
        attr_labels.append(int(label))
        attr_names.append(classes[int(label)])
    # 创建带 CRS 的 GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'class_id': attr_labels,
        'class_name': attr_names,
        'geometry': polygons
    }, crs=crs)
    # 保存为 Shapefile
    gdf.to_file(save_path, encoding='utf-8')


def init_visualizer(args, cfg) -> SegLocalVisualizer:
    """初始化可视化器，直接使用固定类别与调色板"""
    save_dir = args.show_dir or (osp.join(args.out_dir, 'vis') if args.out_dir else '')
    visualizer = SegLocalVisualizer(save_dir=save_dir, alpha=args.opacity)
    visualizer.save_dir = save_dir
    visualizer.set_dataset_meta(classes=CUSTOM_CLASSES, palette=CUSTOM_PALETTE)
    return visualizer


def save_prediction(pred_mask: np.ndarray, save_path: str, crs=None, transform=None):
    """保存预测结果（mask图，像素值为类别索引）"""
    pred_mask = pred_mask.astype(np.uint8)
    if crs and transform:
        # 使用 rasterio 保存带坐标信息的 TIF
        with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=pred_mask.shape[0],
                width=pred_mask.shape[1],
                count=1,
                dtype=pred_mask.dtype,
                crs=crs,
                transform=transform,
        ) as dst:
            dst.write(pred_mask, 1)
    else:
        # 普通保存
        img = Image.fromarray(pred_mask)
        img.save(save_path)


def visualize_and_save_results(test_results: List[Tuple[dict, SegDataSample]], visualizer: SegLocalVisualizer, args):
    """
    可视化与结果保存逻辑：
        1. 提取建筑掩码 (0:背景, 1:建筑)
        2. 提取功能类别 (0, 1, 2) 并映射到掩码区域。
        3. 保存三类文件：原始影像、建筑掩码(0/1)、功能掩码(0/1/2/3)。
    """
    if args.out_dir:
        save_mask_dir = osp.join(args.out_dir, 'masks')
        mmengine.mkdir_or_exist(save_mask_dir)

    for idx, (data_sample, pred_sample) in enumerate(test_results):
        print("data_sample：", data_sample)
        print("pred_sample：", pred_sample,)
        print(f"--- 处理样本 {idx} ---")

        # ---- Step 1. 获取基础信息 ----
        img_path  = data_sample['data_samples'][0].metainfo['img_path']
        img_pil   = Image.open(img_path).convert('RGB')
        img       = np.array(img_pil)
        img_name  = os.path.basename(img_path)
        base_name = osp.splitext(img_name)[0]

        # 处理 BBox 并保存为地理坐标系下的 SHP
        with rasterio.open(img_path) as src: # 使用 rasterio 提取地理信息
            img_crs = src.crs
            img_transform = src.transform
            img = src.read([1, 2, 3]).transpose(1, 2, 0)  # 读取 RGB 用于可视化
            # 如果是单通道或其它，这里可能需要根据实际情况调整
        seg_data_sample = data_sample['data_samples'][0]
        if (hasattr(seg_data_sample, 'gt_instances') and
                seg_data_sample.gt_instances is not None and
                len(seg_data_sample.gt_instances.bboxes) > 0):
            bboxes = seg_data_sample.gt_instances.bboxes.cpu().numpy()
            labels = seg_data_sample.gt_instances.labels.cpu().numpy()
            shp_path = osp.join(save_mask_dir, f"{base_name}_gt_bbox.shp")
            # 传入 crs 和 transform 进行转换
            save_bboxes_to_shp(bboxes, labels, CUSTOM_CLASSES, shp_path, img_crs, img_transform)
            print(f"成功导出带地理坐标的 BBox Shapefile: {shp_path}")

        # ---- Step 2. 提取建筑分割掩码 (与 predict 逻辑对齐) ----
        # 建筑分割头的输出，[2, H, W]
        building_logits     = pred_sample.pred_sem_seg.data  # Tensor: [2, H, W]
        building_mask_torch = building_logits.argmax(dim=0)  # 在类别维（dim=0）上取最大值索引，Tensor [H, W]，值 ∈ {0, 1}
        building_mask_np    = building_mask_torch.cpu().numpy().astype(np.uint8)

        # ---- Step 3. 提取功能分类标签 (与 predict 逻辑对齐) ----
        func_scores       = pred_sample.pred_logits.score     # 提取功能分类头的输出，数据结构示例：[0.1, 0.8, 0.1]
        predicted_func_id = func_scores.argmax(dim=-1).item() # 使用 argmax 得到预测的类别 ID (假设模型预测 ID 为 0, 1, 2)

        # ---- Step 4. 构造用于可视化的语义图 (用于 Visualizer) ----
        h, w        = building_mask_torch.shape                   # 获取影像空间尺寸，用于构造同尺寸语义图
        vis_seg_map = torch.full((h, w), 255, dtype=torch.uint8, device=building_mask_torch.device)
        vis_seg_map[building_mask_torch == 1] = predicted_func_id # 在建筑区域（mask == 1）内：填入预测的功能类别 ID

        # ---- Step 5. 构造用于保存的功能掩码 (0, 1, 2, 3) ----
        # 逻辑：0 代表非建筑，1/2/3 代表具体功能 (即 ID + 1)
        func_save_map                        = np.zeros((h, w), dtype=np.uint8) # 创建一个 NumPy 数组，初始化为 0，0代表非建筑区域
        func_save_map[building_mask_np == 1] = predicted_func_id + 1

        # ---- Step 6. 执行可视化 (生成带调色板和标签的预览图) ----
        vis_sample = SegDataSample()
        vis_sample.pred_sem_seg = PixelData(data=vis_seg_map.unsqueeze(0))
        visualizer.add_datasample(
            name        = img_name,
            image       = img,
            data_sample = vis_sample,
            draw_gt     = False,
            draw_pred   = True,
            wait_time   = args.wait_time,
            # out_file    = osp.join(visualizer.save_dir, img_name) if visualizer.save_dir else None,
            out_file    = None,
            with_labels = args.with_labels
        )

        # ---- Step 7. 保存原始掩码 (用于后续定量分析) ----
        if args.out_dir:
            # # 保存原始遥感影像
            # shutil.copy(img_path, osp.join(save_mask_dir, f"{base_name}_origin_rs.tif"))
            # # 建筑分割 Mask
            # save_prediction(building_mask_np,
            #                 osp.join(save_mask_dir, f"{base_name}_build_pred.tif"),
            #                 crs=img_crs, transform=img_transform)
            # 建筑功能 Mask
            save_prediction(func_save_map,
                            osp.join(save_mask_dir, f"{base_name}_func_pred.tif"),
                            crs=img_crs, transform=img_transform)
            print(f"成功保存样本 {img_name} 的地理空间预测文件。")



def main():
    args = parse_args()

    if not osp.isfile(args.checkpoint):
        print("\n" + "=" * 50)
        print(f"错误：找不到指定的权重文件！")
        print(f"尝试加载的路径: {args.checkpoint}")
        print("请检查 --checkpoint 参数是否正确。")
        print("=" * 50 + "\n")
        sys.exit(1)  # 终止程序
    else:
        print(f"成功找到权重文件: {args.checkpoint}")

    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir') is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 只有在文件存在时才设置
    cfg.load_from = args.checkpoint

    # 构建 Runner（自动加载 test pipeline）
    runner = Runner.from_cfg(cfg)

    # 初始化可视化器
    visualizer = init_visualizer(args, cfg)
    print(f"使用类别:   {visualizer.dataset_meta['classes']}")
    print(f"使用调色板: {visualizer.dataset_meta['palette']}")

    print("开始推理...")
    # 构建 dataloader
    data_loader = runner.build_dataloader(cfg.test_dataloader)

    # 构建模型
    model = runner.model
    model.eval()

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"模型已加载到设备: {device}")

    for data_batch in data_loader:
        with torch.no_grad():
            results = model.test_step(data_batch)  # list[SegDataSample], 长度 B

        batch_inputs = data_batch['inputs']
        batch_metainfo = data_batch['data_samples']

        # 重新构建一一对应的列表
        batch_results_to_visualize = []
        for i in range(len(results)):
            # 为第 i 张图创建一个独立的 data_sample 字典
            single_data_sample_dict = {
                'inputs': [batch_inputs[i]],  # 关键：使用第 i 张图的输入
                'data_samples': [batch_metainfo[i]]  # 关键：使用第 i 张图的元数据
            }
            pred_sample = results[i]  # 第 i 个预测结果
            # (输入, 预测) 配对
            batch_results_to_visualize.append((single_data_sample_dict, pred_sample))
        # 将配对好的列表传入
        visualize_and_save_results(batch_results_to_visualize, visualizer, args)
    print("推理与可视化完成！")



if __name__ == '__main__':
    main()
from typing import Optional, Sequence, Dict
import torch
from mmengine import mkdir_or_exist, MMLogger, print_log
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
from mmseg.registry import METRICS

@METRICS.register_module()
class RefSegClsMetric(BaseMetric):
    default_prefix = 'RefSegClsMetric'

    def __init__(self,
                 num_classes: int          = 3,
                 ignore_index: int         = 255,
                 iou_threshold: float      = 0.8,
                 collect_device: str       = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.iou_threshold = iou_threshold
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def process(self, data_batch: Dict, data_samples: Sequence[Dict]) -> None:
        for data_sample in data_samples:
            # --- Task 1: 建筑分割 (像素级统计) ---
            # 1. 提取 GT 和 Pred (确保维度为 [H, W])
            gt_seg = data_sample['gt_building_seg']['data'].squeeze(0)
            pred_logits = data_sample['pred_sem_seg']['data']
            # argmax 得到类别索引 index (通常 0 为背景，1 为建筑)
            pred_seg = pred_logits.argmax(dim=0).to(gt_seg.device)

            # 2. 建立 显式布尔掩码
            # 必须先将 index 转换为 bool mask，再进行逻辑运算
            gt_mask = (gt_seg == 1)
            pred_mask = (pred_seg == 1)

            # 3. 处理忽略区域 (ignore_index)
            # 仅在 non-ignore 区域内计算指标，防止背景噪声干扰
            valid_mask = (gt_seg != self.ignore_index)
            gt_mask_valid = gt_mask & valid_mask
            pred_mask_valid = pred_mask & valid_mask

            # 4. 计算交集、并集和面积
            # 使用逻辑与 (&) 和逻辑或 (|) 在布尔张量上进行计算
            intersect = (pred_mask_valid & gt_mask_valid).sum().cpu()
            union = (pred_mask_valid | gt_mask_valid).sum().cpu()
            pred_area = pred_mask_valid.sum().cpu()  # 预测出的建筑像素数
            gt_area = gt_mask_valid.sum().cpu()  # 真实的建筑像素数

            # 5. 计算单样本 IoU (用于后续 High-IoU 过滤统计)
            sample_iou = intersect / (union + 1e-10)


            # --- Task 2: 功能分类 (实例级统计) ---
            gt_label = data_sample['gt_label']['label'].view(-1)
            if 'pred_logits' in data_sample:
                # 注意：score 可能是 [C] 或 [1, C]，指定 dim=-1 更安全
                pred_label = data_sample['pred_logits']['score'].argmax(dim=-1).view(-1)
            else:
                pred_label = data_sample['pred_label']['label'].view(-1)

            self.results.append({
                'seg_intersect': intersect,
                'seg_union': union,
                'seg_pred_area': pred_area,
                'seg_gt_area': gt_area,
                'seg_iou': sample_iou.item(),  # 存入 float
                'pred_label': pred_label.cpu(),
                'gt_label': gt_label.cpu()
            })

    def _calculate_cls_metrics(self, all_pred: torch.Tensor, all_gt: torch.Tensor) -> Dict:
        """内部辅助函数：计算分类核心指标"""
        if all_gt.numel() == 0: return None

        valid_mask = (all_gt >= 0) & (all_gt < self.num_classes) & (all_gt != self.ignore_index)
        all_gt, all_pred = all_gt[valid_mask], all_pred[valid_mask]
        if all_gt.numel() == 0: return None

        mat = self.confusion_matrix(all_pred, all_gt, self.num_classes)
        TP = torch.diag(mat)
        FP, FN = mat.sum(dim=0) - TP, mat.sum(dim=1) - TP
        support = mat.sum(dim=1)

        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        total_support = support.sum()
        return {
            'macro_f1': f1.mean().item() * 100,
            'macro_precision': precision.mean().item() * 100,
            'macro_recall': recall.mean().item() * 100,
            'weighted_f1': ((f1 * support).sum() / total_support).item() * 100,
            'overall_acc': (TP.sum() / total_support).item() * 100,
            'f1_per_class': f1,
            'prec_per_class': precision,
            'rec_per_class': recall,
            'support_per_class': support
        }

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        # --- 1. 汇总分割指标 (建筑提取质量) ---
        t_int = sum([item['seg_intersect'] for item in results])
        t_uni = sum([item['seg_union'] for item in results])
        t_pred = sum([item['seg_pred_area'] for item in results])
        t_gt = sum([item['seg_gt_area'] for item in results])

        seg_iou  = (t_int / (t_uni + 1e-10)) * 100.0
        seg_dice = (2 * t_int / (t_pred + t_gt + 1e-10)) * 100.0
        seg_prec = (t_int / (t_pred + 1e-10)) * 100.0
        seg_rec  = (t_int / (t_gt + 1e-10)) * 100.0

        # --- 2. 汇总分类指标 (功能推理质量) ---
        all_gt = torch.cat([item['gt_label'] for item in results])
        all_pred = torch.cat([item['pred_label'] for item in results])
        overall = self._calculate_cls_metrics(all_pred, all_gt)

        # 计算混淆矩阵，定位 Macro F1
        conf_mat = self.confusion_matrix(all_pred, all_gt, self.num_classes)

        # 高 IoU 样本下的分类表现 (验证分割精度对推理的支撑作用)
        high_iou_items = [item for item in results if item['seg_iou'] > self.iou_threshold]
        high_iou = self._calculate_cls_metrics(
            torch.cat([item['pred_label'] for item in high_iou_items]),
            torch.cat([item['gt_label'] for item in high_iou_items])
        ) if high_iou_items else None

        # --- 3. 报表打印 ---
        class_names = self.dataset_meta.get('function_classes', [f'Class_{i}' for i in range(self.num_classes)])

        # 3.1 打印详细分类指标表
        table = PrettyTable()
        table.field_names = ['Class Name', 'F1 (%)', 'Prec (%)', 'Rec (%)', 'Support']
        for i in range(self.num_classes):
            table.add_row([class_names[i],
                           f"{overall['f1_per_class'][i] * 100:.2f}",
                           f"{overall['prec_per_class'][i] * 100:.2f}",
                           f"{overall['rec_per_class'][i] * 100:.2f}",
                           int(overall['support_per_class'][i])])
        # 3.2 打印混淆矩阵 (诊断：谁被误判成了谁？)
        # 修改原因：这是解决类间歧义（如办公 vs 商业）的最直接工具
        conf_table = PrettyTable(['(Pred \ GT)'] + class_names)
        for i in range(self.num_classes):
            conf_table.add_row([class_names[i]] + [int(x) for x in conf_mat[i]])

        # 4. 日志输出
        print_log("\n" + "=" * 85, logger)
        print_log(f" [Task 1 - Building Segmentation]", logger)
        print_log(
            f"  >> mIoU: {seg_iou:.2f}% | Dice: {seg_dice:.2f}% | Prec: {seg_prec:.2f}% | Rec: {seg_rec:.2f}%",
            logger)
        print_log("-" * 85, logger)

        print_log(f" [Task 2 - Function Reasoning (Overall)]", logger)
        print_log(f"  >> Overall Acc: {overall['overall_acc']:.2f}% | Macro F1: {overall['macro_f1']:.2f}% | Weighted F1: {overall['weighted_f1']:.2f}%", logger)
        print_log(f"  >> Macro Prec: {overall['macro_precision']:.2f}% | Macro Rec: {overall['macro_recall']:.2f}%", logger)

        if high_iou:
            print_log(f" [Analysis] High-IoU (>0.8) Macro F1: {high_iou['macro_f1']:.2f}%", logger)
            if high_iou['macro_f1'] > overall['macro_f1'] + 2:
                print_log(f"  * 分割精度显著影响了分类效果，建议进一步优化 SAM 分割质量。", logger)

        print_log("-" * 85, logger)
        print_log(f" Class-wise Performance:\n{table.get_string()}", logger)
        print_log("-" * 85, logger)
        print_log(f" Confusion Matrix (Vertical: Pred, Horizontal: GT):\n{conf_table.get_string()}", logger)

        # 诊断性警告：识别 Macro F1 的杀手类别
        low_f1_classes = [class_names[i] for i, f1 in enumerate(overall['f1_per_class']) if f1 < 0.4]
        if low_f1_classes:
            print_log(f" [Warning] Macro F1 瓶颈识别: {', '.join(low_f1_classes)} 的 F1 分数低于 40%。", logger)
            print_log(f"           建议检查对应功能的 POI 候选质量或增加类别权重。", logger)

        print_log("=" * 85 + "\n", logger)

        return {
            'Building_IoU': seg_iou,
            'Building_Dice': seg_dice,
            'Function_F1_Macro': overall['macro_f1'],
            'Function_F1_Weighted': overall['weighted_f1'],
            'Function_Acc': overall['overall_acc']
        }

    @staticmethod
    def confusion_matrix(preds, gts, num_classes):
        inds = num_classes * gts.long() + preds.long()
        counts = torch.bincount(inds, minlength=num_classes ** 2)
        return counts.reshape(num_classes, num_classes).float()




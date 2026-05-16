from evaluate import evaluate_model

"""    
    基于mask创建外扩bbox并对遥感影像进行裁剪
        test loss 0.7707 | OA 0.6683 | macro_f1 0.6443  
"""


if __name__ == "__main__":
    # 默认加载训练阶段保存的最优模型，并在测试集上计算 loss、OA 和 macro F1。
    evaluate_model(split="test", checkpoint_path="outputs/best_model.pth")

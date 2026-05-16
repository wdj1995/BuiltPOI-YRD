from evaluate import evaluate_model

"""
    1. 全部冻结
        test loss 0.8263 | OA 0.6390 | macro_f1 0.6079
    2. 冻结stage4：
        test loss 0.8363 | OA 0.6282 | macro_f1 0.5998
        提升早停epoch设置：
        test loss 0.8363 | OA 0.6282 | macro_f1 0.5998
    
    
    3. 只将基于mask生成的外扩bbox区域输入resnet中
        test loss 0.7562 | OA 0.6675 | macro_f1 0.6432
        
"""

if __name__ == "__main__":
    # 默认加载训练阶段保存的最优模型，在测试集上计算 loss、OA 和 macro F1。
    evaluate_model(split="test", checkpoint_path="outputs/best_model.pth")

from evaluate import evaluate_model

"""
    冻结bert前10层：
        test loss 0.8394 | OA 0.6514 | macro_f1 0.6353
        
    冻结bert前12层：
        test loss 0.7698 | OA 0.6564 | macro_f1 0.6302
        
    冻结bert前6层：
        test loss 0.7727 | OA 0.6662 | macro_f1 0.6426
        
    对BERT进行完全微调：
        test loss 0.7496 | OA 0.6744 | macro_f1 0.6495
    
"""


if __name__ == "__main__":
    # 默认加载训练阶段保存的最优模型，并在测试集上计算 loss、OA 和 macro F1。
    evaluate_model(split="test", checkpoint_path="outputs/best_model.pth")

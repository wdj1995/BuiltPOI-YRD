"""
    基于预处理后的特征（视觉手工提取特征 + tf-idf特征）进行 SVM 分类
    功能：加载 npz 数据集、超参数搜索 (C, gamma)、测试集评估 (OA, F1)、保存模型与报告
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # 补全缺失的导入
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def train_and_evaluate_svm():
    """
        Test OA : 0.6826
        Test F1 : 0.6582

        分类报告:
                   precision    recall  f1-score   support

               0       0.78      0.83      0.80      1600
               1       0.55      0.56      0.56       914
               2       0.64      0.58      0.61      1270

        accuracy                           0.68      3784
       macro avg       0.66      0.66      0.66      3784
    weighted avg       0.68      0.68      0.68      3784

    :return:
    """


    # --- 1. 路径配置 ---
    data_input_path = "./process_data"
    save_folder     = "./save_model"
    os.makedirs(save_folder, exist_ok=True) # 创建模型保存文件夹（若不存在）

    # --- 2. 加载数据 ---
    print("正在加载预处理后的数据集...")
    data = np.load(os.path.join(data_input_path, "final_dataset.npz")) # 读取npz压缩包

    X_train, y_train = data["X_train"], data["y_train"] # 训练集特征与标签
    X_val, y_val     = data["X_val"],   data["y_val"]   # 验证集特征与标签
    X_test, y_test   = data["X_test"],  data["y_test"]  # 测试集特征与标签

    # 加载标签编码器以获取原始类别名称
    le = joblib.load(os.path.join(data_input_path, "label_encoder.pkl"))
    print(f"训练集规模: {X_train.shape} | 验证集规模: {X_val.shape} | 测试集规模: {X_test.shape}")


    # --- 3. 合并数据用于交叉验证 ---
    print("\n合并 Train + Val 用于交叉验证训练...")
    X_trainval = np.concatenate([X_train, X_val], axis=0) # 合并特征矩阵
    y_trainval = np.concatenate([y_train, y_val], axis=0) # 合并标签向量
    print(f"合并后训练规模: {X_trainval.shape}")


    # --- 4. 定义模型与参数空间 ---
    svm_model = SVC(
        kernel       = "rbf",
        class_weight = "balanced", # 应对样本不平衡
        cache_size   = 2000,
        random_state = 42
    )

    # 定义网格搜索的参数字典
    param_grid = {
        "C"     : [0.01, 0.1, 1, 10, 100],         # 惩罚系数
        "gamma" : ["scale", 1, 0.1, 0.01, 0.001]   # 核函数系数
    }


    # --- 5. 配置网格搜索 ---
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5折分层交叉验证
    # 遍历组合
    grid_search = GridSearchCV(
        estimator   = svm_model,
        param_grid  = param_grid,
        scoring     = "f1_macro",  # 使用宏平均 F1 值作为评价指标
        cv          = cv_strategy, # 交叉验证策略
        n_jobs      = 6,           # 调用所有 CPU 核心并行计算
        verbose     = 2,           # 输出详细进度信息
        refit       = True         # 搜索结束后用全量数据重新拟合最佳模型
    )
    print("\n" + "="*35 + "\n开始 GridSearchCV 超参数搜索\n" + "="*35)
    grid_search.fit(X_trainval, y_trainval) # 执行网格搜索


    # --- 6. 获取搜索结果 ---
    best_model    = grid_search.best_estimator_ # 提取最佳模型
    best_params   = grid_search.best_params_    # 提取最佳参数
    best_cv_score = grid_search.best_score_     # 提取最佳验证得分
    print(f"\n最佳参数: {best_params}")
    print(f"最佳 CV Macro-F1: {best_cv_score:.4f}")


    # --- 7. 保存最佳模型 ---
    model_save_path = os.path.join(save_folder, "best_svm_model.pkl")
    joblib.dump(best_model, model_save_path)     # 序列化保存模型
    print(f"\n最佳模型已保存至: {model_save_path}")



    # # --- 直接加载已有的模型 ---
    # model_save_path = os.path.join(save_folder, "best_svm_model.pkl")
    # if os.path.exists(model_save_path):
    #     print(f"正在从 {model_save_path} 加载已训练模型...")
    #     best_model = joblib.load(model_save_path)
    # else:
    #     raise FileNotFoundError("未找到模型文件，请先运行训练程序！")



    # --- 8. 测试集最终评估 ---
    print("\n正在进行 Test 集最终评估...")
    test_pred = best_model.predict(X_test)                   # 对测试集进行预测
    test_oa   = accuracy_score(y_test, test_pred)            # 计算总准确率(OA)
    test_f1   = f1_score(y_test, test_pred, average="macro") # 计算 Macro-F1
    print(f"Test OA : {test_oa:.4f}")
    print(f"Test F1 : {test_f1:.4f}")

    # 生成分类报告（Precision, Recall, F1-score for each class）
    report = classification_report(
        y_test, test_pred,
        labels=np.arange(len(le.classes_)),
        target_names=[str(c) for c in le.classes_]
    )
    print("\n分类报告:\n", report)


    # --- 9. 保存评估报告与混淆矩阵 ---
    report_save_path = os.path.join(save_folder, "classification_report.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Macro-F1: {best_cv_score:.4f}\n")
        f.write(f"Test OA : {test_oa:.4f}\n")
        f.write(f"Test F1 : {test_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 保存混淆矩阵原始数据
    cm = confusion_matrix(y_test, test_pred)
    np.save(os.path.join(save_folder, "confusion_matrix.npy"), cm)
    print("\n所有结果已保存完成。")




if __name__ == "__main__":
    train_and_evaluate_svm()
import numpy as np
import pandas as pd
import csv
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import matthews_corrcoef, f1_score, roc_curve, auc, precision_score, recall_score
from imblearn.under_sampling import OneSidedSelection
from sklearn.neighbors import NearestNeighbors  # 为RSDS和CLNI-KNN导入

warnings.filterwarnings('ignore')


def get_if_results(X_train, y_train, X_test, y_test, seed):
    """
    方法: Isolation Forest (IF)

    """
    # 使用 IF 识别并删除异常点
    iso = IsolationForest(random_state=0).fit(X_train)
    x_array = iso.predict(X_train)
    de1 = [j for j, val in enumerate(x_array) if val == -1]
    x_pruned = np.delete(X_train, de1, axis=0)
    s_pruned = np.delete(y_train, de1, axis=0)

    # 在清洗后的数据上训练并评估
    if len(x_pruned) == 0:
        return 0, 0, 0, 0, 0, 0  # 如果数据被清空，所有指标为0

    log_reg1 = RandomForestClassifier(random_state=seed)
    log_reg1.fit(x_pruned, s_pruned)
    pre1 = log_reg1.predict(X_test)
    return calculate_all_metrics(y_test, pre1, X_test)


def get_clni_knn_results(X_train, y_train, X_test, y_test, seed):
    """
    方法: CLNI-KNN (Custom KNN Noise Identification)

    """
    # 自定义KNN噪声识别逻辑
    ordered_label_errors = []
    for ii in range(len(y_train)):
        distances = np.sqrt(np.sum((X_train - X_train[ii]) ** 2, axis=1))
        # 获取除自身外的5个最近邻
        sorted_indices = np.argsort(distances)[1:6]
        neighbor_labels = y_train[sorted_indices]

        # 投票决定是否为噪声
        if np.sum(neighbor_labels.ravel() != y_train[ii]) / 5 >= 0.6:
            ordered_label_errors.append(False)  # 是噪声
        else:
            ordered_label_errors.append(True)  # 不是噪声

    x_pruned = X_train[ordered_label_errors]
    s_pruned = y_train[ordered_label_errors]

    # 在清洗后的数据上训练并评估
    if len(x_pruned) == 0:
        return 0, 0, 0, 0, 0, 0

    log_reg = RandomForestClassifier(random_state=seed)
    log_reg.fit(x_pruned, s_pruned)
    pre1 = log_reg.predict(X_test)
    return calculate_all_metrics(y_test, pre1, X_test)


def get_oss_results(X_train, y_train, X_test, y_test, seed):
    """
    方法: One-Sided Selection (OSS)

    """
    # 使用 OSS 清洗数据
    oss = OneSidedSelection(random_state=seed)  # 保持与原逻辑一致，使用循环的seed
    x_pruned, s_pruned = oss.fit_resample(X_train, y_train)

    # 在清洗后的数据上训练并评估
    log_reg1 = RandomForestClassifier(random_state=seed)
    log_reg1.fit(x_pruned, s_pruned)
    pre1 = log_reg1.predict(X_test)
    return calculate_all_metrics(y_test, pre1, X_test)


def calculate_all_metrics(y_test, pre1, X_test):
    y_test_1 = y_test.ravel()
    f1 = f1_score(y_test_1, pre1, pos_label=1, average="binary", zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(y_test_1, pre1)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        roc_auc = 0.5
    prec = precision_score(y_test_1, pre1, pos_label=1, zero_division=0)
    recall = recall_score(y_test_1, pre1, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_test_1, pre1)

    loc_all = np.sum(X_test[:, 4]) + np.sum(X_test[:, 5])
    loc_20 = loc_all * 0.2

    count_t1 = np.sum(y_test_1 == 1)
    if count_t1 == 0:
        return f1, roc_auc, prec, recall, mcc, 0.0

    r_pre1 = np.c_[(X_test[:, 4] + X_test[:, 5]), pre1, y_test_1]
    idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
    sorted_data = r_pre1[idex, :]

    t_loc = 0
    il = 0
    count_effort = 0
    while il < len(sorted_data) and t_loc < loc_20:
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort += 1
        il += 1
    effort = count_effort / count_t1

    return f1, roc_auc, prec, recall, mcc, effort


def run_combined_experiments():
    csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout',
                 7: 'openjpa', 8: 'pig', 9: 'tuscany'}
    seeds = [94733, 16588, 1761, 59345, 27886, 80894, 22367, 65435, 96636, 89300]

    methods = {
        "IF": get_if_results,
        "CLNI-KNN": get_clni_knn_results,
        "OSS": get_oss_results,
    }
    metrics_names = ["f1", "auc", "prec", "recall", "mcc", "effort"]

    output_filename = 'baselines_1014.csv'
    with open(output_filename, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        header = ["dataset"]
        for method_name in methods.keys():
            for metric_name in metrics_names:
                header.append(f"{metric_name}-{method_name}")
        csv_writer.writerow(header)

        for i in range(10):
            dataset_name = csv_order[i]
            print(f"\n--- 正在处理数据集: {dataset_name} ---")
            dataframe = pd.read_csv('dataset/' + dataset_name + '.csv')
            data = np.array(dataframe)
            X, y = data[:, 0:14], data[:, -1].reshape(-1, 1)

            # 严格遵循原始的双重循环平均逻辑
            # 外层字典: 存储每个方法在10次重复中的平均结果
            results_per_seed = {name: {metric: [] for metric in metrics_names} for name in methods.keys()}

            for ix, seed in enumerate(seeds):
                print(f"  重复轮次 {ix + 1}/{len(seeds)} (seed: {seed})...")
                kf = KFold(n_splits=10, shuffle=True, random_state=seed)

                # 内层字典: 存储每个方法在10折中的结果
                results_per_fold = {name: {metric: [] for metric in metrics_names} for name in methods.keys()}

                for train_idx, test_idx in kf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    for method_name, method_func in methods.items():
                        metrics_values = method_func(X_train, y_train, X_test, y_test, seed)
                        for i_metric, metric_name in enumerate(metrics_names):
                            results_per_fold[method_name][metric_name].append(metrics_values[i_metric])

                # 计算10折的平均值，并存入外层列表
                for method_name in methods.keys():
                    for metric_name in metrics_names:
                        mean_fold_val = np.mean(results_per_fold[method_name][metric_name])
                        results_per_seed[method_name][metric_name].append(mean_fold_val)

            # 计算10次重复的最终平均值
            final_row = [dataset_name]
            for method_name in methods.keys():
                for metric_name in metrics_names:
                    final_mean_value = np.mean(results_per_seed[method_name][metric_name])
                    final_row.append(final_mean_value)

            csv_writer.writerow(final_row)
            print(f"--- 数据集 {dataset_name} 处理完成! ---")

    print(f"\n所有实验已完成。结果已保存至 '{output_filename}'")


if __name__ == '__main__':
    run_combined_experiments()
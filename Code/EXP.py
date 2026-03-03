import numpy as np
import pandas as pd
import copy
import random
import xlwt
import csv
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from cleanlab.classification import LearningWithNoisyLabels
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.under_sampling import RandomUnderSampler
from cleanlab.pruning import get_noise_indices
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


def get_noise(label_new,pre_new,xall_new,X_test,y_test,seed_t):
    label_new1=copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)
    xall_new1 = copy.deepcopy(xall_new)
    XX = copy.deepcopy(X_test)
    y_test_no = copy.deepcopy(y_test)



    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    confident_joint = compute_confident_joint(
        s=y_train2,
        psx=pre_new1,  # P(s = k|x)
        thresholds=None
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=y_train2,
        py_method='cnt',
        converge_latent_estimates=False,
    )

    ordered_label_errors = get_noise_indices(
        s=y_train2,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_noise_rate',
    )
    # print(ordered_label_errors)

    x_mask = ~ordered_label_errors
    x_pruned = xall_new1[x_mask]
    # print(label_new_2)
    s_pruned = y_train2[x_mask]

    sample_weight = np.ones(np.shape(s_pruned))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix[k][k]
        sample_weight[s_pruned == k] = sample_weight_k

    log_reg = RandomForestClassifier(random_state=seed_t)
    # log_reg1 = LogisticRegression(solver='liblinear')
    log_reg.fit(x_pruned, s_pruned, sample_weight=sample_weight)
    pre1 = log_reg.predict(XX)
    y_test_11 = y_test_no.ravel()
    y_original = metrics.f1_score(y_test_11, pre1, pos_label=1, average="binary")

    fpr, tpr, thersholds = metrics.roc_curve(y_test_11, pre1)
    roc_auc = metrics.auc(fpr, tpr)
    prec = metrics.precision_score(y_test_11, pre1, pos_label=1)  # 精确率
    recall = metrics.recall_score(y_test_11, pre1, pos_label=1)  # 召回率

    mcc = matthews_corrcoef(y_test_11, pre1)

    loc_all = sum(XX[:, 4]) + sum(XX[:, 5])
    loc_20 = loc_all * 0.2
    r_pre1 = np.zeros((len(pre1), 3))
    count_t1 = 0
    for il in range(len(pre1)):
        r_pre1[il][0] = (XX[il][4] + XX[il][5])
        r_pre1[il][1] = pre1[il]
        r_pre1[il][2] = y_test[il]
        if y_test[il] == 1:
            count_t1 += 1
    idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
    sorted_data = r_pre1[idex, :]
    t_loc = 0
    il = 0
    count_effort = 0
    while t_loc < loc_20:
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort = count_effort + 1
        il = il + 1
    eff = count_effort / count_t1
    return y_original, roc_auc, prec, recall, mcc, eff

def dual_confident_learning(label_new, psx_A, psx_B, xall_new, X_test, y_test, seed_t):

    label_new1 = copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(psx_A)  # 网络A的预测概率
    pre_new2 = copy.deepcopy(psx_B)  # 网络B的预测概率
    xall_new1 = copy.deepcopy(xall_new)
    XX = copy.deepcopy(X_test)
    y_test_no = copy.deepcopy(y_test)
    
    label_1 = label_new1.ravel()
    label_1 = label_1.astype(np.int32)
    # 网络A的置信学习过程
    confident_joint_A = compute_confident_joint(
        s=label_1,
        psx=pre_new1,
        thresholds=None
    )
    
    py_A, noise_matrix_A, inv_noise_matrix_A = estimate_latent(
        confident_joint=confident_joint_A,
        s=label_1,
        py_method='cnt',
        converge_latent_estimates=False,
    )
    
    noise_idx_A = get_noise_indices(
        s=label_1,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix_A,
        confident_joint=confident_joint_A,
        prune_method='prune_by_noise_rate',
    )
    
    # 网络B的置信学习过程
    confident_joint_B = compute_confident_joint(
        s=label_1,
        psx=pre_new2,
        thresholds=None
    )
    
    py_B, noise_matrix_B, inv_noise_matrix_B = estimate_latent(
        confident_joint=confident_joint_B,
        s=label_1,
        py_method='cnt',
        converge_latent_estimates=False,
    )
    
    noise_idx_B = get_noise_indices(
        s=label_1,
        psx=pre_new2,
        inverse_noise_matrix=inv_noise_matrix_B,
        confident_joint=confident_joint_B,
        prune_method='prune_by_noise_rate',
    )
    
    mask_A = ~noise_idx_B  # B筛选的干净样本给A
    mask_B = ~noise_idx_A  # A筛选的干净样本给B
    
    
    xall_new_A = xall_new1[mask_A]
    label_new_A = label_1[mask_A]
    
    xall_new_B = xall_new1[mask_B]
    label_new_B = label_1[mask_B]
    
    # 为高置信样本分配权重
    sample_weight_A = np.ones(np.shape(label_new_A))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix_B[k][k]
        sample_weight_A[label_new_A == k] = sample_weight_k
    
    sample_weight_B = np.ones(np.shape(label_new_B))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix_A[k][k]
        sample_weight_B[label_new_B == k] = sample_weight_k
    
    # 半监督学习扩展
    n_unlabeled = min(int(len(xall_new_A) * 0.2), len(XX))
    if n_unlabeled > 0:
    
        np.random.seed(seed_t)
        unlabeled_indices = np.random.choice(len(XX), n_unlabeled, replace=False)
        X_unlabeled = XX[unlabeled_indices]
        

        clf_A_temp = RandomForestClassifier(random_state=seed_t)
        clf_A_temp.fit(xall_new_A, label_new_A, sample_weight=sample_weight_A)
        
        clf_B_temp = RandomForestClassifier(random_state=seed_t+1)
        clf_B_temp.fit(xall_new_B, label_new_B, sample_weight=sample_weight_B)
        
    
        pseudo_probs_A = clf_A_temp.predict_proba(X_unlabeled)
        pseudo_probs_B = clf_B_temp.predict_proba(X_unlabeled)
        
       
        T = 0.5
        sharpened_A = np.zeros_like(pseudo_probs_A)
        sharpened_B = np.zeros_like(pseudo_probs_B)
        
        for i in range(len(pseudo_probs_A)):
            for j in range(pseudo_probs_A.shape[1]):
                sharpened_A[i,j] = pseudo_probs_A[i,j] ** (1/T)
                sharpened_B[i,j] = pseudo_probs_B[i,j] ** (1/T)
            
            sharpened_A[i] = sharpened_A[i] / np.sum(sharpened_A[i])
            sharpened_B[i] = sharpened_B[i] / np.sum(sharpened_B[i])
    
        pseudo_y_A = np.argmax(sharpened_A, axis=1)
        pseudo_y_B = np.argmax(sharpened_B, axis=1)
        
   
        w = 0.7  # 原始标签权重
        w_clean = 0.5  # 清洁样本权重

        # 获取模型A和B对各自训练样本的预测概率
        probs_A = clf_A_temp.predict_proba(xall_new_A)
        probs_B = clf_B_temp.predict_proba(xall_new_B)

        refined_labels_A = np.copy(label_new_A)
        refined_labels_B = np.copy(label_new_B)

    
        for i in range(len(label_new_A)):
            class_idx = int(label_new_A[i])
            pred_prob = probs_A[i][class_idx]
            
       
            weight = w if pred_prob > 0.6 else w_clean
  
            one_hot = np.zeros(2)
            one_hot[class_idx] = 1
            
            # 加权组合
            combined = weight * one_hot + (1-weight) * probs_A[i]
            refined_labels_A[i] = np.argmax(combined)

        # 同样处理B网络标签
        for i in range(len(label_new_B)):
            class_idx = int(label_new_B[i])
            pred_prob = probs_B[i][class_idx]
            
            weight = w if pred_prob > 0.6 else w_clean
            
            one_hot = np.zeros(2)
            one_hot[class_idx] = 1
            
            combined = weight * one_hot + (1-weight) * probs_B[i]
            refined_labels_B[i] = np.argmax(combined)


        xall_extended_A = np.vstack([xall_new_A, X_unlabeled])
        label_extended_A = np.concatenate([refined_labels_A, pseudo_y_B])  # A使用B的伪标签
        
        xall_extended_B = np.vstack([xall_new_B, X_unlabeled])
        label_extended_B = np.concatenate([refined_labels_B, pseudo_y_A])  # B使用A的伪标签
    else:
        # 如果没有足够的未标记样本
        xall_extended_A = xall_new_A
        label_extended_A = label_new_A
        
        xall_extended_B = xall_new_B
        label_extended_B = label_new_B
    
    # 最终训练两个模型
    clf_A = RandomForestClassifier(random_state=seed_t)
    clf_A.fit(xall_extended_A, label_extended_A)
    
    clf_B = RandomForestClassifier(random_state=seed_t+1)
    clf_B.fit(xall_extended_B, label_extended_B)
    
    # 使用验证集寻找最佳阈值
    validation_size = min(int(0.2 * len(xall_extended_A)), 100) 
    pos_indices = np.where(label_extended_A == 1)[0]
    neg_indices = np.where(label_extended_A == 0)[0]

    if len(pos_indices) >= 5 and len(neg_indices) >= 5:
        np.random.seed(seed_t)  
        val_pos = min(len(pos_indices), int(validation_size/2))
        val_neg = min(len(neg_indices), validation_size - val_pos)
        
        selected_pos = np.random.choice(pos_indices, val_pos, replace=False)
        selected_neg = np.random.choice(neg_indices, val_neg, replace=False)
        val_indices = np.concatenate([selected_pos, selected_neg])
        
        X_val = xall_extended_A[val_indices]
        y_val = label_extended_A[val_indices]

        # 预测验证集
        val_probs_A = clf_A.predict_proba(X_val)[:,1]
        val_probs_B = clf_B.predict_proba(X_val)[:,1]
        val_ensemble_probs = (val_probs_A + val_probs_B) / 2

        # 多目标阈值优化
        best_threshold = 0.5
        best_score = -np.inf
        thresholds = np.linspace(0.2, 0.8, 13) 
        for threshold in thresholds:
            val_preds = (val_ensemble_probs >= threshold).astype(int)
            if len(np.unique(val_preds)) < 2:
                continue
                
            # 计算多个指标
            try:
                f1 = metrics.f1_score(y_val, val_preds, pos_label=1)
                mcc = matthews_corrcoef(y_val, val_preds)
                prec = metrics.precision_score(y_val, val_preds, pos_label=1)
                recall = metrics.recall_score(y_val, val_preds, pos_label=1)
                
                # 综合评分 - 软件缺陷预测通常更重视召回率
                score = 0.3*f1 + 0.2*mcc + 0.1*prec + 0.4*recall
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except:
                continue
    else:
        # 验证集不足时回退到默认阈值
        best_threshold = 0.4  
    
    pred_A = clf_A.predict_proba(XX)[:,1]
    pred_B = clf_B.predict_proba(XX)[:,1]
    
    ensemble_probs = (pred_A + pred_B) / 2
    ensemble_preds = (ensemble_probs >= best_threshold).astype(int)
    
    # 在dual_confident_learning函数中预测部分
    # 评估验证集上的模型性能，动态调整权重
    if len(val_indices) > 10:  # 确保有足够样本
        val_preds_A = (clf_A.predict_proba(X_val)[:,1] >= best_threshold).astype(int)
        val_preds_B = (clf_B.predict_proba(X_val)[:,1] >= best_threshold).astype(int)
        
        try:
          
            f1_A = metrics.f1_score(y_val, val_preds_A, pos_label=1)
            f1_B = metrics.f1_score(y_val, val_preds_B, pos_label=1)
            
            recall_A = metrics.recall_score(y_val, val_preds_A, pos_label=1)
            recall_B = metrics.recall_score(y_val, val_preds_B, pos_label=1)
        
            score_A = 0.4 * f1_A + 0.6 * recall_A
            score_B = 0.4 * f1_B + 0.6 * recall_B
            
 
            sum_scores = score_A + score_B
            if sum_scores > 0:
                weight_A = score_A / sum_scores
                weight_B = score_B / sum_scores
            else:
                weight_A = weight_B = 0.5
     
            weight_A = max(0.3, min(0.7, weight_A))
            weight_B = 1 - weight_A
        except:
            weight_A = weight_B = 0.5
    else:
        weight_A = weight_B = 0.5


    ensemble_probs = weight_A * pred_A + weight_B * pred_B

    y_test_flat = y_test_no.ravel()
    y_original = metrics.f1_score(y_test_flat, ensemble_preds, pos_label=1, average="binary")
    

    np.random.seed(seed_t) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test_flat, ensemble_probs)
    thresholds1 = metrics.auc(fpr, tpr)
    
    prec1 = metrics.precision_score(y_test_flat, ensemble_preds, pos_label=1)
    recall1 = metrics.recall_score(y_test_flat, ensemble_preds, pos_label=1)
    
    mcc1 = matthews_corrcoef(y_test_flat, ensemble_preds)
    
    # 计算Effort@20%
    loc_all = sum(XX[:, 4]) + sum(XX[:, 5])
    loc_20 = loc_all * 0.2
    r_pre1 = np.zeros((len(ensemble_preds), 3))
    count_t1 = 0
    
    for il in range(len(ensemble_preds)):
        r_pre1[il][0] = (XX[il][4] + XX[il][5])
        r_pre1[il][1] = ensemble_preds[il]
        r_pre1[il][2] = y_test_flat[il]
        if y_test_flat[il] == 1:
            count_t1 += 1
            
    idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
    sorted_data = r_pre1[idex, :]
    
    t_loc = 0
    il = 0
    count_effort = 0
    
    while t_loc < loc_20 and il < len(sorted_data):
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort = count_effort + 1
        il = il + 1
        
    eff1 = count_effort / count_t1 if count_t1 > 0 else 0
    
    return y_original, thresholds1, prec1, recall1, mcc1, eff1

def get_psx(g_x,g_y,seed_ix):
    psx = np.zeros((len(g_y), 2))
    psx_A = np.zeros((len(g_y), 2))
    psx_B = np.zeros((len(g_y), 2))
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_ix)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(g_x, g_y)):
        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv = g_x[cv_train_idx], g_x[cv_holdout_idx]
        s_train_cv, s_holdout_cv = g_y[cv_train_idx], g_y[cv_holdout_idx]

        # Fit the clf classifier to the training set and
        # predict on the holdout set and update psx.
        log_reg = RandomForestClassifier(random_state=seed_ix)
        log_reg.fit(X_train_cv, s_train_cv)
        psx_cv = log_reg.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
        psx[cv_holdout_idx] = psx_cv
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_ix)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(g_x, g_y)):
             X_train_cv, X_holdout_cv = g_x[cv_train_idx], g_x[cv_holdout_idx]
             s_train_cv, s_holdout_cv = g_y[cv_train_idx], g_y[cv_holdout_idx]
             rus = RandomUnderSampler(random_state=seed_ix)
             X_resampled, y_resampled = rus.fit_resample(X_train_cv, s_train_cv)

             log_reg = RandomForestClassifier(random_state=seed_ix)
             log_reg.fit(X_resampled, y_resampled)
             psx_cv = log_reg.predict_proba(X_holdout_cv)
             psx_A[cv_holdout_idx] = psx_cv
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_ix+1)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(g_x, g_y)):
             X_train_cv, X_holdout_cv = g_x[cv_train_idx], g_x[cv_holdout_idx]
             s_train_cv, s_holdout_cv = g_y[cv_train_idx], g_y[cv_holdout_idx]

             rus = RandomUnderSampler(random_state=seed_ix+1)
             X_resampled, y_resampled = rus.fit_resample(X_train_cv, s_train_cv)
            
             # 训练网络B并预测
             log_reg = RandomForestClassifier(random_state=seed_ix+1)
             log_reg.fit(X_resampled, y_resampled)
             psx_cv = log_reg.predict_proba(X_holdout_cv)
             psx_B[cv_holdout_idx] = psx_cv
       
    return psx,psx_A,psx_B

warnings.filterwarnings('ignore')

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


#置信学习例子，使用sklearner库，但没有使用封装的cleanlab函数
def con_learn():
    #第一列是原始f值，第二列是运用置信学习去噪的结果，第三列是DCLI
    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('dataset/' + csv_string + '.csv')
        v = dataframe.iloc[:]

        train_v = np.array(v)

        ob = train_v[:, 0:14]
        #label = train_v[:,14: ]

        label_RA=train_v[:,-1 ]
        label_MA = train_v[:, -2]
        label_AG = train_v[:, -3]
        label_B = train_v[:, -4]

        yor_all = [[],[],[],[],[],[]]
        yor_all1 = [[],[],[],[],[],[]]
        yor_all2 = [[],[],[],[],[],[]]

        yor_all_B = [[],[],[],[],[],[]]
        yor_all1_B = [[],[],[],[],[],[]]
        yor_all2_B = [[],[],[],[],[],[]]

        yor_all_AG = [[],[],[],[],[],[]]
        yor_all1_AG = [[],[],[],[],[],[]]
        yor_all2_AG = [[],[],[],[],[],[]]

        yor_all_MA = [[],[],[],[],[],[]]
        yor_all1_MA = [[],[],[],[],[],[]]
        yor_all2_MA = [[],[],[],[],[],[]]

        seed=[94733,16588,1761,59345,27886,80894,22367,65435,96636,89300]
        for ix in range(10):
            sfolder = KFold(n_splits=10, shuffle=True,random_state= seed[ix])
            y_or = [[],[],[],[],[],[]]            #原始f值,AUC,ACC,MCC
            y_or1 = [[],[],[],[],[],[]]           #CL f值
            y_or2 = [[],[],[],[],[],[]]           #DCLI f值

            y_or_B = [[],[],[],[],[],[]]          # 原始f值
            y_or1_B = [[],[],[],[],[],[]]         # CL f值
            y_or2_B = [[],[],[],[],[],[]]         #D CLI f值

            y_or_AG = [[],[],[],[],[],[]]         # 原始f值
            y_or1_AG = [[],[],[],[],[],[]]        # CL f值
            y_or2_AG = [[],[],[],[],[],[]]        # DCLI f值

            y_or_MA = [[],[],[],[],[],[]]         # 原始f值
            y_or1_MA = [[],[],[],[],[],[]]        # CL f值
            y_or2_MA = [[],[],[],[],[],[]]        # DCLI f值
            for train_index, test_index in sfolder.split(ob, label_RA):
                X_train, X_test = ob[train_index], ob[test_index]           #划分训练和测试集
                y_train_RA, y_test = label_RA[train_index], label_RA[test_index]
                y_train_B = label_B[train_index]
                y_train_AG = label_AG[train_index]
                y_train_MA = label_MA[train_index]


                psx_B,psx1_B,psx2_B= get_psx(X_train,y_train_B,seed[ix])
                psx_AG, psx1_AG,psx2_AG = get_psx(X_train, y_train_AG, seed[ix])
                psx_MA, psx1_MA ,psx2_MA= get_psx(X_train, y_train_MA, seed[ix])
                psx_RA, psx1_RA ,psx2_RA= get_psx(X_train, y_train_RA, seed[ix])



                log_reg = RandomForestClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train_RA)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or[0].append(y_original)
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                y_or[1].append(roc_auc)
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
                y_or[2].append(prec)
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)  # 召回率
                y_or[3].append(recall)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or[4].append(mcc)
                loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
                loc_20 = loc_all * 0.2
                r_pre1 = np.zeros((len(pre1), 3))
                count_t1 = 0
                for il in range(len(pre1)):
                    r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
                    r_pre1[il][1] = pre1[il]
                    r_pre1[il][2] = y_test[il]
                    if y_test[il] == 1:
                        count_t1 += 1
                idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
                sorted_data = r_pre1[idex, :]
                t_loc = 0
                il = 0
                count_effort = 0
                while t_loc < loc_20:
                    t_loc += sorted_data[il][0]
                    if sorted_data[il][2] == 1:
                        count_effort = count_effort + 1
                    il = il + 1
                y_or[5].append(count_effort / count_t1)

                y_original, auc, prec, recall, mcc, eff = get_noise(y_train_RA,psx_RA,X_train,X_test,y_test,seed[ix])
                y_or1[0].append(y_original)
                y_or1[1].append(auc)
                y_or1[2].append(prec)
                y_or1[3].append(recall)
                y_or1[4].append(mcc)
                y_or1[5].append(eff)
                y_original, auc, prec, recall, mcc, eff =dual_confident_learning(y_train_RA,psx1_RA,psx2_RA,X_train,X_test,y_test,seed[ix])
                y_or2[0].append(y_original)
                y_or2[1].append(auc)
                y_or2[2].append(prec)
                y_or2[3].append(recall)
                y_or2[4].append(mcc)
                y_or2[5].append(eff)


                log_reg = RandomForestClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train_B)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or_B[0].append(y_original)
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                y_or_B[1].append(roc_auc)
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
                y_or_B[2].append(prec)
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)
                y_or_B[3].append(recall)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or_B[4].append(mcc)
                loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
                loc_20 = loc_all * 0.2
                r_pre1 = np.zeros((len(pre1), 3))
                count_t1 = 0
                for il in range(len(pre1)):
                    r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
                    r_pre1[il][1] = pre1[il]
                    r_pre1[il][2] = y_test[il]
                    if y_test[il] == 1:
                        count_t1 += 1
                idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
                sorted_data = r_pre1[idex, :]
                t_loc = 0
                il = 0
                count_effort = 0
                while t_loc < loc_20:
                    t_loc += sorted_data[il][0]
                    if sorted_data[il][2] == 1:
                        count_effort = count_effort + 1
                    il = il + 1
                y_or_B[5].append(count_effort / count_t1)

                y_original, auc, prec, recall, mcc, eff = get_noise(y_train_B, psx_B, X_train, X_test, y_test, seed[ix])
                y_or1_B[0].append(y_original)
                y_or1_B[1].append(auc)
                y_or1_B[2].append(prec)
                y_or1_B[3].append(recall)
                y_or1_B[4].append(mcc)
                y_or1_B[5].append(eff)
                y_original, auc, prec, recall, mcc, eff = dual_confident_learning(y_train_B, psx1_B, psx2_B, X_train, X_test, y_test, seed[ix])
                y_or2_B[0].append(y_original)
                y_or2_B[1].append(auc)
                y_or2_B[2].append(prec)
                y_or2_B[3].append(recall)
                y_or2_B[4].append(mcc)
                y_or2_B[5].append(eff)

                log_reg = RandomForestClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train_AG)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or_AG[0].append(y_original)
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                y_or_AG[1].append(roc_auc)
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
                y_or_AG[2].append(prec)
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)
                y_or_AG[3].append(recall)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or_AG[4].append(mcc)
                loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
                loc_20 = loc_all * 0.2
                r_pre1 = np.zeros((len(pre1), 3))
                count_t1 = 0
                for il in range(len(pre1)):
                    r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
                    r_pre1[il][1] = pre1[il]
                    r_pre1[il][2] = y_test[il]
                    if y_test[il] == 1:
                        count_t1 += 1
                idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
                sorted_data = r_pre1[idex, :]
                t_loc = 0
                il = 0
                count_effort = 0
                while t_loc < loc_20:
                    t_loc += sorted_data[il][0]
                    if sorted_data[il][2] == 1:
                        count_effort = count_effort + 1
                    il = il + 1
                y_or_AG[5].append(count_effort / count_t1)
                y_original, auc, prec, recall, mcc, eff = get_noise(y_train_AG, psx_AG, X_train, X_test, y_test, seed[ix])
                y_or1_AG[0].append(y_original)
                y_or1_AG[1].append(auc)
                y_or1_AG[2].append(prec)
                y_or1_AG[3].append(recall)
                y_or1_AG[4].append(mcc)
                y_or1_AG[5].append(eff)
                y_original, auc, prec, recall, mcc, eff = dual_confident_learning(y_train_AG, psx1_AG, psx2_AG, X_train, X_test, y_test, seed[ix])
                y_or2_AG[0].append(y_original)
                y_or2_AG[1].append(auc)
                y_or2_AG[2].append(prec)
                y_or2_AG[3].append(recall)
                y_or2_AG[4].append(mcc)
                y_or2_AG[5].append(eff)

                log_reg = RandomForestClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train_MA)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or_MA[0].append(y_original)
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                y_or_MA[1].append(roc_auc)
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)  # 精确率
                y_or_MA[2].append(prec)
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)
                y_or_MA[3].append(recall)
                mcc = matthews_corrcoef(y_test_1, pre1)
                y_or_MA[4].append(mcc)
                loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
                loc_20 = loc_all * 0.2
                r_pre1 = np.zeros((len(pre1), 3))
                count_t1 = 0
                for il in range(len(pre1)):
                    r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
                    r_pre1[il][1] = pre1[il]
                    r_pre1[il][2] = y_test[il]
                    if y_test[il] == 1:
                        count_t1 += 1
                idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
                sorted_data = r_pre1[idex, :]
                t_loc = 0
                il = 0
                count_effort = 0
                while t_loc < loc_20:
                    t_loc += sorted_data[il][0]
                    if sorted_data[il][2] == 1:
                        count_effort = count_effort + 1
                    il = il + 1
                y_or_MA[5].append(count_effort / count_t1)

                y_original, auc, prec, recall, mcc, eff = get_noise(y_train_MA, psx_MA, X_train, X_test, y_test, seed[ix])
                y_or1_MA[0].append(y_original)
                y_or1_MA[1].append(auc)
                y_or1_MA[2].append(prec)
                y_or1_MA[3].append(recall)
                y_or1_MA[4].append(mcc)
                y_or1_MA[5].append(eff)
                y_original, auc, prec, recall, mcc, eff = dual_confident_learning(y_train_MA, psx1_MA, psx2_MA, X_train, X_test, y_test, seed[ix])
                y_or2_MA[0].append(y_original)
                y_or2_MA[1].append(auc)
                y_or2_MA[2].append(prec)
                y_or2_MA[3].append(recall)
                y_or2_MA[4].append(mcc)
                y_or2_MA[5].append(eff)

            for index_i in range(6):
                yor_all[index_i].append(np.mean(y_or[index_i]))
            for index_i in range(6):
                yor_all1[index_i].append(np.mean(y_or1[index_i]))
            for index_i in range(6):
                yor_all2[index_i].append(np.mean(y_or2[index_i]))


            for index_i in range(6):
                yor_all_B[index_i].append(np.mean(y_or_B[index_i]))
            for index_i in range(6):
                yor_all1_B[index_i].append(np.mean(y_or1_B[index_i]))
            for index_i in range(6):
                yor_all2_B[index_i].append(np.mean(y_or2_B[index_i]))

            for index_i in range(6):
                yor_all_AG[index_i].append(np.mean(y_or_AG[index_i]))
            for index_i in range(6):
                yor_all1_AG[index_i].append(np.mean(y_or1_AG[index_i]))
            for index_i in range(6):
                yor_all2_AG[index_i].append(np.mean(y_or2_AG[index_i]))

            for index_i in range(6):
                yor_all_MA[index_i].append(np.mean(y_or_MA[index_i]))
            for index_i in range(6):
                yor_all1_MA[index_i].append(np.mean(y_or1_MA[index_i]))
            for index_i in range(6):
                yor_all2_MA[index_i].append(np.mean(y_or2_MA[index_i]))


        f = open("RA-1014.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all[0]),
            np.mean(yor_all1[0]),np.mean(yor_all2[0]),np.mean(yor_all[1]),np.mean(yor_all1[1]),np.mean(yor_all2[1]),
            np.mean(yor_all[2]),np.mean(yor_all1[2]),np.mean(yor_all2[2]),np.mean(yor_all[3]),np.mean(yor_all1[3]),
            np.mean(yor_all2[3]),np.mean(yor_all[4]),np.mean(yor_all1[4]),np.mean(yor_all2[4]),np.mean(yor_all[5]),
                                                                                      np.mean(yor_all1[5]),np.mean(yor_all2[5])))
        f.close()


        f = open("B-1014.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all_B[0]),
                                                                                      np.mean(yor_all1_B[0]),
                                                                                      np.mean(yor_all2_B[0]),
                                                                                      np.mean(yor_all_B[1]),
                                                                                      np.mean(yor_all1_B[1]),
                                                                                      np.mean(yor_all2_B[1]),
                                                                                      np.mean(yor_all_B[2]),
                                                                                      np.mean(yor_all1_B[2]),
                                                                                      np.mean(yor_all2_B[2]),
                                                                                      np.mean(yor_all_B[3]),
                                                                                      np.mean(yor_all1_B[3]),
                                                                                      np.mean(yor_all2_B[3]),np.mean(yor_all_B[4]),
                                                                                      np.mean(yor_all1_B[4]),
                                                                                      np.mean(yor_all2_B[4]),np.mean(yor_all_B[5]),
                                                                                      np.mean(yor_all1_B[5]),
                                                                                      np.mean(yor_all2_B[5])))
        f.close()


        f = open("AG-1014.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all_AG[0]),
                                                                                      np.mean(yor_all1_AG[0]),
                                                                                      np.mean(yor_all2_AG[0]),
                                                                                      np.mean(yor_all_AG[1]),
                                                                                      np.mean(yor_all1_AG[1]),
                                                                                      np.mean(yor_all2_AG[1]),
                                                                                      np.mean(yor_all_AG[2]),
                                                                                      np.mean(yor_all1_AG[2]),
                                                                                      np.mean(yor_all2_AG[2]),
                                                                                      np.mean(yor_all_AG[3]),
                                                                                      np.mean(yor_all1_AG[3]),
                                                                                      np.mean(yor_all2_AG[3]),np.mean(yor_all_AG[4]),
                                                                                      np.mean(yor_all1_AG[4]),
                                                                                      np.mean(yor_all2_AG[4]),np.mean(yor_all_AG[5]),
                                                                                      np.mean(yor_all1_AG[5]),
                                                                                      np.mean(yor_all2_AG[5])))
        f.close()


        f = open("MA-1014.csv", 'a')
        f.write("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (csv_string, np.mean(yor_all_MA[0]),
                                                                                      np.mean(yor_all1_MA[0]),
                                                                                      np.mean(yor_all2_MA[0]),
                                                                                      np.mean(yor_all_MA[1]),
                                                                                      np.mean(yor_all1_MA[1]),
                                                                                      np.mean(yor_all2_MA[1]),
                                                                                      np.mean(yor_all_MA[2]),
                                                                                      np.mean(yor_all1_MA[2]),
                                                                                      np.mean(yor_all2_MA[2]),
                                                                                      np.mean(yor_all_MA[3]),
                                                                                      np.mean(yor_all1_MA[3]),
                                                                                      np.mean(yor_all2_MA[3]), np.mean(yor_all_MA[4]),
                                                                                      np.mean(yor_all1_MA[4]),
                                                                                      np.mean(yor_all2_MA[4]), np.mean(yor_all_MA[5]),
                                                                                      np.mean(yor_all1_MA[5]),
                                                                                      np.mean(yor_all2_MA[5])))
        f.close()

        print(csv_string+" is done~!")



if __name__ == '__main__':
    con_learn()
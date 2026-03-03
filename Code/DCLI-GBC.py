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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
from cleanlab.classification import LearningWithNoisyLabels
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from cleanlab.pruning import get_noise_indices
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
import scipy.special as special
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
import scipy.special as special
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}

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
        psx=pre_new1,
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

    x_mask = ~ordered_label_errors
    x_pruned = xall_new1[x_mask]
    s_pruned = y_train2[x_mask]

    sample_weight = np.ones(np.shape(s_pruned))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix[k][k]
        sample_weight[s_pruned == k] = sample_weight_k

    log_reg = GradientBoostingClassifier(random_state=seed_t)
    log_reg.fit(x_pruned, s_pruned, sample_weight=sample_weight)

    pre1 = log_reg.predict(XX)
    y_test_11 = y_test_no.ravel()
    y_original = metrics.f1_score(y_test_11, pre1, pos_label=1, average="binary")

    fpr, tpr, thersholds = metrics.roc_curve(y_test_11, pre1)
    roc_auc = metrics.auc(fpr, tpr)
    prec = metrics.precision_score(y_test_11, pre1, pos_label=1)
    recall = metrics.recall_score(y_test_11, pre1, pos_label=1)

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
    while t_loc < loc_20 and il < len(sorted_data):
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort = count_effort + 1
        il = il + 1
    eff=count_effort / count_t1
    return y_original, roc_auc, prec,recall, mcc,eff

def dual_confident_learning(label_new, psx_A, psx_B, xall_new, X_test, y_test, seed_t):

    label_new1 = copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(psx_A)
    pre_new2 = copy.deepcopy(psx_B)
    xall_new1 = copy.deepcopy(xall_new)
    XX = copy.deepcopy(X_test)
    y_test_no = copy.deepcopy(y_test)
    
    label_1 = label_new1.ravel()
    label_1 = label_1.astype(np.int32)
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
    
    mask_A = ~noise_idx_B
    mask_B = ~noise_idx_A
    
    
    xall_new_A = xall_new1[mask_A]
    label_new_A = label_1[mask_A]
    
    xall_new_B = xall_new1[mask_B]
    label_new_B = label_1[mask_B]
    
    sample_weight_A = np.ones(np.shape(label_new_A))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix_B[k][k]
        sample_weight_A[label_new_A == k] = sample_weight_k
    
    sample_weight_B = np.ones(np.shape(label_new_B))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix_A[k][k]
        sample_weight_B[label_new_B == k] = sample_weight_k
    
    n_unlabeled = min(int(len(xall_new_A) * 0.2), len(XX))
    if n_unlabeled > 0:
        np.random.seed(seed_t)
        unlabeled_indices = np.random.choice(len(XX), n_unlabeled, replace=False)
        X_unlabeled = XX[unlabeled_indices]
        
        clf_A_temp = GradientBoostingClassifier(random_state=seed_t)#使用GradientBoostingClassifier
        clf_A_temp.fit(xall_new_A, label_new_A, sample_weight=sample_weight_A)
        
        clf_B_temp = GradientBoostingClassifier(random_state=seed_t+1)
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
        
        w = 0.7
        w_clean = 0.5

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
            
            combined = weight * one_hot + (1-weight) * probs_A[i]
            refined_labels_A[i] = np.argmax(combined)

        for i in range(len(label_new_B)):
            class_idx = int(label_new_B[i])
            pred_prob = probs_B[i][class_idx]
            
            weight = w if pred_prob > 0.6 else w_clean
            
            one_hot = np.zeros(2)
            one_hot[class_idx] = 1
            
            combined = weight * one_hot + (1-weight) * probs_B[i]
            refined_labels_B[i] = np.argmax(combined)

        xall_extended_A = np.vstack([xall_new_A, X_unlabeled])
        label_extended_A = np.concatenate([refined_labels_A, pseudo_y_B])
        
        xall_extended_B = np.vstack([xall_new_B, X_unlabeled])
        label_extended_B = np.concatenate([refined_labels_B, pseudo_y_A])
    else:
        xall_extended_A = xall_new_A
        label_extended_A = label_new_A
        
        xall_extended_B = xall_new_B
        label_extended_B = label_new_B
    
    clf_A = GradientBoostingClassifier(random_state=seed_t)
    clf_A.fit(xall_extended_A, label_extended_A)
    
    clf_B = GradientBoostingClassifier(random_state=seed_t+1)
    clf_B.fit(xall_extended_B, label_extended_B)
    
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

        val_probs_A = clf_A.predict_proba(X_val)[:,1]
        val_probs_B = clf_B.predict_proba(X_val)[:,1]
        val_ensemble_probs = (val_probs_A + val_probs_B) / 2

        best_threshold = 0.5
        best_score = -np.inf
        thresholds = np.linspace(0.2, 0.8, 13)

        for threshold in thresholds:
            val_preds = (val_ensemble_probs >= threshold).astype(int)
            if len(np.unique(val_preds)) < 2:
                continue
            try:
                f1 = metrics.f1_score(y_val, val_preds, pos_label=1)
                mcc = matthews_corrcoef(y_val, val_preds)
                prec = metrics.precision_score(y_val, val_preds, pos_label=1)
                recall = metrics.recall_score(y_val, val_preds, pos_label=1)
                score = 0.3*f1 + 0.2*mcc + 0.1*prec + 0.4*recall
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except:
                continue
    else:
        best_threshold = 0.4
    
    pred_A = clf_A.predict_proba(XX)[:,1]
    pred_B = clf_B.predict_proba(XX)[:,1]
    
    ensemble_probs = (pred_A + pred_B) / 2
    ensemble_preds = (ensemble_probs >= best_threshold).astype(int)
    
    if len(val_indices) > 10:
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


warnings.filterwarnings('ignore')

def safe_mean(data):
    return np.mean(data) if len(data) > 0 else 0

def con_learn():
    file_name = 'GBC-comparison.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["datasets", 
                         "f1-ori", "f1-cl", "f1-cli", "f1-dcli", 
                         "auc-ori", "auc-cl", "auc-cli", "auc-dcli", 
                         "prec-ori", "prec-cl", "prec-cli", "prec-dcli", 
                         "recall-ori", "recall-cl", "recall-cli", "recall-dcli", 
                         "mcc-ori", "mcc-cl", "mcc-cli", "mcc-dcli", 
                         "effort-ori", "effort-cl", "effort-cli", "effort-dcli"])
    
    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('dataset/' + csv_string + '.csv')
        v = dataframe.iloc[:]
        train_v = np.array(v)
        ori_all = []
        ori_all.append(csv_string)

        ob = train_v[:, 0:14]
        label = train_v[:, -1]
        label = label.reshape(-1, 1)

        yor_all = []
        yor_all1 = []
        yor_all2 = []
        yor_all3 = []

        auc_all = []
        auc_all1 = []
        auc_all2 = []
        auc_all3 = []

        prec_all = []
        prec_all1 = []
        prec_all2 = []
        prec_all3 = []

        recall_all = []
        recall_all1 = []
        recall_all2 = []
        recall_all3 = []

        mcc_all = []
        mcc_all1 = []
        mcc_all2 = []
        mcc_all3 = []

        effort_all = []
        effort_all1 = []
        effort_all2 = []
        effort_all3 = []

        seed=[94733,16588,1761,59345,27886,80894,22367,65435,96636,89300]
        for ix in range(10):
            sfolder = KFold(n_splits=10, shuffle=True,random_state=seed[ix])
            y_or = []
            y_or1 = []
            y_or2 = []
            y_or3 = []

            auc_or = []
            auc_or1 = []
            auc_or2 = []
            auc_or3 = []

            prec_or = []
            prec_or1 = []
            prec_or2 = []
            prec_or3 = []

            recall_or = []
            recall_or1 = []
            recall_or2 = []
            recall_or3 = []

            mcc_or = []
            mcc_or1 = []
            mcc_or2 = []
            mcc_or3 = []

            effort_or = []
            effort_or1 = []
            effort_or2 = []
            effort_or3 = []
     
            for train_index, test_index in sfolder.split(ob, label):
                X_train, X_test = ob[train_index], ob[test_index]
                y_train, y_test = label[train_index], label[test_index]

                psx = np.zeros((len(y_train), 2))
                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed[ix])
                for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, y_train)):
                    X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
                    s_train_cv, s_holdout_cv = y_train[cv_train_idx], y_train[cv_holdout_idx]
                    
                    log_reg = GradientBoostingClassifier(random_state=seed[ix])
                    log_reg.fit(X_train_cv, s_train_cv)
                    psx_cv = log_reg.predict_proba(X_holdout_cv)
                    psx[cv_holdout_idx] = psx_cv
                
                psx1 = np.zeros((len(y_train), 2))
                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed[ix])
                for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, y_train)):
                    X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
                    s_train_cv, s_holdout_cv = y_train[cv_train_idx], y_train[cv_holdout_idx]
                    
                    rus = RandomUnderSampler(random_state=seed[ix])
                    X_resampled, y_resampled = rus.fit_resample(X_train_cv, s_train_cv)
                    
                    log_reg = GradientBoostingClassifier(random_state=seed[ix])
                    log_reg.fit(X_resampled, y_resampled)
                    psx_cv1 = log_reg.predict_proba(X_holdout_cv)
                    psx1[cv_holdout_idx] = psx_cv1
                
                psx_A = np.zeros((len(y_train), 2))
                psx_B = np.zeros((len(y_train), 2))
                
                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed[ix])
                for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, y_train)):
                    X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
                    s_train_cv, s_holdout_cv = y_train[cv_train_idx], y_train[cv_holdout_idx]

                    rus = RandomUnderSampler(random_state=seed[ix])
                    X_resampled, y_resampled = rus.fit_resample(X_train_cv, s_train_cv)
                    
                    log_reg = GradientBoostingClassifier(random_state=seed[ix])
                    log_reg.fit(X_resampled, y_resampled)
                    psx_cv = log_reg.predict_proba(X_holdout_cv)
                    psx_A[cv_holdout_idx] = psx_cv
                
                kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed[ix]+1)
                for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, y_train)):
                    X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
                    s_train_cv, s_holdout_cv = y_train[cv_train_idx], y_train[cv_holdout_idx]

                    rus = RandomUnderSampler(random_state=seed[ix]+1)
                    X_resampled, y_resampled = rus.fit_resample(X_train_cv, s_train_cv)
                    
                    log_reg = GradientBoostingClassifier(random_state=seed[ix]+1)
                    log_reg.fit(X_resampled, y_resampled)
                    psx_cv = log_reg.predict_proba(X_holdout_cv)
                    psx_B[cv_holdout_idx] = psx_cv

                log_reg = GradientBoostingClassifier(random_state=seed[ix])
                log_reg.fit(X_train, y_train)
                pre1 = log_reg.predict(X_test)
                y_test_1 = y_test.ravel()
                
                y_original = metrics.f1_score(y_test_1, pre1, pos_label=1, average="binary")
                y_or.append(y_original)
                
                fpr, tpr, thersholds = metrics.roc_curve(y_test_1, pre1)
                roc_auc = metrics.auc(fpr, tpr)
                auc_or.append(roc_auc)
                
                prec = metrics.precision_score(y_test_1, pre1, pos_label=1)
                prec_or.append(prec)
                
                recall = metrics.recall_score(y_test_1, pre1, pos_label=1)
                recall_or.append(recall)
                
                mcc = matthews_corrcoef(y_test_1, pre1)
                mcc_or.append(mcc)
                
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
                while t_loc < loc_20 and il < len(sorted_data):
                    t_loc += sorted_data[il][0]
                    if sorted_data[il][2] == 1:
                        count_effort = count_effort + 1
                    il = il + 1
                effort_or.append(count_effort / count_t1 if count_t1 > 0 else 0)
                
                y_original, thresholds1, prec1, recall1, mcc1, eff1 = get_noise(y_train, psx, X_train, X_test, y_test, seed[ix])
                y_or1.append(y_original)
                auc_or1.append(thresholds1)
                prec_or1.append(prec1)
                recall_or1.append(recall1)
                mcc_or1.append(mcc1)
                effort_or1.append(eff1)
                
                y_original, thresholds2, prec2, recall2, mcc2, eff2 = get_noise(y_train, psx1, X_train, X_test, y_test, seed[ix])
                y_or2.append(y_original)
                auc_or2.append(thresholds2)
                prec_or2.append(prec2)
                recall_or2.append(recall2)
                mcc_or2.append(mcc2)
                effort_or2.append(eff2)
                
                y_original, thresholds3, prec3, recall3, mcc3, eff3 = dual_confident_learning(y_train, psx_A, psx_B, X_train, X_test, y_test, seed[ix])
                y_or3.append(y_original)
                auc_or3.append(thresholds3)
                prec_or3.append(prec3)
                recall_or3.append(recall3)
                mcc_or3.append(mcc3)
                effort_or3.append(eff3)
            
            yor_all.append(safe_mean(y_or))
            yor_all1.append(safe_mean(y_or1))
            yor_all2.append(safe_mean(y_or2))
            yor_all3.append(safe_mean(y_or3))

            auc_all.append(safe_mean(auc_or))
            auc_all1.append(safe_mean(auc_or1))
            auc_all2.append(safe_mean(auc_or2))
            auc_all3.append(safe_mean(auc_or3))

            prec_all.append(safe_mean(prec_or))
            prec_all1.append(safe_mean(prec_or1))
            prec_all2.append(safe_mean(prec_or2))
            prec_all3.append(safe_mean(prec_or3))

            recall_all.append(safe_mean(recall_or))
            recall_all1.append(safe_mean(recall_or1))
            recall_all2.append(safe_mean(recall_or2))
            recall_all3.append(safe_mean(recall_or3))

            mcc_all.append(safe_mean(mcc_or))
            mcc_all1.append(safe_mean(mcc_or1))
            mcc_all2.append(safe_mean(mcc_or2))
            mcc_all3.append(safe_mean(mcc_or3))

            effort_all.append(safe_mean(effort_or))
            effort_all1.append(safe_mean(effort_or1))
            effort_all2.append(safe_mean(effort_or2))
            effort_all3.append(safe_mean(effort_or3))
        
        ori_all.append(safe_mean(yor_all))
        ori_all.append(safe_mean(yor_all1))
        ori_all.append(safe_mean(yor_all2))
        ori_all.append(safe_mean(yor_all3))

        ori_all.append(safe_mean(auc_all))
        ori_all.append(safe_mean(auc_all1))
        ori_all.append(safe_mean(auc_all2))
        ori_all.append(safe_mean(auc_all3))

        ori_all.append(safe_mean(prec_all))
        ori_all.append(safe_mean(prec_all1))
        ori_all.append(safe_mean(prec_all2))
        ori_all.append(safe_mean(prec_all3))

        ori_all.append(safe_mean(recall_all))
        ori_all.append(safe_mean(recall_all1))
        ori_all.append(safe_mean(recall_all2))
        ori_all.append(safe_mean(recall_all3))

        ori_all.append(safe_mean(mcc_all))
        ori_all.append(safe_mean(mcc_all1))
        ori_all.append(safe_mean(mcc_all2))
        ori_all.append(safe_mean(mcc_all3))

        ori_all.append(safe_mean(effort_all))
        ori_all.append(safe_mean(effort_all1))
        ori_all.append(safe_mean(effort_all2))
        ori_all.append(safe_mean(effort_all3))

        csv_writer.writerow(ori_all)
        print(csv_string + " is done~!")
    
    f.close()

if __name__ == '__main__':
    con_learn()



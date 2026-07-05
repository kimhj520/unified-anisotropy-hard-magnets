# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 20:56:38 2025

@author: hjkim
"""
#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shap
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, precision_recall_curve

from sklearn.model_selection import StratifiedKFold
import seaborn as sns

# 1. 데이터 불러오기 및 전처리
print("="*60)
print("MULTI-MODEL REGRESSION ANALYSIS")
print("="*60)

# 파일명 입력받기
input_file = input("분석할 CSV 파일명을 입력하세요: ").strip()
if not input_file.endswith('.csv'):
    input_file += '.csv'
    
# 작업 유형 선택
task_type = input("작업 유형을 선택하세요 ('regression' 또는 'classification'): ").strip().lower()
if task_type not in ['regression', 'classification']:
    print("잘못된 입력입니다. 'regression'을 기본값으로 사용합니다.")
    task_type = 'regression'

df = pd.read_csv(input_file)  # 기존의 하드코딩된 파일명 대신

df_clean = df.dropna()
X = df_clean.iloc[:, 1:-1]  # ID 제외, reference 제외
y = df_clean.iloc[:, -1] 
if task_type == 'classification':
    y = (y >= 0).astype(int)
feature_names = X.columns

# 2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 모델 선택
print("\n" + "="*60)
print("MODEL SELECTION")
print("="*60)

model_choice = input("사용할 모델을 선택하세요 ('xgb', 'rf', 'et'): ").strip().lower()
if model_choice not in ['xgb', 'rf', 'et']:
    print("잘못된 입력입니다. 'xgb'를 기본값으로 사용합니다.")
    model_choice = 'xgb'

model_names = {'xgb': 'XGBoost', 'rf': 'Random Forest', 'et': 'Extra Trees'}

# 4. 초기 모델 설정 및 Feature Importance 계산
if model_choice == 'xgb':
    if task_type == 'regression':
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            objective='reg:squarederror'
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            objective='binary:logistic'  # 또는 multi:softprob (다중분류)
        )
else:  # RF or ET
    if model_choice == 'rf':
        if task_type == 'regression':
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=4
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=4
            )
    else:  # 'et'
        if task_type == 'regression':
            model = ExtraTreesRegressor(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=4
            )
        else:
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=10,
                random_state=42,
                n_jobs=4
            )

# 5. 초기 5-fold CV로 feature importance 계산 및 fold별 성능 출력
n_fold = 5
# K-Fold 초기화 (작업 유형에 따라 다르게)
if task_type == 'regression':
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
else:
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

feature_importances_list = []

if task_type == 'regression':
    mae_list, rmse_list, r2_list = [], [], []
else:
    accuracy_list, f1_list, auc_list = [], [], []

print(f"\n📊 Initial Feature Importance Calculation with {model_names[model_choice]}...")

if task_type == 'regression':
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(X_scaled), total=n_fold), 1):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_tr, y_tr)
        feature_importances_list.append(model.feature_importances_)
        
        # fold별 성능 측정
        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = root_mean_squared_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
else:  # classification
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(X_scaled, y), total=n_fold), 1):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_tr, y_tr)
        feature_importances_list.append(model.feature_importances_)
        
        # fold별 성능 측정
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1]
        
        accuracy = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='weighted')
        auc_score = roc_auc_score(y_te, y_pred_proba)
        
        accuracy_list.append(accuracy)
        f1_list.append(f1)
        auc_list.append(auc_score)

print("\n=== Fold-wise Metrics Summary ===")
if task_type == 'regression':
    print(f"MAE : {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
    print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"R2  : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
else:  # classification
    print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"F1 Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"AUC     : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print("="*60)

# 6. 평균 feature importance 계산
importances_array = np.array(feature_importances_list)
mean_importance = importances_array.mean(axis=0)
std_importance = importances_array.std(axis=0)

# 7. Feature 그룹 정의
crystal_start, crystal_end = 392, 399
struct_start, struct_end = 408, 600
cgcnn_start = 600
cgcnn_end = 632

# 8. 중요도 합산
crystal_sum = mean_importance[crystal_start:crystal_end].sum()
struct_sum = mean_importance[struct_start:struct_end].sum()
cgcnn_sum = mean_importance[cgcnn_start:cgcnn_end].sum()

# 9. 중요도와 이름 축소
reduced_importance = np.concatenate([
    mean_importance[:crystal_start],
    [crystal_sum],
    mean_importance[crystal_end:struct_start],
    [struct_sum],
    [cgcnn_sum]
])

reduced_feature_names = list(feature_names[:crystal_start]) + \
                        [feature_names[crystal_start]] + \
                        list(feature_names[crystal_end:struct_start]) + \
                        [feature_names[struct_start]] + \
                        [feature_names[cgcnn_start]]

# 10. 원본 feature groups 정의
original_feature_groups_dict = {}  # 명시적으로 다른 이름 사용
for i in range(0, 392):
    original_feature_groups_dict[i] = [i]
original_feature_groups_dict[392] = list(range(392, 399))
for i in range(399, 408):
    original_feature_groups_dict[i - 6] = [i]
original_feature_groups_dict[402] = list(range(408, 600))
original_feature_groups_dict[403] = list(range(cgcnn_start, cgcnn_end))

# 이후 사용하는 곳도 모두 변경
original_feature_groups = original_feature_groups_dict

# 11. Feature Selection 함수 정의
def metric_feature_selection(X_scaled, y, reduced_importance, original_feature_groups, 
                           k=n_fold, scoring_metric='mae', model_type='xgb', task_type='regression'):
    """
    scoring_metric: 'mae', 'r2', 'f1', or 'accuracy'
    model_type: 'xgb', 'rf', or 'et'
    task_type: 'regression' or 'classification'
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    print(f"\n🔍 Feature Selection optimizing {scoring_metric.upper()} with {model_names[model_type]}...")
    print(f"   Total groups: {len(reduced_importance)}")
    print(f"   Total features: {X_scaled.shape[1]}")
    
    group_order = np.argsort(reduced_importance)[::-1]
    
    scores_all, num_features = [], []
    best_subset = None
    best_groups = None

    if task_type == 'regression':
        if scoring_metric == 'mae':
            best_score = -np.inf
            scoring = 'neg_mean_absolute_error'
            better = lambda score, best_score: score > best_score
        elif scoring_metric == 'r2':
            best_score = -np.inf
            scoring = 'r2'
            better = lambda score, best_score: score > best_score
    else:  # classification
        if scoring_metric == 'f1':
            best_score = -np.inf
            scoring = 'f1_weighted'
            better = lambda score, best_score: score > best_score
        elif scoring_metric == 'accuracy':
            best_score = -np.inf
            scoring = 'accuracy'
            better = lambda score, best_score: score > best_score
    
    if task_type == 'regression':
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for i in tqdm(range(len(group_order), 0, -1), desc=f"Feature elimination ({scoring_metric.upper()})"):
        selected_groups = group_order[:i]
        selected_indices = []
        
        for group_idx in selected_groups:
            selected_indices.extend(original_feature_groups[group_idx])
        
        selected_indices = sorted(set(selected_indices))
        if len(selected_indices) == 0:
            continue
            
        X_selected = X_scaled[:, selected_indices]
        
        # 모델 선택
        if task_type == 'regression':
            if model_type == 'xgb':
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=5,
                    random_state=42,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    predictor='gpu_predictor'
                )
            elif model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=10,
                    random_state=42,
                    n_jobs=4
                )
            else:  # 'et'
                model = ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=10,
                    random_state=42,
                    n_jobs=4
                )
        else:  # classification
            if model_type == 'xgb':
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    random_state=42,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    predictor='gpu_predictor'
                )
            elif model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    random_state=42,
                    n_jobs=4
                )
            else:  # 'et'
                model = ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=10,
                    random_state=42,
                    n_jobs=4
                )        
        scores = cross_val_score(model, X_selected, y, cv=kf, scoring=scoring)
        score = np.mean(scores)
        
        scores_all.append(score)
        num_features.append(len(selected_groups))
        
        if better(score, best_score):
            best_score = score
            best_subset = selected_indices.copy()
            best_groups = selected_groups.copy()
            disp_score = -score if scoring_metric == 'mae' else score
            print(f"   New best at {len(selected_groups)} groups: {scoring_metric.upper()}={disp_score:.4f}")
    
    best_idx = np.argmax(scores_all)
    disp_score = -best_score if scoring_metric == 'mae' else best_score
    
    print(f"\n   ✅ Best {scoring_metric.upper()} Score: {disp_score:.4f}")
    print(f"   Selected features: {len(best_subset)}")
    print(f"   Selected groups: {len(best_groups)}")
    print(f"   Best found at index {best_idx}: {num_features[best_idx]} groups")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    if scoring_metric == 'mae':
        values = [-s for s in scores_all[::-1]]
        ylabel = 'MAE'
    elif scoring_metric == 'r2':
        values = [s for s in scores_all[::-1]]
        ylabel = 'R²'
    elif scoring_metric == 'f1':
        values = [s for s in scores_all[::-1]]
        ylabel = 'F1 Score'
    elif scoring_metric == 'accuracy':
        values = [s for s in scores_all[::-1]]
        ylabel = 'Accuracy'
    
    plt.plot(num_features[::-1], values, 'r-', linewidth=2)
    best_idx_reversed = len(scores_all) - 1 - best_idx
    plt.scatter(num_features[::-1][best_idx_reversed], 
               values[best_idx_reversed], 
               color='green', s=100, zorder=5)
    plt.annotate(f'Best {ylabel}: {values[best_idx_reversed]:.4f}\n{num_features[::-1][best_idx_reversed]} groups',
                xy=(num_features[::-1][best_idx_reversed], values[best_idx_reversed]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    plt.ylabel(ylabel)
    plt.xlabel('Number of Feature Groups')
    plt.title(f'{ylabel} vs Number of Features ({model_names[model_type]}, Backward Elimination)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_subset, best_groups, best_score

# 12. Feature Selection 실행
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Feature selection 방법 선택
fs_method = input("Feature selection 방법을 선택하세요:\n"
                  "1. 'auto' - 자동 최적화 (backward selection)\n"
                  "2. 숫자 입력 - 상위 N개 그룹 사용 (예: 20)\n"
                  "선택: ").strip()

if fs_method.lower() == 'auto':
    # 기존 코드 그대로
    if task_type == 'regression':
        opt_metric = input("최적화할 metric을 입력하세요 ('mae' 또는 'r2'): ").strip().lower()
        if opt_metric not in ['mae', 'r2']:
            print("잘못된 입력입니다. 'mae'를 기본값으로 사용합니다.")
            opt_metric = 'mae'
    else:
        opt_metric = input("최적화할 metric을 입력하세요 ('f1' 또는 'accuracy'): ").strip().lower()
        if opt_metric not in ['f1', 'accuracy']:
            print("잘못된 입력입니다. 'f1'을 기본값으로 사용합니다.")
            opt_metric = 'f1'

    best_indices, best_groups, best_score = metric_feature_selection(
        X_scaled, y, reduced_importance, original_feature_groups, 
        k=n_fold, scoring_metric=opt_metric, model_type=model_choice, task_type=task_type
    )
    
else:
    # 숫자를 입력받은 경우
    try:
        n_top_groups = int(fs_method)
        if n_top_groups <= 0 or n_top_groups > len(reduced_importance):
            print(f"잘못된 입력입니다. 1에서 {len(reduced_importance)} 사이의 값을 입력하세요.")
            n_top_groups = min(20, len(reduced_importance))
            print(f"기본값 {n_top_groups}을 사용합니다.")
    except ValueError:
        print("잘못된 입력입니다. 기본값 20을 사용합니다.")
        n_top_groups = min(20, len(reduced_importance))
    
    # 상위 N개 그룹 선택
    print(f"\n🔍 Selecting top {n_top_groups} feature groups based on importance...")
    
    # 중요도 순서로 정렬된 그룹 인덱스
    group_order = np.argsort(reduced_importance)[::-1]
    best_groups = group_order[:n_top_groups]
    
    # 선택된 그룹에 해당하는 feature indices 추출
    best_indices = []
    for group_idx in best_groups:
        best_indices.extend(original_feature_groups[group_idx])
    best_indices = sorted(set(best_indices))
    
    # 선택된 feature로 성능 평가
    X_selected = X_scaled[:, best_indices]
    
    # 모델 생성 (기존 코드와 동일한 방식)
    if task_type == 'regression':
        opt_metric = 'mae'  # 기본값 설정
        if model_choice == 'xgb':
            eval_model = xgb.XGBRegressor(
                n_estimators=500, max_depth=5, random_state=42,
                tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor'
            )
        elif model_choice == 'rf':
            eval_model = RandomForestRegressor(
                n_estimators=500, max_depth=10, random_state=42, n_jobs=4
            )
        else:
            eval_model = ExtraTreesRegressor(
                n_estimators=500, max_depth=10, random_state=42, n_jobs=4
            )
        scoring = 'neg_mean_absolute_error'
    else:
        opt_metric = 'f1'  # 기본값 설정
        if model_choice == 'xgb':
            eval_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=5, random_state=42,
                tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor'
            )
        elif model_choice == 'rf':
            eval_model = RandomForestClassifier(
                n_estimators=500, max_depth=10, random_state=42, n_jobs=4
            )
        else:
            eval_model = ExtraTreesClassifier(
                n_estimators=500, max_depth=10, random_state=42, n_jobs=4
            )
        scoring = 'f1_weighted'
    
    # Cross validation으로 성능 평가
    if task_type == 'regression':
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    else:
        kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(eval_model, X_selected, y.values, cv=kf, scoring=scoring)
    best_score = np.mean(cv_scores)
    
    # 결과 출력
    print(f"\n✅ Selected top {n_top_groups} feature groups")
    print(f"   Total features: {len(best_indices)}")
    if task_type == 'regression':
        if scoring == 'neg_mean_absolute_error':
            print(f"   MAE: {-best_score:.4f} ± {np.std(cv_scores):.4f}")
        else:
            print(f"   R2: {best_score:.4f} ± {np.std(cv_scores):.4f}")
    else:
        print(f"   F1 Score: {best_score:.4f} ± {np.std(cv_scores):.4f}")
    
    # 선택된 그룹 이름 출력
    print("\n📊 Selected feature groups:")
    for i, group_idx in enumerate(best_groups[:10], 1):  # 상위 10개만 출력
        group_name = reduced_feature_names[group_idx]
        importance = reduced_importance[group_idx]
        print(f"   {i:2d}. {group_name:<40} (importance: {importance:.4f})")
    if n_top_groups > 10:
        print(f"   ... and {n_top_groups - 10} more groups")

# 13. Hyperparameter Tuning
if fs_method.lower() == 'auto':
    print(f"\n🔧 Hyperparameter Tuning for {opt_metric.upper()} optimized features with {model_names[model_choice]}...")
else:
    print(f"\n🔧 Hyperparameter Tuning for Top {n_top_groups} feature groups with {model_names[model_choice]}...")

# 모델별 하이퍼파라미터 설정
if task_type == 'regression':
    if model_choice == 'xgb':
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
        model = xgb.XGBRegressor(
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor'
        )
    elif model_choice == 'rf':
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['sqrt', 'log2', None]
        }
        model = RandomForestRegressor(random_state=42, n_jobs=4)
    else:  # 'et'
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['sqrt', 'log2', None]
        }
        model = ExtraTreesRegressor(random_state=42, n_jobs=4)
else:  # classification
    if model_choice == 'xgb':
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [1, 2, 3]  # 불균형 데이터 처리용
        }
        model = xgb.XGBClassifier(
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_choice == 'rf':
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['sqrt', 'log2', None],
            "class_weight": ['balanced', None]  # 불균형 데이터 처리용
        }
        model = RandomForestClassifier(random_state=42, n_jobs=4)
    else:  # 'et'
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 5, 7, 10, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['sqrt', 'log2', None],
            "class_weight": ['balanced', None]  # 불균형 데이터 처리용
        }
        model = ExtraTreesClassifier(random_state=42, n_jobs=4)

X_best = X_scaled[:, best_indices]

# scoring 설정
if task_type == 'regression':
    scoring = 'neg_mean_absolute_error' if opt_metric=='mae' else 'r2'
else:
    scoring = 'f1_weighted' if opt_metric=='f1' else 'accuracy'

# Baseline(초기 하이퍼파라미터) 모델 정의
if task_type == 'regression':
    if model_choice == 'xgb':
        baseline_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            objective='reg:squarederror'
        )
    elif model_choice == 'rf':
        baseline_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=4
        )
    else:  # 'et'
        baseline_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=4
        )
else:  # classification
    if model_choice == 'xgb':
        baseline_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_choice == 'rf':
        baseline_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=4
        )
    else:  # 'et'
        baseline_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=4
        )

# Baseline 모델 5-fold CV로 성능 구함
baseline_cv_score = cross_val_score(baseline_model, X_best, y.values, cv=n_fold, scoring=scoring)
baseline_cv_mean = baseline_cv_score.mean()

# RandomizedSearchCV 실행
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=30,
    scoring=scoring,
    cv=5,
    verbose=1,
    n_jobs=4,
    random_state=42
)
search.fit(X_best, y.values)
tuned_cv_mean = search.best_score_

# 비교 후 best_model 선정
if task_type == 'regression':
    if opt_metric == 'mae':
        print(f"\nBaseline MAE: {-baseline_cv_mean:.4f}")
        print(f"Best MAE Score (Tuned): {-tuned_cv_mean:.4f}")
        use_tuned = tuned_cv_mean > baseline_cv_mean  # neg_mae는 클수록 좋음
    else:
        print(f"\nBaseline R2: {baseline_cv_mean:.4f}")
        print(f"Best R2 Score (Tuned): {tuned_cv_mean:.4f}")
        use_tuned = tuned_cv_mean > baseline_cv_mean
else:  # classification
    if opt_metric == 'f1':
        print(f"\nBaseline F1: {baseline_cv_mean:.4f}")
        print(f"Best F1 Score (Tuned): {tuned_cv_mean:.4f}")
    else:
        print(f"\nBaseline Accuracy: {baseline_cv_mean:.4f}")
        print(f"Best Accuracy Score (Tuned): {tuned_cv_mean:.4f}")
    use_tuned = tuned_cv_mean > baseline_cv_mean

if use_tuned:
    best_model = search.best_estimator_
    print("✅ Tuned model selected (RandomizedSearchCV)")
    best_params_used = search.best_params_
else:
    best_model = baseline_model
    best_model.fit(X_best, y.values)
    print("✅ Baseline model selected (default parameters)")
    best_params_used = baseline_model.get_params()

# 14. 최종 모델 평가 (n_fold-fold CV)
print("\n" + "="*60)
if fs_method.lower() == 'auto':
    print(f"FINAL MODEL EVALUATION ({model_names[model_choice]}, {opt_metric.upper()}-optimized)")
else:
    print(f"FINAL MODEL EVALUATION ({model_names[model_choice]}, Top {n_top_groups} groups)")
print("="*60)

if task_type == 'regression':
    X_selected = X_scaled[:, best_indices]
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    # Feature selection 방법에 따라 정보 설정
    if fs_method.lower() == 'auto':
        feature_selection_info = f"{opt_metric.upper()}-optimized"
    else:
        feature_selection_info = f"Top {n_top_groups} groups"
    
    # 평가 지표 저장
    r2_scores, mae_scores, mae_original_scores = [], [], []
    
    # Feature importance 저장 추가
    feature_importances_final = []
    
    # 예측값 수집
    all_y_true_log = np.zeros(len(y))
    all_y_pred_log = np.zeros(len(y))
    
    print(f"\n📊 Evaluating {model_names[model_choice]} with {feature_selection_info} features...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected)):
        model_clone = best_model.__class__(**best_model.get_params())
        model_clone.fit(X_selected[train_idx], y.iloc[train_idx])
        
        # Feature importance 수집
        feature_importances_final.append(model_clone.feature_importances_)
        
        # 예측
        y_pred_log = model_clone.predict(X_selected[test_idx])
        y_true_log = y.iloc[test_idx]
        
        # 저장
        all_y_true_log[test_idx] = y_true_log
        all_y_pred_log[test_idx] = y_pred_log
        
        # 성능 계산
        r2_scores.append(r2_score(y_true_log, y_pred_log))
        mae_scores.append(mean_absolute_error(y_true_log, y_pred_log))
        
    # Parity plot
    plt.figure(figsize=(7, 7))
    plt.scatter(all_y_true_log, all_y_pred_log, alpha=0.5, s=28, edgecolor='k', linewidth=0.2)
    plt.plot([all_y_true_log.min(), all_y_true_log.max()],
             [all_y_true_log.min(), all_y_true_log.max()],
             'r--', lw=2, label='Ideal')
    
    # 성능 지표 계산
    parity_r2 = r2_score(all_y_true_log, all_y_pred_log)
    parity_mae = mean_absolute_error(all_y_true_log, all_y_pred_log)
    parity_rmse = root_mean_squared_error(all_y_true_log, all_y_pred_log)
    
    # 플롯 내 텍스트로 표시
    #plt.text(0.05, 0.95, 
    #         f"$R^2$ = {parity_r2:.4f}\nMAE = {parity_mae:.4f}\nRMSE = {parity_rmse:.4f}",
    #         ha='left', va='top', transform=plt.gca().transAxes, fontsize=12,
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.text(0.05, 0.95, 
             f"$R^2$ = {parity_r2:.4f}\nMAE = {parity_mae:.4f}",
             ha='left', va='top', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.xlabel('Actual $y$ (log scale)', fontsize=13)
    plt.ylabel('Predicted $y$ (log scale)', fontsize=13)
    plt.title(f'Parity Plot ({model_names[model_choice]}, {feature_selection_info}, {n_fold}-fold CV)', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:  # classification
    X_selected = X_scaled[:, best_indices]
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    # Feature selection 방법에 따라 정보 설정
    if fs_method.lower() == 'auto':
        feature_selection_info = f"{opt_metric.upper()}-optimized"
    else:
        feature_selection_info = f"Top {n_top_groups} groups"
    
    # Feature importance 저장 추가
    feature_importances_final = []
    
    # 전체 예측값 수집
    all_y_true = np.zeros(len(y))
    all_y_pred_proba = np.zeros(len(y))
    
    print(f"\n📊 Evaluating {model_names[model_choice]} with {feature_selection_info} features...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected, y)):
        model_clone = best_model.__class__(**best_model.get_params())
        model_clone.fit(X_selected[train_idx], y.iloc[train_idx])
        
        feature_importances_final.append(model_clone.feature_importances_)

        # 예측 확률
        y_pred_proba = model_clone.predict_proba(X_selected[test_idx])[:, 1]
        all_y_true[test_idx] = y.iloc[test_idx]
        all_y_pred_proba[test_idx] = y_pred_proba
    
    # ROC Curve와 최적 threshold 찾기
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic으로 최적 threshold 찾기
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 최적 threshold로 예측
    all_y_pred = (all_y_pred_proba >= optimal_threshold).astype(int)
    
    # ROC Curve 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_names[model_choice]} ({feature_selection_info})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    # PR Curve 계산
    precisions, recalls, pr_thresholds = precision_recall_curve(all_y_true, all_y_pred_proba)
    
    # threshold별 f1-score 계산
    f1s = []
    for thr in pr_thresholds:
        preds = (all_y_pred_proba >= thr).astype(int)
        f1s.append(f1_score(all_y_true, preds, average='binary'))  # 'weighted' 혹은 'binary'
    
    # F1이 최대인 임계값 선택
    best_idx = np.argmax(f1s)
    optimal_threshold = pr_thresholds[best_idx]
    
    print(f"Optimal threshold (max F1): {optimal_threshold:.4f} | Max F1: {f1s[best_idx]:.4f}")
    
    # 최적 threshold로 예측
    all_y_pred = (all_y_pred_proba >= optimal_threshold).astype(int)
    
    # PR 커브 시각화 (최적 F1 지점 표시)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label='PR curve')
    plt.scatter(recalls[best_idx], precisions[best_idx], color='red', s=100,
                label=f'Optimal threshold = {optimal_threshold:.3f}\nMax F1 = {f1s[best_idx]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_names[model_choice]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Easy-plane', 'Easy-axis'],
                yticklabels=['Easy-plane', 'Easy-axis'])
    plt.title(f'Confusion Matrix - {model_names[model_choice]} ({feature_selection_info})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 성능 지표 출력
    print(f"\n📊 Performance Metrics ({n_fold}-fold CV):")
    print(f"   AUC Score:        {roc_auc:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   Accuracy:         {accuracy_score(all_y_true, all_y_pred):.4f}")
    print(f"   F1 Score:         {f1_score(all_y_true, all_y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, 
                              target_names=['Easy-plane', 'Easy-axis']))
    pos_label = 1
    precision_pos = precision_score(all_y_true, all_y_pred, pos_label=pos_label, average='binary')
    recall_pos = recall_score(all_y_true, all_y_pred, pos_label=pos_label, average='binary')
    f1_pos = f1_score(all_y_true, all_y_pred, pos_label=pos_label, average='binary')
    
    print(f"\nPositive Class ({pos_label}) [e.g., easy-axis] Metrics:")
    print(f"   Precision (positive): {precision_pos:.4f}")
    print(f"   Recall (positive):    {recall_pos:.4f}")
    print(f"   F1-score (positive):  {f1_pos:.4f}")
    
# Feature importance 분석
importances_final_array = np.array(feature_importances_final)
mean_importance_final = importances_final_array.mean(axis=0)
std_importance_final = importances_final_array.std(axis=0)

# 선택된 feature의 원래 이름 가져오기
selected_feature_names = [feature_names[i] for i in best_indices]

# Feature index → Group index 매핑 생성
feature_to_group = {}
for group_idx, features in original_feature_groups.items():
    for feat_idx in features:
        feature_to_group[feat_idx] = group_idx

# Feature importance를 group별로 합산
def aggregate_group_importance(feature_importances, feature_indices, feature_to_group_map, reduced_names):
    """
    개별 feature importance를 원래 group으로 다시 합산
    """
    group_importance = {}
    for i, feat_idx in enumerate(feature_indices):
        group_idx = feature_to_group_map[feat_idx]
        if group_idx not in group_importance:
            group_importance[group_idx] = 0
        group_importance[group_idx] += feature_importances[i]
    
    group_items = []
    for group_idx, importance in group_importance.items():
        group_name = reduced_names[group_idx]
        group_items.append((group_name, importance, group_idx))
    
    return group_items

# Group별 importance 계산
group_importances = aggregate_group_importance(
    mean_importance_final, 
    best_indices,
    feature_to_group,
    reduced_feature_names
)

# 중요도 기준으로 정렬
group_importances_sorted = sorted(group_importances, key=lambda x: x[1], reverse=True)

# Top 20 feature groups 시각화
top_n = min(20, len(group_importances_sorted))
top_groups = group_importances_sorted[:top_n]

# 이름과 값 분리
top_names = [item[0] for item in top_groups]
top_values = [item[1] for item in top_groups]

# 가로 막대그래프 그리기
plt.figure(figsize=(12, 10))

# 색상 설정 - 중요도에 따라 그라데이션
colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

bars = plt.barh(range(top_n), top_values, color=colors)

# Y축 라벨 설정
plt.yticks(range(top_n), top_names, fontsize=11)
plt.gca().invert_yaxis()

# 축 및 제목 설정
plt.xlabel("Mean Feature Importance", fontsize=12)
plt.title(f"Top {top_n} Important Feature Groups ({model_names[model_choice]}, {n_fold}-Fold CV)", fontsize=14)

# 각 막대에 값 표시
for i, (bar, value) in enumerate(zip(bars, top_values)):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{value:.4f}', ha='left', va='center', fontsize=9)

# 그리드 추가
plt.grid(True, alpha=0.3, axis='x')

# 여백 조정
plt.tight_layout()
plt.xlim(0, max(top_values) * 1.15)

plt.show()
'''
# 각 fold별 importance 변동성 시각화
print(f"\n📊 Feature Importance Stability Analysis")
print("="*60)

# feature_importances_final이 비어있지 않은지 확인
if len(feature_importances_final) == 0:
    print("Warning: No feature importance data collected during cross-validation.")
else:
    # Top 10 feature의 fold별 변동성 확인
    n_top_groups = min(10, len(group_importances_sorted))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx in range(n_top_groups):
        ax = axes[idx]
        group_name, _, group_idx = group_importances_sorted[idx]
        
        # 각 fold에서 해당 group의 importance 계산
        fold_importances = []
        
        for fold_importance in feature_importances_final:
            # 길이 체크
            if len(fold_importance) != len(best_indices):
                print(f"Warning: Feature importance length mismatch: {len(fold_importance)} vs {len(best_indices)}")
                continue
                
            group_imp = 0
            for i, feat_idx in enumerate(best_indices):
                if feat_idx in feature_to_group and feature_to_group[feat_idx] == group_idx:
                    group_imp += fold_importance[i]
            fold_importances.append(group_imp)
        
        # fold_importances가 비어있지 않은 경우에만 boxplot 그리기
        if fold_importances:
            ax.boxplot(fold_importances)
            ax.set_title(f"{group_name[:30]}{'...' if len(group_name) > 30 else ''}", fontsize=10)
            ax.set_xlabel('All Folds')
            ax.set_ylabel('Importance')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f"{group_name[:30]}{'...' if len(group_name) > 30 else ''}", fontsize=10)
    
    # 사용하지 않는 subplot 숨기기
    for idx in range(n_top_groups, 10):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Feature Importance Variability Across {n_fold} Folds ({model_names[model_choice]}, Top {n_top_groups} Features)", fontsize=14)
    plt.tight_layout()
    plt.show()
'''
# 통계 요약
print("\nTop 20 Feature Groups Summary:")
print("-"*80)
print(f"{'Rank':<5} {'Feature Group':<40} {'Mean Importance':<15} {'Std Dev':<15}")
print("-"*80)

for rank, (name, importance, group_idx) in enumerate(group_importances_sorted[:20], 1):
    # 각 fold에서의 importance 계산하여 표준편차 구하기
    fold_importances = []
    for fold_importance in feature_importances_final:
        group_imp = 0
        for i, feat_idx in enumerate(best_indices):
            if feature_to_group[feat_idx] == group_idx:
                group_imp += fold_importance[i]
        fold_importances.append(group_imp)
    
    std_dev = np.std(fold_importances)
    print(f"{rank:<5} {name[:40]:<40} {importance:<15.6f} {std_dev:<15.6f}")

# 결과 저장
if isinstance(best_indices, np.ndarray):
    selected_features_list = best_indices.tolist()
else:
    selected_features_list = best_indices

# task_type에 따라 성능 지표 다르게 저장
if task_type == 'regression':
    performance_metrics = {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'mae_log_mean': float(np.mean(mae_scores)),
        'mae_log_std': float(np.std(mae_scores)),
        'mae_original_mean': float(np.mean(mae_original_scores)),
        'mae_original_std': float(np.std(mae_original_scores))
    }
else:  # classification
    performance_metrics = {
        'auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(all_y_true, all_y_pred)),
        'f1_score': float(f1_score(all_y_true, all_y_pred, average='weighted')),
        'precision': float(precision_score(all_y_true, all_y_pred, average='weighted')),
        'recall': float(recall_score(all_y_true, all_y_pred, average='weighted')),
        'precision_positive': float(precision_pos),
        'recall_positive': float(recall_pos),
        'f1_positive': float(f1_pos)
    }

feature_importance_results = {
    'task_type': task_type,
    'model_type': model_choice,
    'feature_selection_method': 'auto' if fs_method.lower() == 'auto' else f'top_{n_top_groups}_groups',
    'optimization_metric': opt_metric if fs_method.lower() == 'auto' else 'none',
    'selected_features': selected_features_list,
    'feature_names': [feature_names[i] for i in best_indices],
    'mean_importances': mean_importance_final.tolist(),
    'std_importances': std_importance_final.tolist(),
    'group_importances': [(name, float(imp)) for name, imp, _ in group_importances_sorted],
    'best_params': search.best_params_ if use_tuned else best_params_used,
    'performance_metrics': performance_metrics
}

# 파일명에 task_type도 포함
if fs_method.lower() == 'auto':
    filename = f'feature_importance_results_{task_type}_{model_choice}_{opt_metric}.json'
else:
    filename = f'feature_importance_results_{task_type}_{model_choice}_top{n_top_groups}groups.json'

with open(filename, 'w') as f:
    json.dump(feature_importance_results, f, indent=2)

print(f"\n✅ Feature importance analysis saved to: {filename}")

# 성능 요약 출력
print("\n" + "="*60)
print(f"FINAL MODEL PERFORMANCE SUMMARY ({model_names[model_choice]})")
print("="*60)

print(f"\n📊 Performance Metrics ({n_fold}-fold CV):")

if task_type == 'regression':
    print(f"   R² Score:         {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"   MAE (log scale):  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"   MAE (original):   {np.mean(mae_original_scores):.4f} ± {np.std(mae_original_scores):.4f}")
else:
    # 분류 성능 재출력
    print(f"   AUC Score:        {roc_auc:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   Accuracy:         {accuracy_score(all_y_true, all_y_pred):.4f}")
    print(f"   F1 Score:         {f1_score(all_y_true, all_y_pred, average='weighted'):.4f}")
    print(f"   Precision:        {precision_score(all_y_true, all_y_pred, average='weighted'):.4f}")
    print(f"   Recall:           {recall_score(all_y_true, all_y_pred, average='weighted'):.4f}")
    print(f"   Precision (positive): {precision_pos:.4f}")
    print(f"   Recall (positive):    {recall_pos:.4f}")
    print(f"   F1-score (positive):  {f1_pos:.4f}")
    
print(f"\n📊 Selected Features:")
print(f"   Total features selected: {len(best_indices)}")
print(f"   Total feature groups:    {len(best_groups)}")

print(f"\n📊 Model Configuration:")
print(f"   Model Type: {model_names[model_choice]}")
print(f"   Optimization Metric: {opt_metric.upper()}")
print(f"   Best Hyperparameters: {best_params_used if 'best_params_used' in locals() else 'Default parameters'}")

# SHAP Analysis 추가 코드 (Group 단위 중심)
# 이 코드를 최종 모델 평가 후에 추가하세요

# === SHAP Analysis ===
print("\n" + "="*60)
print("SHAP ANALYSIS (Feature Group Level)")
print("="*60)

# SHAP은 XGBoost와 Tree 기반 모델에서 잘 작동합니다
if model_choice in ['xgb', 'rf', 'et']:
    print(f"\n📊 Computing SHAP values for {model_names[model_choice]}...")
    
    # 전체 데이터로 최종 모델 재학습 (SHAP 분석용)
    final_model = best_model.__class__(**best_model.get_params())
    final_model.fit(X_selected, y)
    
    # SHAP Explainer 생성
    if model_choice == 'xgb':
        explainer = shap.Explainer(final_model)
    else:
        explainer = shap.TreeExplainer(final_model)
    
    # SHAP 값 계산
    if len(X_selected) > 1000:
        sample_idx = np.random.choice(len(X_selected), 1000, replace=False)
        X_sample = X_selected[sample_idx]
        y_sample = y.iloc[sample_idx]
        print(f"   Computing SHAP values for 1000 samples...")
    else:
        X_sample = X_selected
        y_sample = y
        print(f"   Computing SHAP values for all {len(X_selected)} samples...")
    
    shap_values = explainer(X_sample)
    
    if task_type == 'classification' and len(shap_values.values.shape) == 3:
        # Binary classification: positive class (index 1) 선택
        shap_values.values = shap_values.values[:, :, 1]
    
    # 선택된 feature 이름
    selected_feature_names_shap = [feature_names[i] for i in best_indices]
    
    # 1. Feature별 SHAP 값을 Group으로 합산
    n_samples = len(X_sample)
    group_shap_values = {}
    
    for group_idx in set(feature_to_group[feat_idx] for feat_idx in best_indices):
        group_shap_values[group_idx] = np.zeros(n_samples)
    
    # 각 샘플의 SHAP 값을 group별로 합산
    for i, feat_idx in enumerate(best_indices):
        group_idx = feature_to_group[feat_idx]
        group_shap_values[group_idx] += shap_values.values[:, i]
    
    # 2. Group별 평균 절대 SHAP 값 계산 (중요도)
    shap_group_importance = {}
    for group_idx, shap_vals in group_shap_values.items():
        group_name = reduced_feature_names[group_idx]
        importance = np.abs(shap_vals).mean()
        shap_group_importance[group_name] = (importance, shap_vals)
    
    # 중요도 순으로 정렬
    shap_group_sorted = sorted(shap_group_importance.items(), 
                              key=lambda x: x[1][0], reverse=True)
    
    # 3. Top 20 Group SHAP Importance Bar Plot
    top_n = min(20, len(shap_group_sorted))
    top_names = [item[0] for item in shap_group_sorted[:top_n]]
    top_values = [item[1][0] for item in shap_group_sorted[:top_n]]
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
    bars = plt.barh(range(top_n), top_values, color=colors)
    plt.yticks(range(top_n), top_names, fontsize=11)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    plt.title(f"Top {top_n} Feature Groups by SHAP Importance - {model_names[model_choice]}", fontsize=14)
    
    for i, (bar, value) in enumerate(zip(bars, top_values)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{value:.4f}', ha='left', va='center', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.xlim(0, max(top_values) * 1.15)
    plt.show()
    
    # 4. SHAP Summary Plot for Groups (Beeswarm style)
    # Top 10 group의 SHAP 값 분포를 보여줌
    print("\n📊 Creating Group-level SHAP Summary Plot...")
    
    top_10_groups = shap_group_sorted[:10]
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 각 group의 SHAP 값 분포를 violin plot으로 표시
    positions = []
    labels = []
    all_shap_data = []
    
    for i, (group_name, (importance, shap_vals)) in enumerate(top_10_groups):
        positions.append(i)
        labels.append(group_name[:40] + '...' if len(group_name) > 40 else group_name)
        all_shap_data.append(shap_vals)
    
    parts = ax.violinplot(all_shap_data, positions=positions, vert=False, 
                         showmeans=True, showextrema=True)
    
    # 색상 설정
    for pc, (name, (imp, vals)) in zip(parts['bodies'], top_10_groups):
        if vals.mean() > 0:
            pc.set_facecolor('red')
            pc.set_alpha(0.6)
        else:
            pc.set_facecolor('blue')
            pc.set_alpha(0.6)
    
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
    ax.set_title(f'SHAP Value Distribution by Feature Group - {model_names[model_choice]}', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # 5. 개별 예측 분해 (Group 단위)
    print("\n📊 Example prediction breakdown by feature groups...")
    
    # 중간값 예측 찾기
    predictions = final_model.predict(X_sample)
    median_idx = np.argmin(np.abs(predictions - np.median(predictions)))
    
    # 샘플 아이디 가져오기
    sample_id = df_clean.iloc[median_idx, 0]
    
    # 해당 샘플의 group별 SHAP 값
    sample_group_shaps = []
    sample_group_names = []
    
    for group_name, (_, shap_vals) in shap_group_sorted:
        sample_group_shaps.append(shap_vals[median_idx])
        sample_group_names.append(group_name)
    
    # 절대값이 큰 순서로 정렬 (상위 15개만)
    sorted_indices = np.argsort(np.abs(sample_group_shaps))[::-1][:15]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(sorted_indices))
    values = [sample_group_shaps[i] for i in sorted_indices]
    names = [sample_group_names[i] for i in sorted_indices]
    
    colors = ['red' if v > 0 else 'blue' for v in values]
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('SHAP value', fontsize=12)
    
    # expected_value 처리 - 배열인 경우와 스칼라인 경우 모두 처리
    if isinstance(explainer.expected_value, np.ndarray):
        if len(explainer.expected_value) > 0:
            base_value = float(explainer.expected_value[0])
        else:
            base_value = float(explainer.expected_value.item())
    else:
        base_value = float(explainer.expected_value)
    
    ax.set_title(f'Feature Group Contributions for ID: {sample_id}\n'
                f'Base value: {base_value:.3f}, '
                f'Prediction: {predictions[median_idx]:.3f}', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, values)):
        if value > 0:
            ha = 'left'
            x_offset = 0.001
        else:
            ha = 'right'
            x_offset = -0.001
        ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha=ha, va='center', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # 6. 모델 importance vs SHAP importance 비교
    print("\n📊 Comparing Model Feature Importance vs SHAP Importance:")
    print("-"*60)
    
    model_top10 = [name for name, _, _ in group_importances_sorted[:10]]
    shap_top10 = [name for name, _ in shap_group_sorted[:10]]
    
    print("Model Top 10 Feature Groups:")
    for i, name in enumerate(model_top10, 1):
        print(f"  {i:2d}. {name}")
    
    print("\nSHAP Top 10 Feature Groups:")
    for i, name in enumerate(shap_top10, 1):
        print(f"  {i:2d}. {name}")
    
    overlap = set(model_top10) & set(shap_top10)
    print(f"\nOverlap in Top 10: {len(overlap)}/10 feature groups")
    
    # 결과 저장 - expected_value 처리 추가
    if isinstance(explainer.expected_value, np.ndarray):
        if len(explainer.expected_value) > 0:
            base_value_for_json = float(explainer.expected_value[0])
        else:
            base_value_for_json = float(explainer.expected_value.item())
    else:
        base_value_for_json = float(explainer.expected_value)
    
    shap_results = {
        'shap_group_importance': [(name, float(vals[0])) for name, vals in shap_group_sorted],
        'base_value': base_value_for_json,
        'mean_abs_shap_by_group': {name: float(vals[0]) for name, vals in shap_group_sorted}
    }
    
    # 기존 JSON 파일에 추가
    with open(filename, 'r') as f:
        existing_results = json.load(f)
    
    existing_results['shap_analysis'] = shap_results
    
    with open(filename, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"\n✅ SHAP analysis results added to: {filename}")
    
else:
    print(f"⚠️  SHAP analysis is not available for {model_choice}")

print("\n" + "="*60)
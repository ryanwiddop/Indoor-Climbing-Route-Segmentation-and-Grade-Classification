from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix


def main():
    all_pairs = pd.read_csv("all_pairs.csv")
    image_to_pairs = {image_file: all_pairs[all_pairs["image_file"] == image_file].to_dict("records") for image_file in all_pairs["image_file"].unique()}
    
    # XGBoost Model Training
    rng = np.random.default_rng(42)
    all_images = list(image_to_pairs.keys())
    rng.shuffle(all_images)
    
    # split into train, test, and validation sets
    split1 = int(0.80 * len(all_images))
    split2 = int(0.10 * len(all_images))
    train_images = all_images[:split1]
    test_images = all_images[split1:split1+split2]
    val_images = all_images[split1+split2:]
    
    
    train_pairs = [pair for image_file in train_images for pair in image_to_pairs[image_file]]
    test_pairs = [pair for image_file in test_images for pair in image_to_pairs[image_file]]
    train_pairs_df = pd.DataFrame(train_pairs)
    test_pairs_df = pd.DataFrame(test_pairs)
    print(f"Total images: {len(all_images)}, Train images: {len(train_images)}, Test images: {len(test_images)}")
    print(f"Total train pairs: {len(train_pairs)}, Total test pairs: {len(test_pairs)}")
    
    X_train = train_pairs_df.drop(columns=["same_route", "image_file"])
    y_train = train_pairs_df["same_route"].astype(int)
    X_test = test_pairs_df.drop(columns=["same_route", "image_file"])
    y_test = test_pairs_df["same_route"].astype(int)
    X_val = pd.DataFrame([pair for image_file in val_images for pair in image_to_pairs[image_file]]).drop(columns=["same_route", "image_file"])
    y_val = pd.DataFrame([pair for image_file in val_images for pair in image_to_pairs[image_file]])["same_route"].astype(int)
    
    # negative_count = (y_train == 0).sum()
    # positive_count = (y_train == 1).sum()
    # scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1
    
    X_train["cat_min"] = X_train["cat_min"].astype("category")
    X_train["cat_max"] = X_train["cat_max"].astype("category")
    X_test["cat_min"] = X_test["cat_min"].astype("category")
    X_test["cat_max"] = X_test["cat_max"].astype("category")
    X_val["cat_min"] = X_val["cat_min"].astype("category")
    X_val["cat_max"] = X_val["cat_max"].astype("category")
    
    xgb_train = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True
    )
    xgb_test = xgb.DMatrix(
        X_test,
        label=y_test,
        enable_categorical=True
    )
    xgb_val = xgb.DMatrix(
        X_val,
        label=y_val,
        enable_categorical=True
    )
    
    params = {
        'objective': 'binary:logistic',
        "eval_metric": "aucpr",
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        "min_child_weight": 3,
        # "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "verbosity": 0
    }
    model = xgb.train(
        params=params, 
        dtrain=xgb_train, 
        num_boost_round=2000,
        evals=[(xgb_train, "train"), (xgb_val, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    print(f"Best iteration: {model.best_iteration}")

    probs_val = model.predict(xgb_val)
    probs_test = model.predict(xgb_test)
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.3, 0.9, 0.1):
        preds = (probs_val >= threshold).astype(int)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        print(
            threshold,
            precision,
            recall,
            f1,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold}, Best F1: {best_f1}")
    preds = (probs_test >= best_threshold).astype(int)
    importance = model.get_score(importance_type='gain')
    
    print("\nModel Evaluation:")
    print("PR-AUC:", average_precision_score(y_test, probs_test))
    print("Precision:", precision_score(y_test, preds, zero_division=0))
    print("Recall:", recall_score(y_test, preds, zero_division=0))
    print("F1:", f1_score(y_test, preds, zero_division=0))
    print(confusion_matrix(y_test, preds))
    
    total_score = sum(importance.values())
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{score / total_score:.2%}: {feature}")
        
        
if __name__ == "__main__":
    main()
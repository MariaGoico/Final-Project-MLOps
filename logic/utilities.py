from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
# import torch

def set_seed(seed=42):
    import os, random, numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    """torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""


class DataValidator:
    """
    Handles model validation strategy for imbalanced classification.
    Provides:
    - Stratified K-Fold for CV
    - Train / Validation / Test split
    - Class imbalance utilities
    """

    @staticmethod
    def get_stratified_kfold(n_splits=5, random_state=42):
        """
        Get StratifiedKFold for cross-validation.
        Important for imbalanced datasets to maintain class distribution.
        """
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    @staticmethod
    def split_train_test(X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets with stratification.
        (Kept for backward compatibility)
        """
        if y is None:
            raise ValueError("y cannot be None for stratified split")

        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

    @staticmethod
    def split_train_val_test(
        X,
        y,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    ):
        """
        Split data into Train / Validation / Test sets with stratification.

        Default:
        - Train: 70%
        - Validation: 15%
        - Test: 15%
        """

        if y is None:
            raise ValueError("y cannot be None for stratified split")

        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        # First split: Train vs Temp (Val + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_size + test_size),
            stratify=y,
            random_state=random_state
        )

        # Second split: Validation vs Test
        val_ratio = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp,
            random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def get_class_weights(y):
        """
        Calculate class weights for imbalanced dataset.
        """
        from sklearn.utils.class_weight import compute_class_weight

        if y is None:
            raise ValueError("y cannot be None when computing class weights")

        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        return dict(zip(classes, weights))

    @staticmethod
    def get_scale_pos_weight(y):
        """
        Calculate scale_pos_weight parameter for XGBoost.
        """
        if y is None:
            raise ValueError("y cannot be None when computing scale_pos_weight")

        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)

        if pos_count == 0:
            raise ValueError("No positive samples found in y")

        return neg_count / pos_count

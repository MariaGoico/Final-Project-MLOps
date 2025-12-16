from sklearn.model_selection import StratifiedKFold
import numpy as np


# COGER CON PINZAS -------------------------------------------------------- IGUAL ME LO CARGO 
class DataValidator:
    """
    Handles model validation strategy for imbalanced classification
    """
    
    @staticmethod
    def get_stratified_kfold(n_splits=5, random_state=42):
        """
        Get StratifiedKFold for cross-validation.
        Important for imbalanced datasets to maintain class distribution.
        """
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    @staticmethod
    def split_train_test(X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets with stratification
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, 
                              stratify=y, random_state=random_state)
    
    @staticmethod
    def get_class_weights(y):
        """
        Calculate class weights for imbalanced dataset
        """
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    @staticmethod
    def get_scale_pos_weight(y):
        """
        Calculate scale_pos_weight parameter for XGBoost
        Useful for imbalanced classification
        """
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        return neg_count / pos_count if pos_count > 0 else 1.0
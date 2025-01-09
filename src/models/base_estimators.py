from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_optimized_hgb():
    """Returns an optimized HistGradientBoostingClassifier with fixed configuration"""
    return HistGradientBoostingClassifier(
        
        early_stopping=True,
        l2_regularization=1.703509977712629,
        learning_rate=0.05281747140114072,
        max_bins=88,
        max_depth=6,
        max_iter=609,
        max_leaf_nodes=33,
        min_samples_leaf=88,
        n_iter_no_change=9,
        tol=0.016877246471796614,
        validation_fraction=0.761534350092348,
        random_state=42,
    )

def get_optimized_rf():
    """Returns an optimized RandomForestClassifier with fixed configuration"""
    return RandomForestClassifier(
        bootstrap=True,
        ccp_alpha= 0.003437414842164783,
        criterion='entropy',
        max_depth=30,
        max_features='sqrt',
        max_leaf_nodes=34,
        max_samples= 0.7445782866469565,
        min_impurity_decrease=0.008824782106432694,
        min_samples_leaf=7,
        min_samples_split=22,
        min_weight_fraction_leaf=0.004758516640936291,
        n_estimators=103,
        oob_score=False,
        random_state=42
    )
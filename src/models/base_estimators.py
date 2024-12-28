from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_optimized_hgb():
    """Returns an optimized HistGradientBoostingClassifier with fixed configuration"""
    return HistGradientBoostingClassifier(
        class_weight={0: 1, 1: 1},
        early_stopping=True,
        l2_regularization=2.294280717958801,
        learning_rate=0.23040698858775321,
        max_bins=115,
        max_depth=12,
        max_iter=11,
        max_leaf_nodes=33,
        min_samples_leaf=81,
        n_iter_no_change=17,
        tol=0.030672753967226428,
        validation_fraction=0.8821528541761128,
        random_state=42,
    )

def get_optimized_rf():
    """Returns an optimized RandomForestClassifier with fixed configuration"""
    return RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.004681320432681322,
        class_weight={0: 1, 1: 1},
        criterion='entropy',
        max_depth=46,
        max_features='sqrt',
        max_leaf_nodes=30,
        max_samples=0.4343415014797324,
        min_impurity_decrease=0.022289899203164085,
        min_samples_leaf=1,
        min_samples_split=25,
        min_weight_fraction_leaf=0.001594042722389311,
        n_estimators=84,
        oob_score=False,
        random_state=42
    )
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_optimized_hgb():
    """Returns an optimized HistGradientBoostingClassifier with fixed configuration"""
    # return HistGradientBoostingClassifier( # for 20 year prediction
        
    #     early_stopping=True,
    #     l2_regularization=1.703509977712629,
    #     learning_rate=0.05281747140114072,
    #     max_bins=88,
    #     max_depth=6,
    #     max_iter=609,
    #     max_leaf_nodes=33,
    #     min_samples_leaf=88,
    #     n_iter_no_change=9,
    #     tol=0.016877246471796614,
    #     validation_fraction=0.761534350092348,
    #     random_state=42,
    # )

    return HistGradientBoostingClassifier( # for 10 year prediction
        early_stopping=True,
        l2_regularization=3.986258791896507,
        learning_rate=0.018696169165980464,
        max_bins=109,
        max_depth=3,
        max_iter=210,
        max_leaf_nodes=36,
        min_samples_leaf=18,
        n_iter_no_change=6,
        tol=0.09761986580976802,
        validation_fraction=0.35384021980609753,
        random_state=42,
    )

def get_optimized_rf():
    """Returns an optimized RandomForestClassifier with fixed configuration"""
    # return RandomForestClassifier( # for 20 year prediction
    #     bootstrap=True,
    #     ccp_alpha= 0.005315494844610223,
    #     criterion='entropy',
    #     max_depth=27,
    #     max_features='sqrt',
    #     max_leaf_nodes=60,
    #     max_samples= 0.7102831429146235,
    #     min_impurity_decrease=0.009355869445971656,
    #     min_samples_leaf=2,
    #     min_samples_split=14,
    #     min_weight_fraction_leaf=0.004758516640936291,
    #     # min_fraction_leaf=0.019669604717225522,
    #     n_estimators=137,
    #     oob_score=False,
    #     random_state=42
    # )

    return RandomForestClassifier( # for 10 year prediction
        bootstrap=True,
        ccp_alpha=0.007213780778147117,
        criterion='entropy',
        max_depth=32,
        max_features='sqrt',
        max_leaf_nodes=57,
        max_samples=0.6884621965986909,
        min_impurity_decrease=0.007645721955702123,
        min_samples_leaf=4,
        min_samples_split=8,
        min_weight_fraction_leaf=0.09436555948555557,
        n_estimators=158,
        oob_score=False,
        random_state=42
    )
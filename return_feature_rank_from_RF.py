from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt

def return_feature_rank_from_RF(X_train,y_train,X_features=[]):
# Build a forest and compute the impurity-based feature importances
    forest = ExtraTreesClassifier(n_estimators=20,random_state=0)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        if len(X_features)==0:
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        else:
            print("%d. feature %d [%s] (%f)" % (f + 1, indices[f], X_features[indices[f]], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize=(200,10))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r")
    plt.xticks(range(X_train.shape[1]), indices, rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    '''
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    '''
    plt.show()
    
    return (indices,importances)


import numpy as np

import utils

from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

train_features = np.asarray(utils.train_features)
test_features = np.asarray(utils.test_features)

train_labels = [1 - l for l in utils.train_labels]
test_labels = [1 - l for l in utils.test_labels]

dummy_clf = DummyClassifier(strategy="uniform")

def get_res(clf):
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    print(precision_recall_fscore_support(test_labels, preds, average = 'binary'))
    #fpr, tpr, thresholds = roc_curve(test_labels, preds, pos_label=0)
    
    print(roc_auc_score(test_labels, preds))

dummy_clf = DummyClassifier(strategy="uniform")
svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
mlp_clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=1, max_iter=500))

get_res(mlp_clf)

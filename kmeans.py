import utils
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score

train_features = np.asarray([t for t, l in zip(utils.train_features, utils.train_labels) if l == 0])
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

test_features = np.asarray(utils.test_features)
test_features = scaler.fit_transform(test_features)

kmeans = KMeans(n_clusters=6, random_state=0).fit(train_features)
clusters = kmeans.predict(test_features)

print(kmeans.cluster_centers_[clusters].shape)
print(test_features.shape)
#dist = np.linalg.norm(kmeans.cluster_centers_[clusters], test_features)
dist = np.mean((kmeans.cluster_centers_[clusters] - test_features) ** 2, axis = -1)
print(dist)
negatives = [i for i, d in enumerate(dist) if d > 0.7]
print(len(negatives))

preds = [1 if not i in negatives else 0 for i in range(len(dist))]

new_test_labels = [1 - l for l in utils.test_labels]

print(precision_recall_fscore_support(new_test_labels, preds, average = 'binary'))
print(roc_auc_score(new_test_labels, preds))





import tensorflow as tf
import utils
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score

batch_size = 64

train_features = np.asarray([t for t, l in zip(utils.train_features, utils.train_labels) if l == 0])
train_features = np.asarray(train_features)
test_features = np.asarray(utils.test_features)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)

train_labels = np.asarray([[1.0, 0.0] if l == 1 else [0.0, 1.0] for l in utils.train_labels])
test_labels = np.asarray([[1.0, 0.0] if l == 1 else [0.0, 1.0] for l in utils.test_labels])

num_features = 10
num_labels = 2

X = tf.placeholder(dtype = tf.float32, shape = [None, num_features])
Y = tf.placeholder(dtype = tf.float32, shape = [None, num_labels])

W1 = tf.get_variable(name = 'w1', shape = [10, 32], dtype = tf.float32)
W2 = tf.get_variable(name = 'w2', shape = [32, 64], dtype = tf.float32)

h1 = tf.matmul(X, W1)
h1 = tf.nn.relu(h1)

h2 = tf.matmul(h1, W2)
h2 = tf.nn.relu(h2)

h3 = tf.matmul(h2, tf.transpose(W2))
h3 = tf.nn.relu(h3)

logits = tf.matmul(h3, tf.transpose(W1))

dist = tf.reduce_sum((logits - X) ** 2, axis = -1)
loss = tf.nn.l2_loss(logits - X)
#loss = tf.reduce_mean(dist)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits))

#preds = tf.argmax(logits, axis = -1)

#acc = tf.reduce_mean(tf.cast(tf.equal(preds, tf.argmax(Y, axis = -1)), tf.float32))

opt = tf.train.AdamOptimizer(5e-5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(50000):
        inds = np.random.choice(len(train_features), size = batch_size)
        lv, _ = sess.run([loss, opt], feed_dict = {X: train_features[inds], Y: train_labels[inds]})
        if step % 100 == 0:
            print('step: %d, loss: %f'%(step, lv))
            #av = sess.run(acc, feed_dict = {X: test_features, Y: test_labels})
            #print('test accuracy:%f'%av)
    
    test_dist = sess.run(dist, feed_dict = {X: test_features, Y: test_labels})

print(test_dist)
new_test_labels = [1 - l for l in utils.test_labels]
predicts = [0 if d > 0.05 else 1 for d in test_dist]
print(precision_recall_fscore_support(new_test_labels, predicts, average = 'binary'))
print(roc_auc_score(new_test_labels, predicts))

    


        
        

import tensorflow as tf
import utils
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score

batch_size = 64

train_features = np.asarray(utils.train_features)
test_features = np.asarray(utils.test_features)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)[:, :, None]
test_features = scaler.fit_transform(test_features)[:, :, None]

train_labels = np.asarray([[1.0, 0.0] if l == 1 else [0.0, 1.0] for l in utils.train_labels])
test_labels = np.asarray([[1.0, 0.0] if l == 1 else [0.0, 1.0] for l in utils.test_labels])

num_features = 10
num_labels = 2

probs = [1.0] * len(train_labels)
for i, l in enumerate(utils.train_labels):
    if l == 1:
        probs[i] *= 3

total = sum(probs)
for i, p in enumerate(probs):
    probs[i] /= total

X = tf.placeholder(dtype = tf.float32, shape = [None, num_features, 1])
Y = tf.placeholder(dtype = tf.float32, shape = [None, num_labels])

h1 = tf.layers.conv1d(X, filters = 4, kernel_size = 3, padding='same')
h1 = tf.nn.relu(h1)

h2 = tf.layers.conv1d(h1, filters = 8, kernel_size = 3, padding='same')
h2 = tf.nn.relu(h2)

h2 = tf.reshape(h2, [-1, 8 * num_features])

h3 = tf.layers.dense(h2, 16, activation = tf.nn.relu)

logits = tf.layers.dense(h3, 2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits))

preds = tf.argmax(logits, axis = -1)

acc = tf.reduce_mean(tf.cast(tf.equal(preds, tf.argmax(Y, axis = -1)), tf.float32))

opt = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(50000):
        inds = np.random.choice(len(train_features), size = batch_size, p = probs)
        lv, _ = sess.run([loss, opt], feed_dict = {X: train_features[inds], Y: train_labels[inds]})
        if step % 100 == 0:
            print('step: %d, loss: %f'%(step, lv))
            av = sess.run(acc, feed_dict = {X: test_features, Y: test_labels})
            print('test accuracy:%f'%av)
    
    predicts = sess.run(preds, feed_dict = {X: test_features, Y: test_labels})

new_test_labels = [1 - l for l in utils.test_labels]

print(precision_recall_fscore_support(new_test_labels, predicts, average = 'macro'))
print(roc_auc_score(new_test_labels, predicts))

    


        
        

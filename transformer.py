import tensorflow as tf
import utils
import numpy as np
import embed
import attention

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, roc_auc_score

batch_size = 512

train_features = np.asarray(embed.train_features)
test_features = np.asarray(embed.test_features)

scaler = StandardScaler()
#train_features = scaler.fit_transform(train_features)[:, :, None]
#test_features = scaler.fit_transform(test_features)[:, :, None]

train_labels = np.asarray([[1.0, 0.0] if l == 1 else [0.0, 1.0] for l in utils.train_labels])
test_labels = np.asarray([[1.0, 0.0] if l == 1 else [0.0, 1.0] for l in utils.test_labels])

num_features = 10
num_labels = 2
num_dim = 64

probs = [1.0] * len(train_labels)
for i, l in enumerate(utils.train_labels):
    if l == 1:
        probs[i] *= 5

total = sum(probs)
for i, p in enumerate(probs):
    probs[i] /= total

X = tf.placeholder(dtype = tf.int32, shape = [None, num_features])
Y = tf.placeholder(dtype = tf.float32, shape = [None, num_labels])
is_training = tf.placeholder(dtype = tf.bool, shape = ())

config = attention.BertConfig(vocab_size=embed.max_ind + 1, 
                             hidden_size=num_dim, 
                             num_hidden_layers=2, 
                             num_attention_heads=4, 
                             intermediate_size=num_dim * 4)

model = attention.BertModel(config=config, 
                           is_training=is_training,
                           input_ids=X)

output_layer = model.get_pooled_output()


output_weights = tf.get_variable(
      "output_weights", [2, num_dim],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable(
      "output_bias", [2], initializer=tf.zeros_initializer())

keep_prob = tf.cond(is_training, lambda: 0.5, lambda: 1.0)
output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)
logits = tf.matmul(output_layer, output_weights, transpose_b=True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits))

preds = tf.argmax(logits, axis = -1)

acc = tf.reduce_mean(tf.cast(tf.equal(preds, tf.argmax(Y, axis = -1)), tf.float32))

opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        inds = np.random.choice(len(train_features), size = batch_size, p = probs)
        lv, _ = sess.run([loss, opt], 
            feed_dict = {X: train_features[inds], Y: train_labels[inds], is_training: True})
        if step % 100 == 0:
            print('step: %d, loss: %f'%(step, lv))
            av = sess.run(acc, 
                feed_dict = {X: test_features, Y: test_labels, is_training: False})
            print('test accuracy:%f'%av)
    
    predicts = sess.run(preds, 
        feed_dict = {X: test_features, Y: test_labels, is_training: False})

new_test_labels = [1 - l for l in utils.test_labels]

print(precision_recall_fscore_support(new_test_labels, predicts, average = 'macro'))
print(roc_auc_score(new_test_labels, predicts))

    


        
        

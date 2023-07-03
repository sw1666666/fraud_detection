import utils

num_features = 10
num_discrete = 100.0
num_train = len(utils.train_features)
num_test = len(utils.test_features)

cont_cols = [0, 5, 7]

def get_features(features):
    res = [[] for _ in range(num_features)]
    for line in features:
        for i, f in enumerate(line):
            res[i].append(f)
    return res

def get_all_features():
    res = get_features(utils.train_features)
    res_test = get_features(utils.test_features)
    for i, feature in enumerate(res_test):
        res[i].extend(feature)
    return res

def deal_continuous_features(features):
    inds = sorted(range(len(features)), key = lambda t: features[t])
    res = [int(float(i) / len(features) * num_discrete) for i in inds]
    return res

def get_bias(bias, features):
    new_bias = max(features)
    for i, f in enumerate(features):
        features[i] += bias
    return bias + new_bias

def transform_features():
    res = []
    bias = 0
    all_features = get_all_features()
    for i, features in enumerate(all_features):
        if i in cont_cols:
            features = deal_continuous_features(features)
        bias = get_bias(bias, features)
        res.append(features)
    
    train_features = [[] for _ in range(num_train)]
    test_features = [[] for _ in range(num_test)]

    for i, features in enumerate(res):
        for j, f in enumerate(features):
            if j < num_train:
                train_features[j].append(f)
            else:
                test_features[j - num_train].append(f)
    return train_features, test_features, bias

train_features, test_features, max_ind = transform_features()
            
print('====================================')
print(train_features[:10])
print(test_features[:10])    
print(max_ind)
    
    

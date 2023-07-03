import pickle

feature2id = {}

total_cols = 11
time_cols = [0]
discrete_cols = [1, 2, 3, 4, 6, 8, 9]
label_col = 10

with open('train.pkl', 'rb') as fin:
    train = pickle.load(fin)

with open('test.pkl', 'rb') as fin:
    test = pickle.load(fin)

def get_train():
    return train

def get_test():
    return test

def get_absolute_time(time):
    hour, minute = time.strip().split(':')
    return float(hour) + float(minute) / 60

def get_features_id_map(all_inputs):
    features = sorted(list(set(all_inputs)))
    f2id = dict(zip(features, range(len(features))))
    return f2id


def get_feature_map():
    total_features = [[] for _ in range(total_cols)]
    for line in train + test:
        for i, f in enumerate(line):
            if i in discrete_cols:
                total_features[i].append(f)
    
    for i in range(total_cols):
        if i in discrete_cols:
            feature2id[i] = get_features_id_map(total_features[i])

get_feature_map()

def get_features_and_labels(lines):
    features, labels = [], []
    for line in lines:
        feature = []
        for i, f in enumerate(line[:-1]):
            if i in time_cols:
                feature.append(get_absolute_time(line[i]))
            elif i in discrete_cols:
                feature.append(feature2id[i][f])
            else:
                feature.append(float(f))
        features.append(feature)
        labels.append(int(line[-1]))
    return features, labels

train_features, train_labels = get_features_and_labels(train)
test_features, test_labels = get_features_and_labels(test)

print(train_features[:100])
print(train_labels[:100])
        
    
                
    
    
    
    
    
    

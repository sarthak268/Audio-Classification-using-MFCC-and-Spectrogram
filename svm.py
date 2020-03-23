# Saves pickle files for SVM models for computing precision and recall

from sklearn.svm import SVC
import numpy as np
import pickle
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV

def make_data_label_files(arr):
    labels = []
    data = []

    for i in range(len(arr)):

        for j in range(len(arr[i])):

            labels.append(i)
            data.append(arr[i][j].reshape(-1))

    return data, labels

if (__name__ == '__main__'):

    #method = 'spectro'
    method = 'mfcc'
    noise = True

    if (noise):
        noise_str = '_noise'
    else:
        noise_str = ''

    with open (method + '_training' + noise_str + '.txt', 'rb') as fp:
        spectro_training = pickle.load(fp)

    spectro_features_train, spectro_labels_train = make_data_label_files(spectro_training)
    
    with open (method + '_validation' + noise_str + '.txt', 'rb') as fp:
        spectro_testing = pickle.load(fp)

    spectro_features_test, spectro_labels_test = make_data_label_files(spectro_testing)
    

    model_spectro = SVC(C=0.5, kernel='poly',gamma='auto')
    # cs = [0.001, 0.01, 0.1, 1, 10]
    # gammas = [0.001, 0.01, 0.1, 1, 10]

    # params = {'C':cs,'gamma':gammas}
    # grid_search = GridSearchCV(SVC(), params)
    spectro_features_train = np.nan_to_num(spectro_features_train)
    model_spectro.fit(spectro_features_train, spectro_labels_train)
    # grid_search.fit(spectro_features_test, spectro_labels_test)
    # print (grid_search.best_params)
   
    # clf_spectro.fit(spectro_features_train, spectro_labels_train)
    # pred_spectro = clf_spectro.predict(spectro_features_test)
    # print (accuracy_score(pred_spectro, spectro_labels_test))

    with open(method + noise_str + '_svm.pkl','wb') as f:
        pickle.dump(model_spectro, f)
    
    #model = grid_search.best_estimator_
    spectro_features_test = np.nan_to_num(spectro_features_test)
    pred_spectro = model_spectro.predict(spectro_features_test)
    print (accuracy_score(pred_spectro, spectro_labels_test))

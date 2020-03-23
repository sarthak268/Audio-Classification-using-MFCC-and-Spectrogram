# must run svm.py before this on the output of question 1 or 2. 
# It will save pickle files for SVM models.

from sklearn.svm import SVC
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score

def make_data_label_files(arr):
    labels = []
    data = []

    for i in range(len(arr)):

        for j in range(len(arr[i])):

            labels.append(i)
            data.append(arr[i][j].reshape(-1))

    return data, labels

model_path = './spectro_svm_poly_05.pkl'
test_file_path = './spectro_validation.txt'

with open(model_path,'rb') as f:
    model = pickle.load(f)
    
with open (test_file_path, 'rb') as fp:
    spectro_testing = pickle.load(fp)

spectro_features_test, spectro_labels_test = make_data_label_files(spectro_testing)   

spectro_features_test = np.nan_to_num(spectro_features_test)
    
pred_spectro = model.predict(spectro_features_test)
print (precision_score(pred_spectro, spectro_labels_test, average='weighted'))
print (recall_score(pred_spectro, spectro_labels_test, average='weighted'))
    

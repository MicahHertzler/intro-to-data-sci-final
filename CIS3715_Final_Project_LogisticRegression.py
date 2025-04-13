import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import random
random.seed(123)

df = pd.read_csv(r"C:\Users\Massimo Camuso\Desktop\Academics\Spring 2025\CIS 3715 (Principles of Data Science)\Final Project\preprocessed_loan_data.csv")

df.info()

counts = df['Default'].value_counts()
#print(counts) # Target variable is heavily imbalanced 

X = df.drop(['Default', 'LoanTerm_12', 'LoanPurpose_Auto', 'LoanID'], axis=1)
y = df['Default']


#print(X, y)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                            test_size=0.12, 
                                                            random_state=0)
#print("train_val: {}, test: {}".format(X_train_val.shape[0], X_test.shape[0]))

scale_cols = ['Age', 'Income', 'MonthsEmployed']
alr_scaled_cols = ['DTIRatio', 'CreditScore']
scaler = StandardScaler()
X_train_val_scaled = X_train_val.copy()

X_train_val_scaled[scale_cols] = scaler.fit_transform(X_train_val[scale_cols])
X_test_scaled = X_test.copy()
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
#^^ USE THESE VARIABLES FOR MODELING; NOT THE ORIGINAL VARIABLES
#print(X_train_val_scaled[alr_scaled_cols])
#print(X_test_scaled[alr_scaled_cols])

#---------------------------TRAINING WITH CROSS VALIDATION - FIT LASSO---------------

folds = 5

num_train_val = X_train_val_scaled.shape[0]

index_of_samples = np.arange(num_train_val)
shuffle(index_of_samples)

index_of_folds = np.array_split(index_of_samples, folds)
print(index_of_folds)

regularization_coefficient = np.arange(0.1, 15.5, 0.5)

#best_acc = 0.0
best_prec = 0.0
best_recall = 0.0
best_f1 = 0.0
best_reg_prec = 0.0
best_reg_recall = 0.0
best_reg_f1 = 0.0

for reg in regularization_coefficient:
    #10-fold cross validation
    #sum_acc = 0.0
    sum_prec = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    for fold in range(folds):
        index_of_folds_temp = index_of_folds.copy()
        
        valid_index = index_of_folds_temp[fold] #get the index of the validation set
        train_index = np.hstack([index_of_folds_temp[i] for i in range(folds) if i != fold]) #get the index of the training set
        
        # training set
        X_train = X_train_val_scaled.iloc[train_index]
        y_train = y_train_val.iloc[train_index]
        
        # validation set
        X_valid = X_train_val_scaled.iloc[valid_index]
        y_valid = y_train_val.iloc[valid_index]
                
        # build the model with different hyperparameters
        clf = LogisticRegression(penalty='l1', C=reg, solver='saga', max_iter=5000, class_weight='balanced')
        
        #train the model with the training set
        clf.fit(X_train, y_train)
        
        y_valid_pred = clf.predict(X_valid)
        #acc = accuracy_score(y_valid, y_valid_pred)
        #sum_acc += acc

        prec = precision_score(y_valid, y_valid_pred)
        sum_prec += prec

        recall = recall_score(y_valid, y_valid_pred)
        sum_recall += recall

        f1 = f1_score(y_valid, y_valid_pred)
        sum_f1 += f1
        
    
    #cur_acc = sum_acc / folds
    curr_prec = sum_prec / folds
    curr_recall = sum_recall / folds
    curr_f1 = sum_f1 / folds
    
    print("reg_coeff: {}, prec: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(1.0/reg, curr_prec, curr_recall, curr_f1))
    
    # store the best hyperparameter
    #if cur_acc > best_acc:
        #best_acc = cur_acc
        #best_reg = reg

    if curr_prec > best_prec:
        best_prec = curr_prec
        best_reg_prec = reg

    if curr_recall > best_recall:
        best_recall = curr_recall
        best_reg_recall = reg

    if curr_f1 > best_f1:
        best_f1 = curr_f1
        best_reg_f1 = reg

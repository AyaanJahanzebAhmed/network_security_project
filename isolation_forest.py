from tabulate import tabulate
import warnings
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

def onlygrid(X,y):
  #parameter grid for grid search 
  param_grid = {
      'n_estimators': [10,100],
      'contamination': [0.1, 0.15, 0.2],
      'max_samples': [.2,.25,100]
  }
  # train/test split 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
  isolation_forest = IsolationForest()
  # grid search 
  grid_search = GridSearchCV(estimator=isolation_forest, param_grid=param_grid, cv=5, scoring='f1')
  grid_search.fit(X_train)
  best_params = grid_search.best_params_
  # Fitting isolation forest with the best parameters 
  isolation_forest = IsolationForest(**best_params)
  isolation_forest.fit(X_train)

  y_pred = isolation_forest.predict(X_test)
  y_pred[y_pred == 1] = 0    # Normal
  y_pred[y_pred == -1] = 1    # Anomaly

  # printing metrics

  averagee = 'weighted'
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average=averagee)  
  recall = recall_score(y_test, y_pred, average=averagee)
  conf_matrix = confusion_matrix(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')  
  print("ISOLATION FOREST \n")
  print("best hyper parameters :",best_params)
  print("Accuracy:", accuracy)
  print("precision",precision)
  print("Recall:", recall)
  print("Confusion matrix: \n ")
  print(conf_matrix)
  print("\n")
  plt.figure(figsize=(8, 6))
  plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
  plt.title('Confusion Matrix')
  plt.colorbar()
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()


  # ROC Curve
  y_pred_prob = grid_search.best_estimator_.decision_function(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
  roc_auc = roc_auc_score(y_test, y_pred_prob)
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.show()

def pca(X,y):
  # parameter grid for grid search 
  param_grid = {
      'n_estimators': [10,100],
      'contamination': [0.1, 0.15, 0.2],
      'max_samples': [.2,.25,100]
  }
  # PCA 
  pca = PCA(n_components=2)
  X_reduce = pca.fit_transform(X)

  # train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  isolation_forest = IsolationForest()
  #grid search 
  grid_search = GridSearchCV(estimator=isolation_forest, param_grid=param_grid, cv=5, scoring='f1')
  grid_search.fit(X_train)
  best_params = grid_search.best_params_
  
  #Fitting isolation forest with the best parameters 
  isolation_forest = IsolationForest(**best_params)
  isolation_forest.fit(X_train)

  y_pred = isolation_forest.predict(X_test)
  y_pred[y_pred == 1] = 0    # Normal
  y_pred[y_pred == -1] = 1    # Anomaly

  #printing metrics 
  averagee = 'weighted'                                                                 
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average=averagee)  
  recall = recall_score(y_test, y_pred, average=averagee)
  conf_matrix = confusion_matrix(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')  

  print("\n")
  print("Isolation forest with PCA ")
  print("best hyper parameters :",best_params)
  print("Accuracy:", accuracy)
  print("precision",precision)
  print("Recall:", recall)
  print("Confusion matrix: \n") 
  print(conf_matrix)
  print("\n")
  plt.figure(figsize=(8, 6))
  plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
  plt.title('Confusion Matrix')
  plt.colorbar()
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()
  print("\n")
  #ROC curve 
  y_pred_prob = grid_search.best_estimator_.decision_function(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
  roc_auc = roc_auc_score(y_test, y_pred_prob)
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.show()


def isolation_forest(loc):
  warnings.filterwarnings("ignore")                 #to ignore warnings
  trainn = pd.read_csv(loc)    #loading data set 
  columns = ['duration','protocol','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
  ,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
  ,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
  ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
  ,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
  ,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','classs','level']
  trainn.columns = columns
  trainn = trainn.drop(columns='level')
  trainn["classs"] = trainn.classs.apply(lambda x: 0 if x == "normal" else 1)
  train=trainn[["src_bytes","dst_bytes","protocol","classs"]]   #feature selection
  X = train.drop(columns="classs")
  y = train["classs"]
  param_grid = {
      'n_estimators': [10,100],
      'contamination': [0.1, 0.15, 0.2],
      'max_samples': [.2,.25,100]
  }

  cols_to_ohe = train.select_dtypes("object").columns               #One Hot Encoding of data
  ohe = OneHotEncoder(sparse_output=False)
  num_cols = ohe.fit_transform(X[cols_to_ohe])
  num_cols_names = ohe.get_feature_names_out(cols_to_ohe)
  ohe_df = pd.DataFrame(num_cols, columns=num_cols_names)
  X_ohe = pd.concat([X.drop(columns=cols_to_ohe), ohe_df], axis=1)
  onlygrid_output=onlygrid(X_ohe,y)                                    #runs isolation forest 
  pca_output=pca(X_ohe,y)                                              #runs isolation forest with pca 

isolation_forest("/content/KDDTrain+.txt")
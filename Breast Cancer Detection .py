#!/usr/bin/env python
# coding: utf-8
in 2019 , 

1,62,468 new cases and 87,090 deaths were reported for breast cancer in India.
Among Indian women, breast cancer is the commonest cancer in Indian women overal

Breast cancer is the malignant tumor (a tumor with the potential to invade other tissues or spread to other parts of the body) that starts in the cells of the breast. It occurs both in men and women. However male breast cancer is rare.


It is very deficult for doctor to detect breast cancer at initial stages of it thats the reacone it sprad in body and cause the death.

To handel this problem we use Machine learning algorithms it gives 90%-95% Accureccymalignant - Person Having cancer 
benign - Persone having No cancer

0 means malignant tumor
1 mean benign tumor
#  # Import Libraries

# In[1]:


import pandas as pd                     #For Data Manupulation and data Analysis
import numpy as np                      #For Mathematical Caluculations 
import matplotlib.pyplot as plt         #For Data Visualization
import seaborn as sns                   #For Data Visualization


# # Data Load 
# 

# In[2]:


from sklearn.datasets import load_breast_cancer 
cancer_dataset = load_breast_cancer()
type(cancer_dataset)


# # Data Manupulation 

# In[3]:


# Keys In dataset 
cancer_dataset.keys()


# In[4]:


cancer_dataset['data']


# In[5]:


cancer_dataset['target']


# In[6]:


print(cancer_dataset['DESCR'])


# In[7]:


cancer_dataset['target_names']


# In[8]:


print(cancer_dataset['feature_names'])


# # Dataframe  

# In[9]:


# Data Frame Load
cancer_df = pd.read_csv('breast_cancer_dataframe.csv')


# In[10]:


# Dataframe Head 
cancer_df.head(5)


# In[11]:


cancer_df.tail(5)


# In[12]:


cancer_df.info()


# In[13]:


# Numerical distribution of data
cancer_df.describe() 


# In[14]:


# Finding Their is any null data 
cancer_df.isnull().sum()


# # Data Visualization

# In[15]:


# Pairplot of cancer Dataframe 
sns.pairplot(cancer_df, hue = 'target') 


# In[16]:


sns.pairplot(cancer_df, hue = 'target', 
             vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'] )


# In[17]:


# From Data visualsation we find that data is classification type of data 


# In[18]:


sns.countplot(cancer_df['target'])


# In[19]:


# Counter Plot for feature mean radius 
plt.figure(figsize=(20,8))
sns.countplot(cancer_df['mean radius'])


# In[20]:


# Counter Plot for feature mean Area
plt.figure(figsize=(20,8))
sns.countplot(cancer_df['mean area'])


# # Heatmap  

# In[21]:


plt.figure(figsize=(16,9))
sns.heatmap(cancer_df) 


# # Heatmap of coorelation Matrix 

# In[22]:


cancer_df.corr()


# In[23]:


# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.corr(), annot = True, cmap ='coolwarm', linewidths=2)


# ## Correlation Barplot

# In[24]:


# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)


# In[25]:


# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)


# In[26]:


cancer_df2.corrwith(cancer_df.target)


# In[27]:


# # visualize correlation barplot
# plt.figure(figsize = (16,5))
# ax = sns.barplot(cancer_df2.corrwith(cancer_df.target).index, cancer_df2.corrwith(cancer_df.target))
# ax.tick_params(labelrotation = 90)

plt.figure(figsize = (16,5))
ax = sns.barplot(cancer_df2.corrwith(cancer_df.target).index, cancer_df2.corrwith(cancer_df.target))
ax.tick_params(labelrotation = 90)


# ## Split Data into Train and Test 

# In[28]:


# Input Variable 
X = cancer_df.drop(['target'], axis = 1) 


# In[29]:


X.head()


# In[30]:


# Output Variable 
y = cancer_df['target'] 
y.head(6)


# In[31]:


from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=5)


# In[32]:


X_train.shape


# In[33]:


X_test.shape


# # Feature scaling  

# In[34]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[35]:


X_train_sc


# ## Machine Learning Model Building

# In[36]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# # Support Vector Classification 

# In[37]:


# Without Standerd Scaler 
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)


# In[38]:


# With Standard Scalar 
from sklearn.svm import SVC 
svc_classifier_sc = SVC()
svc_classifier_sc.fit(X_train_sc,y_train)

y_pred_svc_sc = svc_classifier_sc.predict(X_test_sc)
accuracy_score(y_test,y_pred_svc_sc)


# # Logistic Regression
# 

# In[39]:


# Without Standard Scalar 
from sklearn.linear_model import LogisticRegression 
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train,y_train)

lr_classifier_pred = lr_classifier.predict(X_test)
accuracy_score(lr_classifier_pred,y_test)


# In[40]:


# With Standard Scalar 
from sklearn.linear_model import LogisticRegression 
lr_classifier_sc = LogisticRegression()
lr_classifier_sc.fit(X_train_sc,y_train)

y_pred_svc_sc = lr_classifier_sc.predict(X_test_sc)
accuracy_score(y_test,y_pred_svc_sc)


# # K – Nearest Neighbor Classifier

# In[41]:


# Without Standard Scalar 

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)


# In[42]:


# Train with Standard scaled Data

knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_knn_sc)


#  # Naive Bayes Classifier

# In[43]:


# Without Standard Scalar

from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)


# In[44]:


# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_nb_sc)


# # Decision Tree Classifier
# 

# In[45]:


# Without Standard Scalar 

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# In[46]:


# With Standard Sclar 
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# # Random Forest Classifier

# In[47]:


# Without Standard Scalar 

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)


# In[48]:


# With Standard Scalar 
# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_rf_sc)


# # AdaBoost Classifier

# In[49]:


# Without Standard Scalar 

from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(X_train, y_train)
y_pred_adb = adb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_adb)


# In[50]:


# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(X_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_adb_sc)


# # XGBoost Classifier

# In[54]:


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


# In[55]:


# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)


# ## XGBoost Parameter Tuning Randomized Search 

# In[56]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}


# In[57]:


# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
random_search.fit(X_train, y_train)


# In[58]:


random_search.best_params_


# In[59]:


random_search.best_estimator_


# In[60]:


# training XGBoost classifier with best parameters
xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
       learning_rate=0.1, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)

xgb_classifier_pt.fit(X_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)


# In[61]:


accuracy_score(y_test, y_pred_xgb_pt)


# # Grid Search

# In[62]:


from sklearn.model_selection import GridSearchCV 
grid_search = GridSearchCV(xgb_classifier, param_grid=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
grid_search.fit(X_train, y_train)


# In[63]:


grid_search.best_estimator_


# In[64]:


xgb_classifier_pt_gs = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
       learning_rate=0.3, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
xgb_classifier_pt_gs.fit(X_train, y_train)
y_pred_xgb_pt_gs = xgb_classifier_pt_gs.predict(X_test)
accuracy_score(y_test, y_pred_xgb_pt_gs)


# # Confusion Matrix

# In[65]:


cm = confusion_matrix(y_test, y_pred_xgb_pt)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()


# ## Classification Report Of model

# In[66]:


print(classification_report(y_test, y_pred_xgb_pt))


# # Cross-validation of the ML model

# In[67]:


# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = X_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())


# # Save XGBoost Classifier model using Pickel

# In[68]:


## Pickle
import pickle

# save model
pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))

# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred),'\n')

# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents span<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Goal" data-toc-modified-id="Goal-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Goal</a></span></li><li><span><a href="#importing-the-libraries" data-toc-modified-id="importing-the-libraries-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>importing the libraries</a></span></li><li><span><a href="#Plot-settings" data-toc-modified-id="Plot-settings-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Plot settings</a></span></li><li><span><a href="#Reading-the-data" data-toc-modified-id="Reading-the-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Reading the data</a></span></li><li><span><a href="#Exploratory-data-analysis-(EDA)" data-toc-modified-id="Exploratory-data-analysis-(EDA)-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploratory data analysis (EDA)</a></span><ul class="toc-item"><li><span><a href="#Dashboard-with-sweetviz" data-toc-modified-id="Dashboard-with-sweetviz-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Dashboard with sweetviz</a></span></li><li><span><a href="#Pairplot-of-the-all-the-features" data-toc-modified-id="Pairplot-of-the-all-the-features-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Pairplot of the all the features</a></span></li><li><span><a href="#Pearson-correlation-and-feature-selection" data-toc-modified-id="Pearson-correlation-and-feature-selection-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Pearson correlation and feature selection</a></span></li><li><span><a href="#Class-distribution-in-Diabetes-dataset" data-toc-modified-id="Class-distribution-in-Diabetes-dataset-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Class distribution in Diabetes dataset</a></span></li></ul></li><li><span><a href="#Preparing-data-for-machine-learning" data-toc-modified-id="Preparing-data-for-machine-learning-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Preparing data for machine learning</a></span></li><li><span><a href="#Machine-learning-Models" data-toc-modified-id="Machine-learning-Models-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Machine learning Models</a></span><ul class="toc-item"><li><span><a href="#k-Nearest-Neighbors-(KNN)" data-toc-modified-id="k-Nearest-Neighbors-(KNN)-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>k-Nearest Neighbors (KNN)</a></span><ul class="toc-item"><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.1.2"><span class="toc-item-num">7.1.2&nbsp;&nbsp;</span>K-fold cross validation</a></span></li></ul></li><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Logistic Regression</a></span><ul class="toc-item"><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.2.1"><span class="toc-item-num">7.2.1&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.2.2"><span class="toc-item-num">7.2.2&nbsp;&nbsp;</span>K-fold cross validation</a></span></li></ul></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Decision Tree</a></span><ul class="toc-item"><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.3.1"><span class="toc-item-num">7.3.1&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.3.2"><span class="toc-item-num">7.3.2&nbsp;&nbsp;</span>K-fold cross validation</a></span></li><li><span><a href="#Feature-importance-in-Decision-trees" data-toc-modified-id="Feature-importance-in-Decision-trees-7.3.3"><span class="toc-item-num">7.3.3&nbsp;&nbsp;</span>Feature importance in Decision trees</a></span></li></ul></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Random Forest</a></span><ul class="toc-item"><li><span><a href="#Feature-importance-in-Random-Forest" data-toc-modified-id="Feature-importance-in-Random-Forest-7.4.1"><span class="toc-item-num">7.4.1&nbsp;&nbsp;</span>Feature importance in Random Forest</a></span></li><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.4.2"><span class="toc-item-num">7.4.2&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.4.3"><span class="toc-item-num">7.4.3&nbsp;&nbsp;</span>K-fold cross-validation</a></span></li></ul></li><li><span><a href="#Gradient-Boosting" data-toc-modified-id="Gradient-Boosting-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>Gradient Boosting</a></span><ul class="toc-item"><li><span><a href="#Feature-importance-in-Gradient-Boosting" data-toc-modified-id="Feature-importance-in-Gradient-Boosting-7.5.1"><span class="toc-item-num">7.5.1&nbsp;&nbsp;</span>Feature importance in Gradient Boosting</a></span></li><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.5.2"><span class="toc-item-num">7.5.2&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.5.3"><span class="toc-item-num">7.5.3&nbsp;&nbsp;</span>K-fold cross-validation</a></span></li></ul></li><li><span><a href="#Support-Vector-Machine-(SVM)" data-toc-modified-id="Support-Vector-Machine-(SVM)-7.6"><span class="toc-item-num">7.6&nbsp;&nbsp;</span>Support Vector Machine (SVM)</a></span><ul class="toc-item"><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.6.1"><span class="toc-item-num">7.6.1&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.6.2"><span class="toc-item-num">7.6.2&nbsp;&nbsp;</span>K-fold cross-validation</a></span></li></ul></li><li><span><a href="#Neural-Networks" data-toc-modified-id="Neural-Networks-7.7"><span class="toc-item-num">7.7&nbsp;&nbsp;</span>Neural Networks</a></span><ul class="toc-item"><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.7.1"><span class="toc-item-num">7.7.1&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.7.2"><span class="toc-item-num">7.7.2&nbsp;&nbsp;</span>K-fold cross-validation</a></span></li></ul></li><li><span><a href="#XGBoost" data-toc-modified-id="XGBoost-7.8"><span class="toc-item-num">7.8&nbsp;&nbsp;</span>XGBoost</a></span><ul class="toc-item"><li><span><a href="#Confusion-matrix" data-toc-modified-id="Confusion-matrix-7.8.1"><span class="toc-item-num">7.8.1&nbsp;&nbsp;</span>Confusion matrix</a></span></li><li><span><a href="#K-fold-cross-validation" data-toc-modified-id="K-fold-cross-validation-7.8.2"><span class="toc-item-num">7.8.2&nbsp;&nbsp;</span>K-fold cross-validation</a></span></li><li><span><a href="#Feature-importance-in-XGBoost" data-toc-modified-id="Feature-importance-in-XGBoost-7.8.3"><span class="toc-item-num">7.8.3&nbsp;&nbsp;</span>Feature importance in XGBoost</a></span></li></ul></li></ul></li><li><span><a href="#Model-comparision" data-toc-modified-id="Model-comparision-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Model comparision</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Conclusion</a></span></li><li><span><a href="#In-Future" data-toc-modified-id="In-Future-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>In Future</a></span></li></ul></div>

# # Goal  
# 
# **Understanding and predicting of diabetes based of medical conditions**.

# # importing the libraries

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import sweetviz as sv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


get_ipython().run_line_magic('matplotlib', 'inline')


# # Plot settings

# In[6]:


plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 23
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['figure.figsize'] = 7,4
sns.set_style('ticks')


# # Reading the data

# In[8]:


csv_file='diabetes.csv'
df = pd.read_csv(csv_file)


# In[9]:


df.iloc[:,:-1].describe().style.background_gradient(axis=0,cmap='Accent')


# In[10]:


df.head(10).style.bar(axis=0)


# # Exploratory data analysis (EDA)

# In[12]:


print("dimension of diabetes data: {}".format(df.shape))


# In[13]:


df.info()


#  * Dataset is small and well labeled. There are no null values present.
#  * very suitable to supervised machine learning formulation.
#  * This is a binary classification problem, where we have 2 classes in the target **(y)** (i.e.`df['Outcome']`) and the medical conditions can be used as the feature (**X**).

# ## Dashboard with sweetviz

# In[16]:


sv.analyze([df,'Diabetes data' ], target_feat='Outcome').show_html()


# ## Pairplot of the all the features

# In[18]:


sns.pairplot(df.iloc[:,:-1], diag_kind='hist')


# ## Pearson correlation and feature selection

# In[20]:


plt.figure(figsize=(10,7))
corr_mat = df.iloc[:,:-1].corr()
sns.heatmap(corr_mat, fmt='0.2f', annot=True, lw=2, cbar_kws={'label':'Pearson Correlation (r)'})
plt.xticks(size=15,rotation=90)
plt.yticks(size=15)
plt.tight_layout()
plt.savefig('Correlation.png',dpi=300);


# * All the features hold very little correlation amongst each other. Only age and pregnancies shows a significant  strong correlation. Therefore, keeping the all the features. 

# ## Class distribution in Diabetes dataset

# In[23]:


plt.figure(figsize=(5,5))
sns.countplot(df['Outcome'],label='Count')
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylabel('Outcome',size=15)
plt.xlabel('Count',size=15)
sns.despine(top=True)
plt.title('Class distribution',size=26, weight='bold')
plt.tight_layout()
plt.savefig('Class-distribution.png');


# # Preparing data for machine learning

# In[25]:


X = df.iloc[:,:-1].values   ##features selection
y = df.iloc[:,-1].values    ## target selection


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=27)


# * It is importannt to use stratify inside **train_test_split**. It keeps the same distribution of target same  within train and testing datasets. if variable `y` is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, `stratify=y` will make sure that random split has 40% of 0's and 60% of 1's

# # Machine learning Models

# ## k-Nearest Neighbors (KNN)

# In[30]:


training_accuracy = [] 
test_accuracy = []
training_f1 = []
test_f1 = []

neighbors_settings = range(2,20)

for n_neighbors in neighbors_settings:
    print(f'working on neighbors {n_neighbors}')
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_train_pred=knn.predict(X_train)
    y_pred=knn.predict(X_test)
    
    training_accuracy.append(accuracy_score(y_train,y_train_pred))
    test_accuracy.append(accuracy_score(y_test, y_pred))
    
    training_f1.append(f1_score(y_train,y_train_pred))
    test_f1.append(f1_score(y_test, y_pred))
    


# In[31]:


fig = plt.figure(figsize=(14,10))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

fig.add_subplot(2,2,1)
plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy',size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('n_neighbors',size=20)
plt.title('Accuracy Score',size=20, weight='bold')
plt.legend([],frameon=False)

fig.add_subplot(2,2,2)
plt.plot(neighbors_settings, training_f1)
plt.plot(neighbors_settings, test_f1)
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylabel('F1 Score',size=20)
plt.xlabel('n_neighbors',size=20)
plt.title('F1-Score',size=20,weight='bold')
plt.legend(['Training','Testing'],frameon=False, bbox_to_anchor=(0.4,-0.2), ncol=2);


# `Accuracy score`, `F1-score` suggest that `n_neighbors=19` is the optimum choice **(by looking a testing accuracies)**. 

# In[33]:


knn = KNeighborsClassifier(n_neighbors=19).fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('F1-score of K-NN classifier on test set: {:.2f}'.format(f1_score(y_test, y_pred)))


# ### Confusion matrix

# In[35]:


def normalized_confusion_matrix(y_test, conf_mat, model):
    _ , counts = np.unique(y_test,return_counts=True)
    conf_mat = conf_mat/counts
    plt.figure(figsize=(6,5))
    ax=sns.heatmap(conf_mat,fmt='.2f',annot=True,annot_kws={'size':20},lw=2, cbar=True, cbar_kws={'label':'% Class accuracy'})
    plt.title(f'Confusion Matrix ({model})',size=22)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.figure.axes[-1].yaxis.label.set_size(20) ##colorbar label
    cax = plt.gcf().axes[-1]  ##colorbar ticks
    cax.tick_params(labelsize=20) ## colorbar ticks
    plt.savefig(f'confusion-matrix-{model}.png',dpi=300)


# In[36]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat, 'kNN')


# ### K-fold cross validation

# In[38]:


k_fold_knn_accuracy = cross_val_score(knn, X, y, cv=10) ##10-fold cross validation
k_fold_knn_f1 = cross_val_score(knn, X, y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[39]:


print(f'Average accuracy after 10 fold cross validation :{k_fold_knn_accuracy.mean().round(2)} +/- {k_fold_knn_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation :{k_fold_knn_f1.mean().round(2)} +/- {k_fold_knn_f1.std().round(2)}')


# ## Logistic Regression

# In[65]:


plt.figure(figsize=(8,6))
Clist=[1,0.01,100]

for C in Clist : 

    logreg = LogisticRegression(C=C,solver='newton-cg').fit(X_train, y_train) #keeping C=1 a
    y_train_pred = logreg.predict(X_train)
    y_pred = logreg.predict(X_test)

    print('C : {} Training set accuracy: {:.3f}'.format(C,accuracy_score(y_train, y_train_pred)))
    print('C : {} Test set accuracy: {:.3f}'.format(C,accuracy_score(y_test, y_pred)))

    print('C : {} Training set F1-score: {:.3f}'.format(C,f1_score(y_train, y_train_pred)))
    print('C : {} Test set F1-score: {:.3f}'.format(C, f1_score(y_test, y_pred)))
    print('\n')
    
   # Assuming df has 9 columns including the target
    diabetes_features = [x for i, x in enumerate(df.columns) if i != 8]  # Exclude the last column (target)

# Use the correct number of ticks for the plot
plt.plot(logreg.coef_.T, marker='o', label=f"C={C}")
plt.xticks(range(len(diabetes_features)), diabetes_features, rotation=90)  # Adjust to match the length
plt.hlines(0, 0, len(diabetes_features) - 1)
plt.ylim(-5, 5)


# The default value of C=1 provides with 78% accuracy on training and 77% accuracy on test set.

# Stronger regularization (`C=0.001`) pushes coefficients more and more toward zero. Inspecting the plot more closely suggests that feature `DiabetesPedigreeFunction`, for `C=100`, `C=1` and `C=0.001`, the coefficient is positive. This indicates that high `DiabetesPedigreeFunctio` feature is related to a sample being`diabetes`, regardless which model we look at.

# ### Confusion matrix

# In[67]:


logreg = LogisticRegression(C=1,solver='newton-cg').fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[69]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'Logistic Regression')


# ### K-fold cross validation

# In[71]:


k_fold_logreg_accuracy = cross_val_score(logreg, X, y, cv=10) ##10-fold cross validation
k_fold_logreg_f1 = cross_val_score(logreg, X, y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[73]:


print(f'Average accuracy after 10 fold cross validation :{k_fold_logreg_accuracy.mean().round(2)} +/- {k_fold_logreg_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation :{k_fold_logreg_f1.mean().round(2)} +/- {k_fold_logreg_f1.std().round(2)}')


# ## Decision Tree

# In[75]:


max_depth=range(1,20)
training_accuracy = [] 
test_accuracy = []
training_f1 = []
test_f1 = []

for depth in max_depth : 
    tree = DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_leaf=1).fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_pred = tree.predict(X_test)
    
    training_accuracy.append(accuracy_score(y_train,y_train_pred))
    test_accuracy.append(accuracy_score(y_test, y_pred))
    
    training_f1.append(f1_score(y_train,y_train_pred))
    test_f1.append(f1_score(y_test, y_pred))


# In[77]:


fig = plt.figure(figsize=(14,10))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

fig.add_subplot(2,2,1)
plt.plot(max_depth, training_accuracy, label='training accuracy')
plt.plot(max_depth, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy',size=20)
plt.xlabel('Max depth',size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('Accuracy Score',size=20, weight='bold')
plt.legend([],frameon=False)

fig.add_subplot(2,2,2)
plt.plot(max_depth, training_f1)
plt.plot(max_depth, test_f1)
plt.ylabel('F1 Score',size=20)
plt.xlabel('Max depth',size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('F1-Score',size=20,weight='bold')
plt.legend(['Training','Testing'],frameon=False, bbox_to_anchor=(0.4,-0.2), ncol=2);


# The training accuracy on the after training set is 100%, while the test set accuracy is much worse. This means tree is  overfitting and not generalizing well to new data. We set `max_depth=4`, limiting the depth of the tree decreases overfitting. This leads to a lower accuracy on the training set, but an improvement on the test set.

# In[79]:


tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1, random_state=0).fit(X_train, y_train)
y_pred=tree.predict(X_test)

print("Accuracy on test: {:.3f}".format(accuracy_score(y_pred, y_test)))
print("F1-score on test set: {:.3f}".format(f1_score(y_pred, y_test)))


# ### Confusion matrix

# In[81]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'Decision Tree')


# ### K-fold cross validation

# In[83]:


k_fold_tree_accuracy = cross_val_score(tree, X, y, cv=10) ##10-fold cross validation
k_fold_tree_f1 = cross_val_score(tree, X, y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[85]:


print(f'Average accuracy after 10 fold cross validation :{k_fold_tree_accuracy.mean().round(2)} +/- {k_fold_tree_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation :{k_fold_tree_f1.mean().round(2)} +/- {k_fold_tree_f1.std().round(2)}')


# ### Feature importance in Decision trees
# Feature importance rates how important each feature is for the decision a tree makes. It is a number between 0 and 1
# 
#      0 means “not used at all” 
#      1 means “perfectly predicts the target”

# In[87]:


def plot_feature_importances(model, figure):
    n_features = 8
    plt.figure(figsize=(10,6))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features,size=15)
    plt.xticks(size=15)
    plt.xlabel('Feature importance',size=20,)
    #plt.ylabel('Feature',size=20)
    plt.ylim(-1, n_features)
    sns.despine(top=True)
    plt.title(f'{figure}',size=20)
    plt.tight_layout()
    plt.savefig(f'feature-image{figure}.png',dpi=300)

plot_feature_importances(tree,'Decision Tree')


# Feature `Glucose` is the most important feature.

# ## Random Forest

# In[89]:


rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
y_pred=rf.predict(X_test)

print("Accuracy on test: {:.3f}".format(accuracy_score(y_pred, y_test)))
print("F1-score on test set: {:.3f}".format(f1_score(y_pred, y_test)))


#  The default parameters of the random forest work well. This conclusion is derived after trying various values of  `max_depth` and `n_estimators`. 

# ### Feature importance in Random Forest

# In[91]:


plot_feature_importances(rf, 'Random Forest')


# Similarly to the single decision tree, the random forest also gives a lot of importance to the `Glucose` feature and it also chooses `BMI` to be the 2nd most informative features. 

# ### Confusion matrix

# In[93]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'Random Forest')


# ### K-fold cross-validation

# In[95]:


k_fold_rf_accuracy = cross_val_score(rf, X, y, cv=10) ##10-fold cross validation
k_fold_rf_f1 = cross_val_score(rf, X, y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[97]:


print(f'Average accuracy after 10 fold cross validation :{k_fold_tree_accuracy.mean().round(2)} +/- {k_fold_tree_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation :{k_fold_tree_f1.mean().round(2)} +/- {k_fold_tree_f1.std().round(2)}')


# ## Gradient Boosting

# In[99]:


gb = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.01).fit(X_train, y_train)
y_pred=gb.predict(X_test)

print("Accuracy on train: {:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy on test: {:.3f}".format(accuracy_score(y_pred, y_test)))
print("F1-score on test set: {:.3f}".format(f1_score(y_pred, y_test)))


# Default values result in **overfitting**. Therefore, `max_depth = 1` and `learning_rate = 0.01` is chosen which reduces the gap between training and test accuracy. 

# ### Feature importance in Gradient Boosting

# In[101]:


plot_feature_importances(gb, 'Gradient Boosting')


# Gradient Boosting gives the most importance to `Glucose`.

# ### Confusion matrix

# In[103]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'Gradient Boosting')


# ### K-fold cross-validation

# In[105]:


k_fold_gb_accuracy = cross_val_score(gb, X, y, cv=10) ##10-fold cross validation
k_fold_gb_f1 = cross_val_score(gb, X, y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[107]:


print(f'Average accuracy after 10 fold cross validation :{k_fold_gb_accuracy.mean().round(2)} +/- {k_fold_gb_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation :{k_fold_gb_f1.mean().round(2)} +/- {k_fold_gb_f1.std().round(2)}')


# ## Support Vector Machine (SVM)

# In[109]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#SVM requiest feature scaling
svc = SVC(C=1000, gamma='auto').fit(X_train_scaled, y_train)
y_pred=svc.predict(X_test_scaled)

print("Accuracy on train: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test: {:.3f}".format(accuracy_score(y_pred, y_test)))
print("F1-score on test set: {:.3f}".format(f1_score(y_pred, y_test)))


# `C = 1000`, `gamma='auto'` gives the best performance and minimum **overfitting**. 

# ### Confusion matrix

# In[111]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'SVM')


# ### K-fold cross-validation

# In[113]:


k_fold_svm_accuracy = cross_val_score(svc, scaler.fit_transform(X), y, cv=10) ##10-fold cross validation
k_fold_svm_f1 = cross_val_score(svc, scaler.fit_transform(X), y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[115]:


print(f'Average accuracy after 10 fold cross validation : {k_fold_svm_accuracy.mean().round(2)} +/- {k_fold_svm_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation : {k_fold_svm_f1.mean().round(2)} +/- {k_fold_svm_f1.std().round(2)}')


# ## Neural Networks

# In[117]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

mlp = MLPClassifier(random_state=0, alpha=1, max_iter=2000).fit(X_train_scaled, y_train)
y_pred=mlp.predict(X_test_scaled)

print("Accuracy on train: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test: {:.3f}".format(accuracy_score(y_pred, y_test)))
print("F1-score on test set: {:.3f}".format(f1_score(y_pred, y_test)))


# ### Confusion matrix

# In[119]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'Neural Networks')


# ### K-fold cross-validation

# In[121]:


k_fold_mlp_accuracy = cross_val_score(mlp, scaler.fit_transform(X), y, cv=10) ##10-fold cross validation
k_fold_mlp_f1 = cross_val_score(mlp, scaler.fit_transform(X), y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[123]:


print(f'Average accuracy after 10 fold cross validation : {k_fold_mlp_accuracy.mean().round(2)} +/- {k_fold_mlp_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation : {k_fold_mlp_f1.mean().round(2)} +/- {k_fold_mlp_f1.std().round(2)}')


# ## XGBoost

# In[125]:


xgb = XGBClassifier(random_state=0, learning_rate =0.01,n_estimators=100, max_depth=5,gamma=1).fit(X_train,y_train)

y_pred=xgb.predict(X_test)

print("Accuracy on train: {:.3f}".format(xgb.score(X_train, y_train)))
print("Accuracy on test: {:.3f}".format(xgb.score(X_test, y_test)))
print("F1-score on test set: {:.3f}".format(f1_score(y_pred, y_test)))


# ### Confusion matrix

# In[127]:


conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'XGBoost')


# ### K-fold cross-validation

# In[129]:


k_fold_xgb_accuracy = cross_val_score(xgb, X, y, cv=10) ##10-fold cross validation
k_fold_xgb_f1 = cross_val_score(xgb, X, y, cv=10, scoring='f1_macro') ##10-fold cross validation


# In[131]:


print(f'Average accuracy after 10 fold cross validation : {k_fold_xgb_accuracy.mean().round(2)} +/- {k_fold_xgb_accuracy.std().round(2)}')
print(f'Average F1-score after 10 fold cross validation : {k_fold_xgb_f1.mean().round(2)} +/- {k_fold_xgb_f1.std().round(2)}')


# ### Feature importance in XGBoost

# In[133]:


plot_feature_importances(xgb, 'XGBoost')


# # Model comparision

# In[135]:


all_accuracies = [k_fold_knn_accuracy.mean().round(2),
                   k_fold_logreg_accuracy.mean().round(2),
                   k_fold_tree_accuracy.mean().round(2),
                   k_fold_rf_accuracy.mean().round(2),
                   k_fold_gb_accuracy.mean().round(2),
                   k_fold_svm_accuracy.mean().round(2),
                   k_fold_mlp_accuracy.mean().round(2),
                   k_fold_xgb_accuracy.mean().round(2)]

all_accuracies_errors = [k_fold_knn_accuracy.std().round(2),
                   k_fold_logreg_accuracy.std().round(2),
                   k_fold_tree_accuracy.std().round(2),
                   k_fold_rf_accuracy.std().round(2),
                   k_fold_gb_accuracy.std().round(2),
                   k_fold_svm_accuracy.std().round(2),
                   k_fold_mlp_accuracy.std().round(2),
                   k_fold_xgb_accuracy.std().round(2)]


# In[137]:


all_f1 = [k_fold_knn_accuracy.mean().round(2),
                   k_fold_logreg_f1.mean().round(2),
                   k_fold_tree_f1.mean().round(2),
                   k_fold_rf_f1.mean().round(2),
                   k_fold_gb_f1.mean().round(2),
                   k_fold_svm_f1.mean().round(2),
                   k_fold_mlp_f1.mean().round(2),
                   k_fold_xgb_f1.mean().round(2)]

all_f1_errors = [k_fold_knn_f1.std().round(2),
                   k_fold_logreg_f1.std().round(2),
                   k_fold_tree_f1.std().round(2),
                   k_fold_rf_f1.std().round(2),
                   k_fold_gb_f1.std().round(2),
                   k_fold_svm_f1.std().round(2),
                   k_fold_mlp_f1.std().round(2),
                   k_fold_xgb_f1.std().round(2)]


# In[139]:


models=['kNN','Logistic Regression','Decision Tree','Random Forest','Gradient Boosting','SVM','Neural Networks', 'XGBoost']


# In[141]:


model_data = pd.DataFrame([all_accuracies,all_accuracies_errors,all_f1,all_f1_errors],columns=models, index = ['Accuracy','STD_acc','F1-macro','STD_f1']).T


# In[143]:


model_data.style.background_gradient(cmap='coolwarm',axis=0)


# In[145]:


model_data[['Accuracy','F1-macro']].plot.barh(figsize=(10,7))
plt.legend(frameon=False,bbox_to_anchor=(0.7,0.8), prop={'size':20})
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlim([0.65,0.8]);
plt.title('Model performances',size=20)
sns.despine(top=True)
plt.tight_layout()
plt.savefig('model-comparision.png',dpi=300)


# In[147]:


plt.figure(figsize=(10,7))
diabetes_accuracy = [0.52, 0.60, 0.66, 0.57, 0.33, 0.34, 0.61, 0.64]
plt.barh(models,diabetes_accuracy, color='red',alpha=0.5)
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('Accuracy in Diabetes Detection',size=20)
sns.despine(top=True)
plt.tight_layout()
plt.savefig('diabetes-detection.png',dpi=300)


# # Conclusion 

# * **Logistic regression** and **Neural Netowrks** seems to provide the best performance based on **10-fold cross validation** of the dataset. **Logistic regression** achieves a higher *F1-score* as well, which is better metric for model evalution.
# * From the confusion matricies, decision tree has the highest success in detecting the diabetes.
# * **Feature selection** suggests the `Glucose` is the most crucial factor for the successful prediction of diabetes. 

# #  In Future

# * More hyperparameter tuning.
# * More classifiers such as `catboost` to improve the performance.
# * Generating a Tableau Dashboard

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# read data:

os.chdir(r'C:\Users\dell\Downloads')
data=pd.read_csv("Social_Network_Ads.csv")
data


# In[3]:


# basic info:

data.info()


# # Data preprocessing:

# In[4]:


# null value checking:
data.isnull().sum()


# # EDA:
# 

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


plt.style.available


# In[7]:


plt.style.use("seaborn-pastel")


# In[8]:


plt.figure(figsize=(10,5))
sns.countplot(data=data,x="Gender",hue="Purchased")


# here we can see that the dataset is balanced with male and female customers and the rate of buying is also same among both the sex..

# In[9]:


plt.figure(figsize=(10,5))
sns.histplot(data=data,x="Age",hue="Purchased",bins=20)


# using this histogram we see that the people abouve age 45 are more likely prone to buy products based upon the add recommendation..

# # feature selection:
# 

# In[10]:


x= data.iloc[:,2:4].values
y=data.iloc[:,-1].values
y


# In[11]:


# feature scaling:

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# # Model prep:

# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.33)


# ###  logistic regression:

# In[13]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

# prediction:

lr_result=lr.predict(x_test)
lr_result


# ###  KNN algorithm:

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors=5)
kn.fit(x_train,y_train)

#prediction:

kn_result=kn.predict(x_test)


# ### navie bayes:

# In[15]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

# prediction:

nb_result=nb.predict(x_test)


# ### desicion tree:
# 

# In[16]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

# prediction:

dt_result=dt.predict(x_test)


# ### random forest:

# In[17]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=5,criterion="entropy")
rf.fit(x_train,y_train)

#prediction:

rf_result=rf.predict(x_test)


# ### svm algorithm:

# In[18]:


# 1.linear kernal:

from sklearn.svm import SVC
svm_linear=SVC(kernel='linear')
svm_linear.fit(x_train,y_train)

# predicton:

svm_linear_result=svm_linear.predict(x_test)


# In[20]:


# 2.sigmoid kernal:

from sklearn.svm import SVC
svm_sigmoid=SVC(kernel='sigmoid')
svm_sigmoid.fit(x_train,y_train)

# predicton:

svm_sigmoid_result=svm_sigmoid.predict(x_test)


# In[21]:


# 3.rbf kernal:

from sklearn.svm import SVC
svm_rbf=SVC(kernel='rbf')
svm_rbf.fit(x_train,y_train)

# predicton:

svm_rbf_result=svm_rbf.predict(x_test)


# 
# # Model evaluation:

# ### model evaluation :
# 

# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

results = [kn_result,nb_result,dt_result,rf_result,svm_linear_result,svm_sigmoid_result,svm_rbf_result]
algorithms=["KNN","Navie bayes","Desision Tree","Random Forest","SVM_Linear","SVM_Sigmoid","SVM_Rbf"]
test =y_test

# function to calculate the confusion matrix for all the algorithms:

def matrix(results,test,algorithms):
    for result,algorithm in zip(results,algorithms):
        print("The confusion matrix for",algorithm,"algorithm is,\n", confusion_matrix(result,test))

# function to calculate the accuracy score for all the algorithms:
        
def score(results,test,algorithms):
    for result,algorithm in zip(results,algorithms):
        print("The Accuracy score for",algorithm,"algorithm is,\n",round(accuracy_score(result,test),2),"%")
        
# function to calculate the accuracy score for all the algorithms:

def report(results,test,algorithms):
    for result,algorithm in zip(results,algorithms):
        print("The classification report for",algorithm,"algorithm is,\n", classification_report(result,test))
      


# 
# ### Confusion matrix:

# In[23]:


matrix(results,test,algorithms)


# ### Accuracy score:

# In[24]:


score(results,test,algorithms)


# ### Classification Reports:

# In[25]:


report(results,test,algorithms)


# here we got the model evaluation matrix for all the implemented algorithms and we can see various scores depending upon the model and we can select the model with best accuracy score..

# there are various hyperparameters which controll the model performance. inorder to get the best hyperparameter for models we can do a hyper parameter tuning using various methods,one of them is GRIDSEARCH CV

# # Hyperparameter Tuning:

# In[26]:


# Grid search 

# KNN:n_neighbors:

from sklearn.model_selection import GridSearchCV
parameters={'n_neighbors':[5,7,8,10,12]}
grid=GridSearchCV(estimator=kn,param_grid=parameters,scoring="accuracy")
grid.fit(x_train,y_train)

print(grid.best_params_)
print(grid.best_score_)


# here we see that using gridsearch cv we found out that the best value for n_neighbors is 7,which helps to get the best accuracy..

# In[27]:


# SVM:Kernals:

from sklearn.model_selection import GridSearchCV
parameters={'kernel':['linear','sigmoid','rbf']}
grid=GridSearchCV(estimator=svm_sigmoid,param_grid=parameters,scoring="accuracy")
grid.fit(x_train,y_train)

print(grid.best_params_)
print(grid.best_score_)


# ## now we try to implement this same dataset using DEEPLEARNING technique called perceptron:

# # Deep learning:
# 
# 

# In[28]:


x_train


# In[29]:


x_train.shape


# In[30]:


y_train


# In[31]:


y_train.shape


# # Model building:
# 

# In[32]:


import tensorflow as tf
from tensorflow import keras


# In[33]:


early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=30,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False
)


# Early stopping is used to stop the model iterartion when the best evaluation parameters are achieved.
# This technique helps in reduction of time consumption..

# In[48]:


model = keras.Sequential([
    keras.layers.Dense(20,input_shape=(2,),activation="relu"),
    keras.layers.Dropout(0.02),
    keras.layers.Dense(5,activation="relu"),
    keras.layers.Dropout(0.02),
    keras.layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])

model.fit(x_train,y_train,epochs=100,validation_split=0.3,validation_batch_size=20,callbacks=early_stopping)


# In[49]:


model.history.history.keys()


# ## summay of loss and accuracy:

# In[50]:



plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[51]:


plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train",'test'],loc="upper right")
plt.show()


# # Model evaluation:

# In[52]:


model.evaluate(x_test,y_test)


# # Model prediction:

# In[53]:


y_pred=model.predict(x_test)
y_pred[:5]


# In[54]:


y_test[:5]


# In[55]:


y_predicted=[]

for i in y_pred:
    if i >=0.5:
        y_predicted.append(1)
    else:
        y_predicted.append(0)


# In[56]:


y_predicted[:5]


# # Report Generation:

# In[57]:


from sklearn.metrics import classification_report, accuracy_score ,confusion_matrix


# In[58]:


print(classification_report(y_test,y_predicted))


# In[59]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test,y_predicted),annot=True,fmt="d")
plt.show()


# In[60]:


accuracy=accuracy_score(y_test,y_predicted)
print("The accuracy score using multilayer perceptron is,",round(accuracy,2),"%")


# here we see that this simple classification problem gives about 91% accuracy whilr using the Artificial Neural Network Algorithm..

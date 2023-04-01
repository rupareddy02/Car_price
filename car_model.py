#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[11]:


#CAR PRICE PREDICTION WITH MACHINE LEARNING


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# Load the dataset
data = pd.read_csv('C:\\Users\\rupar\\OneDrive\\Documents\\car_model.csv')
print(data)


# In[5]:


# Remove any missing values
#clean and preprocess the data by removing any missing values and converting categorical variables into numerical ones:
data = data.dropna()
data


# In[6]:


# Convert categorical variables into numerical ones using one-hot encoding
data = pd.get_dummies(data, columns=['CarName', 'fueltype', 'price', 'horsepower'])
data


# In[7]:


print("Type of dataset: {}".format(type(data['car_ID'])))
print("Shape of dataset: {}".format(data['car_ID'].shape))
print("Target:\n{}".format(data['car_ID']))


# In[14]:


#train_test_split(samples,features,randomstate)
X_train, X_test, y_train, y_test = train_test_split(data['car_ID'], data['car_ID'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[23]:


y_new = np.array([[9,2.6, 1, 0.2]])
print("y_new.shape: {}".format(X_new.shape))


# In[9]:


X_new = np.array([[6, 4.2, 5, 0.3]])
print("X_new.shape: {}".format(X_new.shape))


# In[10]:



# plotting a line graph
print("Line graph: ")
plt.plot(data['car_ID'], data['fuelsystem'])
plt.show()
  
# plotting a scatter plot
print("Scatter Plot:  ")
plt.scatter(data['car_ID'], data['fuelsystem'])
plt.show()


# In[11]:


# plotting a histogram
plt.hist(data["fuelsystem"])
plt.show()


# In[ ]:





# In[28]:



# Reshape the data
X_train = X_new.reshape(-1, 1)
y_train = y_new.reshape(-1, 1)
X_test  = X_new.reshape(-1, 1)
y_test  = y_new.reshape(-1, 1)

# Fit the model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
score = model.score(X_test,y_test)
print('Accuracy of the Model:', score)


# In[ ]:



                                        ******** THANK YOU  ***********


# In[ ]:





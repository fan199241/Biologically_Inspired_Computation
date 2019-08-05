
# coding: utf-8

# # Coursework1

# In[10]:


'''
Created on 28 Aug 2018

@author: marta
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Loading the dataset
os.getcwd()
train_dataset = h5py.File('trainCats.h5', "r")
trainSetX = np.array(train_dataset["train_set_x"][:]) # your train set features -- print(trainSetX.shape) -- (209, 64, 64, 3)
trainSetY = np.array(train_dataset["train_set_y"][:]) # your train set labels -- print(trainSetY.shape) -- (1, 209)
trainSetY = trainSetY.reshape((1, trainSetY.shape[0]))

test_dataset = h5py.File('testCats.h5', "r")
testSetX = np.array(test_dataset["test_set_x"][:]) # your test set features -- print(testSetX.shape) -- (50, 64, 64, 3)
testSetY = np.array(test_dataset["test_set_y"][:]) # your test set labels -- print(testSetY.shape) -- (1, 50)
testSetY = testSetY.reshape((1, testSetY.shape[0]))

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

# Example of a picture
index = 20
plt.imshow(trainSetX[index])
plt.show()
print ("y = " + str(trainSetY[:, index]) + ", it's a '" + classes[np.squeeze(trainSetY[:, index])].decode("utf-8") +  "' picture.")


# In[2]:


# Flatten the pictures
trainSetXF= trainSetX.reshape(trainSetX.shape[0], -1).T # print(trainSetXF.shape) -- (12288, 209)
testSetXF = testSetX.reshape(testSetX.shape[0], -1).T # print(testSetXF.shape) -- (12288, 50)

# Standardise the dataset
trainSetXS = trainSetXF/255
testSetXS = testSetXF/255


# In[3]:


# for key in train_dataset.keys():
    # print(key)


# In[4]:


# train_dataset['train_set_x'].shape


# In[5]:


# Initialize the model's parameters
# m -- the number of training examples
m = trainSetXS.shape[1]
w = np.zeros((trainSetXS.shape[0],1))
b = 0
J = 0
lr = 0.05
n_i = 1000


# In[6]:


# Activation Function: Sigmoid
# z = W^T*X+b
# s -- sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# In[11]:


# Update the parameters
# lr -- learning rate
# n_i -- the number of the iterations
costValues = []
for i in range(n_i):
    
    # Forward propagation
    # w -- weights
    # b -- bias

    X = trainSetXS
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    
    # Cost function (J)
    Y = trainSetY
    J = -1/m * np.sum((Y * np.log(a), (1-Y) * np.log(1-a)))
    
    # Gradient Descent
    dw = 1/m * np.dot(X, (a-Y).T)
    db = 1/m * np.sum(a-Y)
    
    w = w - lr * dw
    b = b - lr * db
        
    costValues.append(J)


# In[ ]:


# print(costValues)


# In[12]:


# costValues array with the costs for each iteration
plt.plot(costValues)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()


# In[23]:


# Accuracy

y_predict_train = sigmoid(np.dot(w.T, trainSetXF) + b)
y_predict_test = sigmoid(np.dot(w.T, testSetXF) + b)

print('the train accuracy is ', np.mean(y_predict_train == trainSetY))
print('the test accuracy is ', np.mean(y_predict_test == testSetY))


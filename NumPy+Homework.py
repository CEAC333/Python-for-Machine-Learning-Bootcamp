
# coding: utf-8

# # Question 1

# In[1]:

import numpy as np


# # Question 2

# In[4]:

np.zeros(10)


# # Question 3

# In[5]:

np.ones(10)


# # Question 4

# In[9]:

np.ones(10) * 5


# In[10]:

np.zeros(10) + 5


# # Question 5

# In[16]:

np.arange(10,51)


# # Question 6

# In[17]:

np.arange(10,51,2)


# # Question 7

# In[20]:

np.arange(9).reshape(3,3)


# In[24]:

a = np.arange(9)
a.reshape(3,3)


# # Question 8

# In[25]:

np.eye(3)


# # Question 9

# In[26]:

np.random.rand(1)


# # Question 10

# In[28]:

np.random.randn(25)


# # Question 11

# In[31]:

np.arange(1,101).reshape(10,10)/100


# In[32]:

np.linspace(0.01,1,100).reshape(10,10)


# # Question 12

# In[33]:

np.linspace(0,1,20)


# # Numpy Index and Selection

# In[39]:

mat = np.arange(1,26).reshape(5,5)
mat


# # Question 13

# In[41]:

mat[2:,1:]


# # Question 14

# In[45]:

mat[3,4]


# # Question 15

# In[47]:

mat


# In[48]:

mat[:3,1]


# # Question 16

# In[61]:

mat[4:]


# # Question 17

# In[62]:

mat


# In[63]:

mat[3:5,:]


# # Question 18

# In[69]:

np.sum(mat)


# In[70]:

mat.sum()


# # Question 19

# In[73]:

mat.std()


# # Question 20

# In[75]:

mat.sum(axis=0)


# In[ ]:





# coding: utf-8

# # Series

# In[1]:

import numpy as np


# In[2]:

import pandas as pd


# In[4]:

labels = ['a','b','c']
my_data = [10,20,30]
arr = np.array(my_data)
d = {'a':10,'b':20,'c':30}


# In[5]:

pd.Series(data = my_data)


# In[7]:

pd.Series(data=my_data,index=labels)


# In[8]:

pd.Series(arr)


# In[9]:

pd.Series(d)


# In[10]:

d


# In[11]:

pd.Series(data=[sum,print,len])


# In[12]:

ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])


# In[13]:

ser1


# In[14]:

ser2 = pd.Series([1,2,5,4],['USA','Germany','USSR','Japan'])


# In[22]:

ser2


# In[16]:

ser1['USA']


# In[17]:

ser3 = pd.Series(data=labels)


# In[18]:

ser3[0]


# In[19]:

ser1 + ser2


# # Data Frames - Part 1

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

from numpy.random import randn


# In[3]:

np.random.seed(101)


# In[4]:

df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])


# In[5]:

df


# In[6]:

df['W']


# In[7]:

type(df['W'])


# In[8]:

type(df)


# In[9]:

df['X']


# In[10]:

df[['W','Z']]


# In[11]:

df['new'] = df['W'] + df['Y']


# In[12]:

df


# In[13]:

df.drop('new',axis=1,inplace=True)


# In[14]:

df


# In[15]:

df.drop('E')


# In[16]:

df.shape


# In[17]:

df['Y']


# In[18]:

df[['Z','X']]


# In[19]:

df.loc['A']


# In[20]:

df.iloc[2]


# In[21]:

df.loc['B','Y']


# In[22]:

df


# In[23]:

df.loc[['A','B'],['W','Y']]


# # Data Frames - Part 2

# In[24]:

df > 0


# In[25]:

booldf = df>0


# In[26]:

booldf


# In[27]:

df[booldf]


# In[28]:

df[df>0]


# In[29]:

df['W']>0


# In[30]:

df['W']


# In[31]:

df[df['W']>0]


# In[32]:

df[df['Z']<0]


# In[33]:

resultdf = df[df['W']>0]


# In[34]:

resultdf


# In[35]:

resultdf['X']


# In[36]:

df[df['W']>0]


# In[37]:

df[df['W']>0]['X']


# In[39]:

df[df['W']>0][['Y','X']]


# In[41]:

boolser = df['W']>0
result = df[boolser]


# In[42]:

result


# In[43]:

boolser = df['W']>0
result = df[boolser]
mycols = ['Y','X']
result[mycols]


# In[48]:

df[(df['W'] > 0) | (df['Y'] > 1)]


# In[54]:

newind = 'CA NY WY OR CO'.split()


# In[55]:

newind


# In[56]:

df['States'] = newind


# In[57]:

df


# In[50]:

df.reset_index()


# In[58]:

df.set_index('States')


# # Data Frames Part 3

# In[59]:

import numpy as np
import pandas as pd


# In[60]:

# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[61]:

hier_index


# In[62]:

df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])
df


# In[65]:

df=pd.DataFrame(randn(6,2),hier_index,['A','B'])


# In[66]:

df


# In[67]:

df.loc['G1']


# In[69]:

df.loc['G1'].loc[1]


# In[70]:

df.index.names


# In[71]:

df.index.names = ['Groups', 'Num']


# In[72]:

df


# In[73]:

df.loc['G2']


# In[74]:

df.loc['G2'].loc[2]


# In[76]:

df.xs


# In[77]:

df.loc['G1']


# In[81]:

df.xs(1,level='Num')


# In[82]:

df.xs('G1')


# # Missing Data

# In[83]:

import numpy as np
import pandas as pd


# In[84]:

d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}


# In[85]:

df = pd.DataFrame(d)


# In[86]:

df


# In[87]:

df.dropna()


# In[88]:

df.dropna(axis=1)


# In[89]:

df.dropna(thresh=2)


# In[91]:

df.fillna(value='FILL VALUE')


# In[93]:

df['A']


# In[95]:

df['A'].fillna(value=df['A'].mean())


# # Groupby

# In[1]:

import numpy as np
import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}


# In[2]:

df = pd.DataFrame(data)


# In[3]:

df


# In[6]:

byComp = df.groupby('Company')


# In[7]:

byComp.mean()


# In[8]:

byComp.sum()


# In[9]:

byComp.std()


# In[10]:

byComp.sum()


# In[13]:

byComp.sum().loc['FB']


# In[15]:

df.groupby('Company').sum().loc['FB']


# In[17]:

df.groupby('Company').max()


# In[18]:

df.groupby('Company').min()


# In[19]:

df.groupby('Company').describe()


# In[20]:

df.groupby('Company').describe().transpose()


# In[21]:

df.groupby('Company').describe().transpose()['FB']


# In[22]:

df.groupby('Company').describe().transpose()['MSFT']


# In[23]:

df.groupby('Company').describe().transpose()['GOOG']


# # Merging Joining and Concatenating

# In[24]:

import numpy as np
import pandas as pd


# In[25]:

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])


# In[26]:

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 


# In[27]:

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])


# In[28]:

df1


# In[29]:

df2


# In[30]:

df3


# # Concatenation

# In[32]:

pd.concat([df1,df2,df3])


# In[34]:

pd.concat([df1,df2,df3],axis=1)


# # Data Frames

# In[36]:

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    


# In[37]:

left


# In[38]:

right


# # Merging

# In[40]:

pd.merge(left,right,how='inner',on='key')


# In[41]:

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})


# In[42]:

pd.merge(left, right, on=['key1', 'key2'])


# In[43]:

pd.merge(left, right, how='outer', on=['key1', 'key2'])


# In[44]:

pd.merge(left, right, how='right', on=['key1', 'key2'])


# In[45]:

pd.merge(left, right, how='left', on=['key1', 'key2'])


# # Joining

# In[46]:

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


# In[47]:

left.join(right)


# In[48]:

left.join(right, how='outer')


# # Operations

# In[49]:

import numpy as np
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()


# In[50]:

df['col2'].unique()


# In[51]:

df['col2'].nunique()


# In[52]:

df['col2'].value_counts()


# In[53]:

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]


# In[54]:

newdf


# In[55]:

def times2(x):
    return x*2


# In[56]:

df['col1'].apply(times2)


# In[57]:

df['col3'].apply(len)


# In[58]:

df['col1'].sum()


# In[59]:

del df['col1']


# In[61]:

df


# In[62]:

df.columns


# In[63]:

df.index


# In[64]:

df


# In[65]:

df.sort_values(by='col2') #inplace=False by default


# In[66]:

df.isnull()


# In[67]:

# Drop rows with NaN Values
df.dropna()


# In[68]:

import numpy as np


# In[69]:

df = pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()


# In[70]:

df.fillna('FILL')


# In[71]:

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)


# In[72]:

df


# In[73]:

df.pivot_table(values='D',index=['A', 'B'],columns=['C'])


# # Data Input and Output

# In[4]:

import pandas as pd


# In[5]:

pwd


# In[ ]:




# In[ ]:




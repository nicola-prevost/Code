#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


# In[3]:


sol = pd.read_csv('NewD.csv')
sol


# In[45]:


sns.boxplot(sol['XLOGP'])


# In[46]:


x1=np.where(sol['XLOGP']<-2)
print(x1)


# In[47]:


x=sol.drop( [253, 568, 569, 589, 602, 619, 627, 643])
print(x)


# In[49]:


x1=x['XLOGP']
y=x['Calculated pEC50']
plt.scatter(x1,y)
plt.xlabel('XLOGP', fontsize=20)
plt.ylabel('pEC50', fontsize=20)
plt.show()


# In[53]:


plt.hist(x['XLOGP'])


# In[50]:


import statsmodels.api as sm

x1 = sm.add_constant(x1)
print(x1)
result = sm.OLS(y, x1).fit()

result.summary()


# In[27]:


import statistics
statistics.stdev(y)


# In[51]:


result.rsquared


# In[29]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[31]:


print(mean_squared_error(x1, y))
print(mean_absolute_error(x1, y))


# In[4]:


import math

df = pd.DataFrame(sol)
s = df['Complexity']
ms=[]
x=0
for elem in sol.SMILES:
    results=math.sqrt(s[x])
    
    ms.append(results)
    x=x+1


# In[5]:


print(ms)


# In[6]:


plt.hist()


# In[42]:


plt.hist(x1)


# In[7]:


import statsmodels.api as sm

y1=df['Calculated pEC50']
#ms = sm.add_constant(ms)
print(ms)
result = sm.OLS(y, ms).fit()

result.summary()


# In[10]:


df2 = df.assign(Complexity2=ms)
df2.to_csv("NewD.csv", index=False)


# In[11]:


df2


# In[30]:


sns.boxplot(df2['Complexity2'])


# In[31]:


x1=np.where(df2['Complexity2']<3)
print(x1)


# In[32]:


x=df2.drop( [191, 192, 318, 343, 392])
print(x)


# In[40]:


x1=df2['Complexity2']
y=df2['Calculated pEC50']
plt.scatter(x1,y)
plt.xlabel('Complexity 2', fontsize=20)
plt.ylabel('pEC50', fontsize=20)
plt.show()


# In[44]:


import statsmodels.api as sm

x1 = sm.add_constant(x1)
print(x1)
result = sm.OLS(y, x1).fit()

result.summary()


# In[42]:


from sklearn.metrics import r2_score
r2 = r2_score(x1, y)


# In[43]:


r2


# In[ ]:





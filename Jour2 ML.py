#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")


# In[2]:


df = pd.read_csv("train.csv")
df
# retirer les colonnes PassengerId, Name, Ticket, Cabin
data = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
data
# Transformer Sex et Embarked en Valeur numérique
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data
data["Embarked"] = data["Embarked"].map({
    np.nan: 0, "S": 1, "C": 2, "Q": 3
})
data
# faire un df.dropna() pour ne garder que les valeur non nan
data.dropna(subset=["Age"], inplace=True)
data.count()


# In[3]:


data.head(10)


# In[4]:


data[data["Survived"] == 0].loc[:10]


# In[5]:


data[data["Survived"] == 0].iloc[:10]


# In[6]:


data_train = data.iloc[:500]
data_test = data.iloc[501:]


# In[7]:


# data.to_numpy()


# In[8]:


Y_train = data_train["Survived"]
X_train = data_train.drop(columns=["Survived"])

Y_test = data_test["Survived"]
X_test = data_test.drop(columns=["Survived"])


# In[9]:


from sklearn.svm import SVC

clf = SVC()

clf.fit(X_train, Y_train)

score = clf.score(X_test, Y_test)
score


# In[10]:


# len(Y_test[Y_test == 0]) / len(Y_test)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

clf.fit(X_train, Y_train)

score = clf.score(X_test, Y_test)
score


# In[12]:


clf.predict(X_test.iloc[:10])


# In[13]:


Y_test.iloc[:10]


# In[14]:


from sklearn.neighbors import KNeighborsClassifier

score_list = []
for n in range(1, 30):
    clf = KNeighborsClassifier(n_neighbors=n)

    clf.fit(X_train, Y_train)

    score = clf.score(X_test, Y_test)
    score_list.append((n, score))

score_list


# In[15]:


import numpy as np

# help(np.random.randn)
X = np.arange(30)
Y = np.random.randn(30)
# scores = np.array()
score_array = np.array(list(zip(X,Y)))
# help(np.random.random)


score_array = np.array(score_list)


# In[16]:


score_array


# In[17]:


score_array[0,1]


# In[18]:


score_array.shape


# In[19]:


score_array[0,:] # selectionner la premiere ligne
score_array[:,0] # selectionner la premiere colonne


# In[31]:


# import tkinter
get_ipython().run_line_magic('matplotlib', 'qt')

import matplotlib.pyplot as plt

plt.plot(score_array[:,0], score_array[:,1])
plt.show()
plt.savefig("scores.png")


# In[34]:


i = np.argmax(score_array[:,1])
score_array[i,0]


# In[37]:


data.plot.scatter("Age", "Fare", alpha=0.5, c=data["Survived"], cmap="Set1")


# In[43]:


clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, Y_train)

Y_predicted = clf.predict(X_test)
plt.subplot(2,1,1)
plt.scatter(X_test["Age"], X_test["Fare"], alpha=0.5, c=Y_predicted, cmap="Set1")

plt.subplot(2,1,2)
plt.scatter(X_test["Age"], X_test["Fare"], alpha=0.5, c=Y_test, cmap="Set1")


# In[41]:


# help(plt.subplot)


# In[ ]:





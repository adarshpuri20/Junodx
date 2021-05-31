#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from plotnine.data import economics
from plotnine import ggplot, aes, geom_line,geom_point
from factor_analyzer import FactorAnalyzer
#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import tensorflow as tf


# In[145]:


path='../Data/20210412_trainingdata.csv'
#Dataframe of Training Data
df_train=pd.read_csv(path)


# In[146]:


df_train.head(10)


# In[147]:


df_train['pred02'].iloc[np.where(df_train['pred02']<1)]=df_train['pred02'].iloc[np.where(df_train['pred02']<1)]*100


# In[148]:


df_train.describe()


# In[149]:


sns.distplot(df_train['class01'])


# In[150]:


sns.distplot(df_train['class02'])


# In[151]:


#scatter plot pred01/class01
var = 'pred01'
data = pd.concat([df_train['class01'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='class01', ylim=(0,4));


# In[152]:


#scatter plot pred01/class02
var = 'pred01'
data = pd.concat([df_train['class02'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='class02', ylim=(0,4));


# In[153]:


#scatter plot pred02/class01
var = 'pred02'
data = pd.concat([df_train['class01'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='class01', ylim=(0,4));


# In[154]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True);
#use pred06 not 5 and not 7


# In[155]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True);
#use pred06 not 5 and not 7


# In[156]:


#scatterplot
sns.set()
df=df_train.drop(['ID'],axis=1)
sns.pairplot(df)
plt.show()


# In[157]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[158]:


df_train=df_train.dropna()
df_train.isnull().sum().max()


# ## Univariate analysis
# 

# In[159]:


#histogram and normal probability plot
def normalcheck(df_train,col):
    sns.distplot(df_train[col], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train[col], plot=plt)


# In[160]:


normalcheck(df_train,'pred01')


# In[161]:


normalcheck(df_train,'pred02')


# In[162]:


normalcheck(df_train,'pred03')


# In[163]:


normalcheck(df_train,'pred04')


# In[164]:


normalcheck(df_train,'pred05')


# In[165]:



normalcheck(df_train,'pred06')


# In[166]:


normalcheck(df_train,'pred07')


# In[167]:


#### Geometric object to use for drawing
(
    ggplot(df_train)  # What data to use
    + aes(x='pred02', y='pred06')  # What variable to use
    + geom_point(aes(color='factor(class02)',shape='factor(class01)'))
)


# In[168]:


#### Geometric object to use for drawing
(
    ggplot(df_train)  # What data to use
    + aes(x='pred02', y='class01')  # What variable to use
    + geom_point(aes(color='factor(class02)'))
)


# In[169]:


df=df_train.drop(['ID','class01','class02'],axis=1)
df.head()


# In[170]:


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value
#In this Bartlett ’s test, the p-value is 0. The test was statistically significant,
#indicating that the observed correlation matrix is not an identity matrix.


# In[171]:


from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
kmo_model
#The overall KMO for our data is 0.69, which is not good


# In[172]:


kmo_all


# In[173]:


# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(df)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev


# In[174]:


# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
#factor above 1 to be taken


# In[175]:


fa = FactorAnalyzer()
fa.set_params(n_factors=3, rotation='varimax')
fa.fit(df)
loadings = fa.loadings_
loadings


# In[176]:


fa


# In[177]:


fa.get_factor_variance()
#Total 86% cumulative Variance explained by the 2 factors.


# ### Univariate Selection
# 
# Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# 
# Using chi-squared (chi²) statistical test for non-negative features to select best features

# In[193]:


path='../Data/20210412_trainingdata.csv'

#Dataframe of Training Data
df=pd.read_csv(path)
print('\nNumber of NUll values in the data',df.isnull().sum())

#Drops the Nan Values
df=df.dropna()
df['pred02'].iloc[np.where(df['pred02']<1)]=df['pred02'].iloc[np.where(df['pred02']<1)]*100
#Splitting class01 and class02 in two variables

class01=df['class01']
class02=df['class02']

#one hot encoding each classes
class01=tf.keras.utils.to_categorical(
    class01, num_classes=4, dtype='float32')
class02=tf.keras.utils.to_categorical(
    class02, num_classes=4, dtype='float32')

#Concatenates the one hot encoded vectors
encoded_class=np.concatenate((class01,class02),axis=1)
df


# In[194]:


def data_preprocessing(df):
    '''The Function Normalizes the data for each column.
        Input is the training data frame
        and the output Nomralized Data frame'''
    df['pred02'].iloc[np.where(df['pred02']<1)]=df['pred02'].iloc[np.where(df['pred02']<1)]*100
    df['pred05']=np.log(df['pred05'].values+1)
    df['pred06']=np.log(df['pred06'].values+1)
    df=df.drop(['ID','class01','class02','pred07'],axis=1)
    df=df.dropna()

    return df

df_train=data_preprocessing(df)
df_train


# In[195]:


class01=df['class01']
class02=df['class02']


# In[196]:


bestfeatures2 = SelectKBest(score_func=chi2, k='all')
fit2 = bestfeatures2.fit(df_train,class02)
bestfeatures1 = SelectKBest(score_func=chi2, k='all')
fit1 = bestfeatures1.fit(df_train,class01)


# In[197]:


dfscores1 = pd.DataFrame(fit1.scores_)
dfcolumns1 = pd.DataFrame(df_train.columns)
dfscores2 = pd.DataFrame(fit2.scores_)
dfcolumns2 = pd.DataFrame(df_train.columns)


# In[198]:


#concat two dataframes and scale for better visualization 
featureScores = pd.concat([dfcolumns1,dfscores1/1000],axis=1)
featureScores.columns = ['Feature','Score']  #naming the dataframe columns
featureScores


# The chi2 shows pre01, pred06 are strong features for class01

# In[199]:


#concat two dataframes and scale for better visualization 
featureScores = pd.concat([dfcolumns2,dfscores2/1000],axis=1)
featureScores.columns = ['Feature','Score']  #naming the dataframe columns
featureScores


# The chi2 shows pre01, pred06 are strong features for class02

# But Correlation Matrix tells us pred01 is not a good learner so dropped

# ### Feature Importance
# The feature importance of each feature of the dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.

# In[200]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model1 = ExtraTreesClassifier()
model1.fit(df_train,class01)
model2 = ExtraTreesClassifier()
model2.fit(df_train,class02)


# In[201]:


print(model1.feature_importances_) #use inbuilt class1 feature_importances of tree based classifiers
print(model2.feature_importances_) #use inbuilt class2 feature_importances of tree based classifiers


# In[202]:


#class01 feature importance treeclassifier model
feat_importances = pd.Series(model1.feature_importances_, index=df_train.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()


# In[203]:


feat_importances = pd.Series(model2.feature_importances_, index=df_train.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()


# In general, the idea of random forests is to reduce the variance of the predictions while retaining
# low bias by averaging over many noisy trees.

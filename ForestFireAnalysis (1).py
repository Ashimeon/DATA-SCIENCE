#!/usr/bin/env python
# coding: utf-8

# # Forest Fire analysis and prediction

# - Data Collection
# - Data Pre-Processing
# - Exploratory Data Analysis
# - Feature Engineering
# - Feature Selection
# - Model Building
# - Model Selection
# - Hyperparameter Tuning
# - Regression

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("./DATA/Algerian_forest_fires_dataset.csv")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.describe()


# In[9]:


data.info()


# In[10]:


data.day.unique()


# In[11]:


data[data.isnull().any(axis=1)] # inorder to check the row which is having the missing values


# # Here after 123 we have the data set of new region

# In[12]:


data.loc[:122,'Region']=1  #upto 122 ,Region =1 but intialize as 1.0
data.loc[122:,'Region']=2  #After 122 , Region = 2 this as 2.0
data[['Region']] = data[['Region']].astype(int) #1.0 is coverted to 1 astype integer
#It is used to convert the data in the "Region" column of a pandas DataFrame to integer type.
data.head()


# In[13]:


data.tail()


# In[14]:


data.drop([122,123,168], axis=0, inplace=True)#.reset_index(drop=True)
data.day.unique()
#  Drop rows at index labels 122 and 123,168


# In[15]:


filtered_rows = data[data['day'] == 'day']
filtered_rows


# In[16]:


data.drop([124], axis=0, inplace=True)#.reset_index(drop=True)
data.day.unique()


# In[17]:


data.shape


# In[18]:


data.month.unique()


# In[19]:


data.year.unique()


# In[20]:


data.columns.unique()


# In[21]:


data


# In[22]:


data.shape


# In[23]:


data.columns


# In[24]:


# Spaces were fixed in the column names
# The str.strip() function is applied to each column name using the str accessor, which allows you to perform string operations on the elements of a column.
# After applying the strip() function, the modified column names are assigned back to the data.columns attribute, updating the column names in the DataFrame.
data.columns = data.columns.str.strip()
data.columns 


# In[25]:


data.isna().sum()


# In[26]:


data.reset_index()


# In[27]:


data.reset_index(drop=True, inplace=True)
data


# In[28]:


# Check if the default index is in proper format without gaps
is_proper_index = data.index.equals(pd.RangeIndex(len(data)))

print(is_proper_index)


# 

# In[29]:


data.isna().sum()


# In[30]:


print(data.duplicated())
print(data[data.duplicated()])


# In[31]:


data.shape


# ### ANALYSE

# In[32]:


data.describe(include = 'all')


# In[33]:


data["Classes"].value_counts()


# our dependent feature(Classes) containig only two categories but due to misspace it is showing multiple category so need to change the spaceing in order to make two category

# In[34]:


data.Classes = data.Classes.str.strip()


# In[35]:


data["Classes"].value_counts()


# In[36]:


data["Region"].value_counts()


# In[37]:


data.describe(include='all')


# In[38]:


data[['month', 'day', 'year', 'Temperature', 'RH', 'Ws']] = data[['month', 'day', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

objects = [features for features in data.columns if data[features].dtypes == 'O']  #"o" object type
for i in objects:
    if i != 'Classes':  #exxcept classes
        data[i] = data[i].astype(float)


# In[39]:


print(data.dtypes)


# In pandas, the object data type is commonly used to represent categorical data. While the object data type can also be used to store other types of data (such as strings), it is often used to represent variables with a limited number of discrete categories or labels.

# In[40]:


data.describe(include="all")


# In[41]:


data[:122]


# In[42]:


data[122:]


# In[43]:


# Encoding Not fire as 0 and Fire as 1
data['Classes']= np.where(data['Classes']== 'not fire',0,1)
data.head(10)
#If the condition is true, 
# the corresponding element in the 'Classes' column is assigned the value 0.
#  Otherwise, it is assigned the value 1.


# In[44]:


data.tail()


# In[45]:


# Check counts
data.Classes.value_counts()


# In[46]:


correlation = data.corr()
correlation


# The value of correlation helps in determining the strength and direction of the relationship between two variables. When using correlation for feature selection, the value of correlation can guide you in selecting relevant features

# # Visualize

# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


plt.figure(figsize=(20,15))  #size of figure
sns.heatmap(correlation,annot= True,linewidths=1, linecolor="white", cbar=True, cmap = "Paired",xticklabels="auto", yticklabels="auto")
     


# In[49]:


data.to_csv('./DATA/ForestFireDataCleaned.csv', index=False)


# In[50]:


sns.countplot(x='Classes',data=data)
plt.title('Class Distributions \n 0: No Fire || 1: Fire', fontsize=14)
plt.show()


# In[51]:


sns.countplot(x='Region',hue='Classes',data=data)
plt.title('Region Distributions \n 1: Region 1 || 2: Region 2', fontsize=14)
plt.show()


# In[52]:


# PLot density plot for all features
#plt.style.use('seaborn')
data.hist(bins=50, figsize=(20,15), ec = 'b')
plt.show()


# In[53]:


# countplot over target variable

sns.countplot(x='Temperature',hue='Classes', data=data)
plt.title('Class Distribution \n 0: No Fire || 1: Fire', fontsize=14)


# In[54]:


data.month.unique()


# In[55]:


#month wise fire analysis for region 1
dftemp= data.loc[data['Region']== 1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= dftemp,ec = 'black', palette= 'Set2')#ec='black' sets the edge color of the categorical plot elements to black, and palette='Set2' sets the color palette to the 'Set2' palette.
plt.title('Fire Analysis Month wise for Region-1', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# In[56]:


#month wise fire analysis for region 2
dftemp= data.loc[data['Region']== 2]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= dftemp,ec = 'black', palette= 'Set2')#ec='black' sets the edge color of the categorical plot elements to black, and palette='Set2' sets the color palette to the 'Set2' palette.
plt.title('Fire Analysis Month wise for Region-2', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# Yearly plot

# In[57]:


plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='year',hue='Classes',data= data,ec = 'black', palette= 'Set2')#ec='black' sets the edge color of the categorical plot elements to black, and palette='Set2' sets the color palette to the 'Set2' palette.
plt.title('Fire Analysis of Year 2012', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Year', weight= 'bold')
plt.legend(loc='upper right')
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# In[58]:


# ! pip install scikit-learn


# In[59]:


plt.subplots(figsize=(15,6))
sns.set_style('whitegrid')
sns.countplot(x='Rain', hue='Classes', data=data)

# Set the labels and title
plt.xlabel('Rain')
plt.ylabel('Count')
plt.title('Distribution of Classes by Rain')

# Show the plot
plt.show()


# ### With the increase in Rain Fire chance is descreases

# In[60]:


plt.subplots(figsize=(15,6))
sns.set_style('whitegrid')
sns.countplot(x='Ws', hue='Classes', data=data)

# Set the labels and title
plt.xlabel('WS')
plt.ylabel('Count')
plt.title('Distribution of Classes by Wind Speed:6 to 29')

# Show the plot
plt.show()


# In[61]:


plt.subplots(figsize=(15,6))
sns.set_style('whitegrid')
sns.countplot(x='RH', hue='Classes', data=data)

# Set the labels and title
plt.xlabel('RH')
plt.ylabel('Count')
plt.title('Distribution of Classes by humidity rate')

# Show the plot
plt.show()


# In[62]:


plt.subplots(figsize=(15,6))
sns.set_style('whitegrid')
sns.countplot(x='month', hue='Classes', data=data)

# Set the labels and title
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Distribution of Classes by Month')

# Show the plot
plt.show()


# In[63]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Assuming 'data' is the DataFrame containing the dataset
# Splitting the features and target variable
X = data.drop('Classes', axis=1)
y = data['Classes']

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Display the correlation matrix
print(correlation_matrix)

# Train a Random Forest classifier
rfmodel = RandomForestClassifier()
rfmodel.fit(X, y)

# Get feature importances
feature_importances = pd.Series(rfmodel.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display feature importances
print(feature_importances)


# In[64]:


plt.subplots(figsize=(18,10))

sns.heatmap(correlation_matrix , annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# From above anlysis  we find some non importance features:
# - Year (due to missing values)
# - Ws (wind speed) low correlation
# - DAY(low corr)
# - Month(low corr)

# In[65]:


df = data.drop(['day','month','year','Ws'], axis=1)
df.head(10)


# Spliting Data Set 

# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


y = df['Classes']
X = df.drop('Classes',axis=1)
y.tail()


# In[68]:


X.head()


# In[69]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
X_train.head()


# In[70]:


X_train.info()


# In[71]:


del data,X,y,df


# In[72]:


from sklearn.model_selection import GridSearchCV
#TRain Function is defined
def train(X_train, y_train, model, hyperparameters):
    grid_search = GridSearchCV(estimator=model,param_grid=hyperparameters, cv = 5)
    grid_search.fit(X_train, y_train)
    
    
    #print the best hyperparameters found
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    
    # Train the model with best hyperparametres
    best_model = model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    # Print the intercept and coefficients of the best model
    # print('Intercept is :', best_model.best_estimator_.intercept_)
    # print('Coefficient is :', best_model.best_estimator_.coef_)

    # Evaluate the best model on the test data
    scores = best_model.score(X_test, y_test)
    print('Score_test_data:', scores)
    
    return best_params, best_model


# In[73]:


from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error

# EVALUATION

def evaluate_model(X_test, y_test, best_model):
    # it will evaluate the score by taking testing data with best model
    
    #predict the target values for the best set
    y_pred = best_model.predict(X_test)
    
    # Calculate the MSE
    mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared
    r2 = r2_score(y_test, y_pred)

# Calculate the adjusted R-squared
    # adjusted_r2 = adjusted_r2_score(y_test, y_pred)

# Calculate the MAE
    mae = mean_absolute_error(y_test, y_pred)

# Print the scores
    print("MSE:", mse)
    print("R-squared:", r2)
    # print("Adjusted R-squared:", adjusted_r2)
    print("MAE:", mae)

    return mse,r2,mae


# # Linear Regression

# In[74]:


from sklearn.linear_model import LinearRegression
# Define the hyperparameters to tune
hyperparameters = {
    # "regularization": ["l1", "l2"],
    # "learning_rate": [0.01, 0.001, 0.0001],
    # "number_of_epochs": [10, 50, 100],
}
model = LinearRegression()
_,best_model = train(X_train,y_train,model,hyperparameters)
# print('Intercept is :',best_model.intercept_)
# print('Coefficient is :',best_model.coef_)
scores = evaluate_model(X_test,y_test,best_model)    


# # Ridge

# In[75]:


from sklearn.linear_model import Ridge
# Define the hyperparameters to tune
hyperparameters = {
    "alpha": np.logspace(-4, 4, 10),
}

# Create a Ridge model
model = Ridge()
_,best_model = train(X_train,y_train,model,hyperparameters)
scores = evaluate_model(X_test,y_test,best_model)   



# # Lasso

# In[76]:


from sklearn.linear_model import Lasso
# Define the hyperparameters to tune
hyperparameters = {
    "alpha": np.logspace(-4, 4, 10),
}

# Create a Lasso model
model = Lasso()
_,best_model = train(X_train,y_train,model,hyperparameters)
scores = evaluate_model(X_test,y_test,best_model)   



# # Decision Tree

# In[77]:


from sklearn.tree import DecisionTreeRegressor
# Define the hyperparameters to tune
hyperparameters = {
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10],
}

# Create a decision tree regressor
model = DecisionTreeRegressor()
_,best_model = train(X_train,y_train,model,hyperparameters)
scores = evaluate_model(X_test,y_test,best_model)   



# In[78]:


import matplotlib.pyplot as plt

# Scores of each model
linear_score = 0.6840775270198252
lasso_score = 0.6695115483312191  
ridge_score = 0.6753837888234437
decision_tree_score = 0.9861363636363636

# Models names
models = ['Linear Regression', 'Lasso', 'Ridge', 'Decision Tree']

# Scores for each model
scores = [linear_score, lasso_score, ridge_score, decision_tree_score]

# Plotting the scores
plt.bar(models, scores)
plt.title('Comparison of Model Scores')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.show()


# In[ ]:





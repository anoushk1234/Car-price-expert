# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd


# %%
df=pd.read_csv("dataset\car data.csv")


# %%
df.shape


# %%
a=['Seller_Type',"Transmission","Owner"]
for item in a:
    print(df[item].unique(),end="\n")


# %%
df.isnull().sum()


# %%
df.describe()


# %%
df.columns


# %%
processed_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# %%
processed_dataset.head()


# %%
processed_dataset['Current_Year']=2021


# %%
processed_dataset.head()


# %%
processed_dataset['no_of_years']=processed_dataset['Current_Year']-processed_dataset['Year']


# %%
processed_dataset.head()


# %%
processed_dataset.drop(["Year"],axis=1,inplace=True)


# %%
processed_dataset.head()


# %%
processed_dataset.drop(['Current_Year'],axis=1,inplace=True)


# %%
processed_dataset.head()


# %%
#dropped one column because dummy variable trap- if 2 features are there u can get value of one feature from other when its one hot encoded
processed_dataset=pd.get_dummies(processed_dataset,drop_first=True)


# %%
processed_dataset.head()


# %%
processed_dataset.corr()


# %%
import seaborn as sns 


# %%
sns.pairplot(processed_dataset)


# %%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
#heatmap
corr_matrix=processed_dataset.corr()
# print(eng_features)
# print(processed_dataset.columns)
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap="RdYlGn")


# %%
processed_dataset.head()


# %%
x=processed_dataset.iloc[:,1:]
y=processed_dataset.iloc[:,0]


# %%
y.head()


# %%
x.head()


# %%
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)


# %%
print(model.feature_importances_)


# %%
ftr_imp=pd.Series(model.feature_importances_,index=x.columns)
ftr_imp.nlargest(5).plot(kind='barh')
plt.show()


# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# %%
x_train.head()


# %%
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()


# %%
import numpy as np


# %%
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200, num=12)]
print(n_estimators)


# %%
#random srch cv
from sklearn.model_selection import RandomizedSearchCV
random_params={'n_estimators':n_estimators,
             'max_features':['auto','sqrt'],
             'max_depth':[int(x) for x in np.linspace(start=5,stop=30, num=6)],
             'min_samples_split':[2,5,10,15,100],
             'min_samples_leaf':[1,2,5,10]}
print(random_params)


# %%
rf_tuned=RandomizedSearchCV(estimator=rf_reg,param_distributions=random_params,scoring="neg_mean_squared_error",n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


rf_tuned.fit(x_train,y_train)
# %%
predictions=rf_tuned.predict(x_test)
# %%
predictions
# %%
sns.displot(y_test-predictions)
# %%
plt.scatter(y_test,predictions)
# %%
import pickle
file =open('car_price_model.pkl','wb')
pickle.dump(rf_tuned,file)

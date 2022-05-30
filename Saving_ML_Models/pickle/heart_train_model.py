
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pickle

url = "https://raw.githubusercontent.com/MadhuraSrinivasan/ML-Projects/main/heart.csv"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
heart_df = pd.read_csv(url,names=names)

#print(heart_df)
x1 = heart_df.iloc[::,0:len(heart_df.columns)-1]
#print(x)
y1= heart_df['target']

x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3)

model1 = RandomForestClassifier(min_samples_split=3,n_estimators=200,n_jobs=-1)
model1.fit(x_train,y_train)

#to calculate accuracy 
result = model1.score(x_test,y_test)

#saving model to the disk 
filename1 = 'heart_model.sav'
pickle.dump(model1, open(filename1,'wb'))


 



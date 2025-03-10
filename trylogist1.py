from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
import pandas as pd

og_dataframe = pd.DataFrame(pd.read_csv('C:\\salilpython\\minprojectsem4\\dataset_diab\\diabetes.csv'))


# without insulin
dat_no_insulin = og_dataframe.drop('Insulin',axis=1)

# splitting columns for scaling- keep pregnancies and  DiabetesPedigreeFunction 
split_df = dat_no_insulin.drop(['Pregnancies','DiabetesPedigreeFunction','Outcome'],axis=1)

# column transform
scaler_obj = ColumnTransformer([('scaler',StandardScaler(),split_df.columns)],remainder='passthrough')
scaled_arr = scaler_obj.fit_transform(dat_no_insulin)
scaled_df = pd.DataFrame(scaled_arr,columns=['Glucose','BloodPressure','SkinThickness','BMI','Age','Pregnancies','DiabetesPedigreeFunction','Outcome'])

# train test split
scaled_df_X = scaled_df[['Glucose','BloodPressure','SkinThickness','BMI','Age','Pregnancies','DiabetesPedigreeFunction']]
scaled_df_Y = scaled_df['Outcome']
X_train,X_test,Y_train,Y_test = train_test_split(scaled_df_X,scaled_df_Y,random_state=42,test_size=0.2,stratify = scaled_df_Y)

# apply logistic regression
log_model = LogisticRegression(random_state=42,n_jobs=-1,class_weight='balanced',max_iter=800,C=0.8)
log_model.fit(X_train,Y_train)

# predictions
Y_pred = log_model.predict(X_test)
print(accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
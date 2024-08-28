<H3>ENTER YOUR NAME: JEEVAGOWTHAM S </H3>
<H3>ENTER YOUR REGISTER NO: 212222230053</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 19.08.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
``` py
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))


```


## OUTPUT:
### The Dataset:
![image](https://github.com/user-attachments/assets/1b5e37ae-cf46-4e4b-99cc-96558d156492)

### Dropping unwanted features
![image](https://github.com/user-attachments/assets/9bb116a3-fa64-4895-ba61-91b6f13bdc74)

### Checking for null values
![image](https://github.com/user-attachments/assets/48d8d3cc-8dad-41ec-8240-c8a2e9e07391)

### Checking for duplication
![image](https://github.com/user-attachments/assets/32fa44a8-5597-4852-a32e-6a9c8318d22c)

### Describing the dataset
![image](https://github.com/user-attachments/assets/727966e6-2772-4364-9740-86f9936be6fc)

### Scaling the values
![image](https://github.com/user-attachments/assets/d9c85c51-7400-4e80-9bba-f9cf1adcfe5b)

### X Features
![image](https://github.com/user-attachments/assets/2d0e2250-8088-4f24-b32c-dbd8196a8292)

### Y Features
![image](https://github.com/user-attachments/assets/f2d47c8e-32bd-4a3f-b5d3-0e9cfc1deeb4)

### Splitting the training and testing dataset
![image](https://github.com/user-attachments/assets/66523834-cb32-42fb-9a39-9c2e3b651dbc)










## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



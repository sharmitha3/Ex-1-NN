<H3>ENTER YOUR NAME : SHARMITHA V</H3>
<H3>ENTER YOUR REGISTER NO : 212223110048</H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>
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
~~~
Developed by:Preetha.S
Register no :212222230110

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))
~~~
## OUTPUT:

![image](https://github.com/user-attachments/assets/cd062303-dbdc-49ad-9590-e278b59f69b7)

X VALUES 
![307609070-edb69d82-0e6a-4d0f-ab38-6ed5f25d388d](https://github.com/user-attachments/assets/9eb27172-5916-4fb0-9617-2c839633432c)

YVALUES
![307609082-f95f53c9-b209-40de-9587-0fb3018dea5c](https://github.com/user-attachments/assets/f8ccef71-6d6b-4997-bcbc-3de888fae494)

NULL VALUES 
![307609092-946da728-2c72-470b-bfe6-06b94c42e77e](https://github.com/user-attachments/assets/e2024f09-f8d0-43c4-95bf-8fdc3a9b2a0f)

DUPLICATED VALUES

![307609131-f89fe104-7abc-46a0-979c-9d1d2c24e817](https://github.com/user-attachments/assets/be774b68-8358-40f3-ab31-8c9abaf1bf5e)

DESCRIPTION:
![307609154-58a0f531-3c53-4319-a8d6-ab7b1f866846](https://github.com/user-attachments/assets/3c160420-16fc-44ab-8eb2-869b57f54888)

NORMALIZED DATASET:
![307609173-2c7b171d-c312-4404-8cab-9af1e1721856](https://github.com/user-attachments/assets/a2369a6d-7df7-4abe-85e1-9b6b40ba9345)

TRAINING DATA:
![307609209-b0b85b94-dad2-411e-ad0a-f7d5ba55555a](https://github.com/user-attachments/assets/f8851300-52ca-4aab-9f6d-252251b05d0c)

TESTING DATA:
![307609243-8455f181-eac8-47a3-b934-4b1ee4f0197a](https://github.com/user-attachments/assets/0e75525d-a519-4d77-b839-95385ded19e9)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



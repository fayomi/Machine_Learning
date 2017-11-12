#This machine learning script tries to predict which people survived the titanic


import numpy as np
import pandas as pd


# the machine learning algorithm
from sklearn.ensemble import RandomForestClassifier
#Test-Train split
from sklearn.cross_validation import train_test_split
#to switch off pandas warnings
pd.options.mode.chained_assignment=None
#used to write our model to a file
from sklearn.externals import joblib

data=pd.read_csv('titanic_train.csv')

# There are a lot of ages missing so we will fill them out by providing the median age
median_age=data['age'].median()
#we then modify the dataset to put the median age where age is not specified
data['age'].fillna(median_age, inplace=True)
print data.head()
#define input data and expected output data
#This is our input data
data_input=data[['pclass', 'age', 'sex']]
print data_input.head(),
#This is our expected output data
expected_output=data['survived']
print expected_output.head()

# now we CLEAN our data by codifying all the strings into numbers
# pclass 1st=1, 2nd=2, 3rd=3
data_input['pclass'].replace('3rd', 3, inplace=True)
data_input['pclass'].replace('2nd', 2, inplace=True)
data_input['pclass'].replace('1st', 1, inplace=True)
# A different way of codifying if there are only two variables
#where data_input sex equals female, give 0, else, 1
# sex female=0, male=1
data_input['sex']=np.where(data_input['sex']=='female', 0, 1)
print data_input.head()

#time to divide data into training section and test section
#x is input, y is expected output (ML terminology convention)
#train_test_split was imported from sklearn(input, expected output, test_size=between 0 and 1, random state=42)

x_train, x_test, y_train, y_test =train_test_split(data_input, expected_output, test_size=0.33, random_state=42)

# define RandomForestClassifier variable
rf=RandomForestClassifier (n_estimators=100)

# To train the input with expected output we use the fit() function
rf.fit(x_train, y_train)

# time to test how accurate our model is
accuracy=rf.score(x_test, y_test)
# we multiply it by 100 to get a percentage. so now our model is about 78% accurate
print accuracy*100

# Now we write our model to a file so it can be reused, using joblib
# joblib.dump(model, 'assigned name', compress=9) the compress, puts it all in one file
joblib.dump(rf, 'titanic_model1', compress=9)

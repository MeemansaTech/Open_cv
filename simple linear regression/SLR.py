# SIMPLE LINEAR REGRESSION
#******************************************************************************************************************
import numpy as np 			
#np is the shortcut name of numpy
import matplotlib.pyplot as plt		
#plt is the shortcut name of matplotlib.pyplot
import pandas as pd			
#pd is the shotcut name of pandas

#******************************************************************************************************************

#  IMPORTING THE DATASET
dataset = pd.read_csv(r"C:\Users\kdata\Desktop\3.DATA\3. AMXWAM DATASCIENCE ACADEMY\2. AMXWAM 4th Batch (Apr 21 - July 21)\1.AMXWAM CLASS\3. JUNE'21\14.12th Jan_Class-14\SIMPLE LINEAR REGRESSION\Salary_Data.csv")
X = dataset.iloc[:, :-1].values	
#iloc returns a pandas series when one row is selected(:-1 exclude column from right side)
y = dataset.iloc[:,1].values    
#: colon will read from first to last column dataset (3 to get the only dependent variale becuase python indexing start from 0)

#*******************************************************************************************************************

#SPLITING THE DATASET IN TRAINING SET & TESTING SET

from sklearn.model_selection import train_test_split
#import the train_test_split class from sklearn.model_selection
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#if you ctrl+I then you declare array first, Testsize is we will consider 1/3rd data, random_state would be none
#you can select b/w 20-30% & not required to entered train size, if you want you can try
# Feature scaling are not required becuase most of the library are automatically take care of the feature scalling part
# simple linear regression feature selection is not required
# till above steps we are done with data preprocessing part

#*******************************************************************************************************************

#Fitting simple linear regression to the training set & specific librar will take car for fit into training 
#we will import the LinearRegression class from sklearn & create an linearregresson object of the class
#LinearRegressor class contain several methods one of them is fit method 
from sklearn.linear_model import LinearRegression
# We will import LinearRegression class from sklearn.linear_model
regressor = LinearRegression()
#Lets create regressor object for LinearRegression class
regressor.fit(X_train, y_train)

#fit the regressor object to our training set & by using this method model will learn automatically
#by using regressor.fit machine able to learn training data using simple linear regression model 
#machine is able to understand the corelation b/w No. of experience vs the salary (X & Y)
#This is most simple machine learning model which we will build or we created 1st machine learning model

#*******************************************************************************************************************

#PREDICTING THE TEST RESULT 
#1st step we import the library, 2nd step data has been preprocess
#3rd step import simplelinearregressor to build the machine learning model and model has trained the training phase
#we are tellin to machine to learn the corelation b/w No of yr exper vs salary itself using regressor.fit method 
#Next step we will see how our machine is predict which will for the test set of observation

y_pred = regressor.predict(X_test)

#let me create an object of predicted value & y_pred is the vector of prediction of dependent variable
#in this simple linear regression example since the depnendent variable is salary this vector will contain only predicted salary for all observation of our test set
#his time we will use predict method insted of fit method, this predict will use the prediction of aslary 
#now our vector prediction is ready to build & let me execute this, check the y_pred in variable explore
#lets try to understand the distinguish b/w y_test & y_pred, y_test is real or actual salary , y_pred is predicted salary
#y_pred salary has predicted by simplelinear regression model, lets take compare b/w actual & predicted salary for 1st observation
#actual & predicted sal correspond to same employee, actual sary of 1st employee is 37k& machine predict sal is 40k & i can say that this is good prediction
#if you look at the 7th employee actual sal is 55k & as per model predicted sal is 64k, in this condition the sal is overestimated
#now lets see the graphic visualization or corelation b/w 2 variabe so that we will have clear understanding
#if you check the last observation last employ has almost accurate prediction on actual & predicted salary
#next part is to see the ploting line for straight fitting line b/w actual & predicted

#*******************************************************************************************************************

#VISUALISING THE TRAINING SET RESULTS

#Lets plot our observation point & SLR model and we will see how predicted salary how close to the actual salary or actual observation
#before plot the graph x-axis we will take no.of experience, y-axis we will take salary part & we will take scatter plot for this observation
plt.scatter(X_train, y_train, color = 'red') 
#plt.scatter is function of pyplot & we will make observation of the dataset
#first we will assign the x-train for x-coordinate (containt no.of users experience) & for y-coordinate we will take y_train which has real salary & we will mark as color as red & regression line is blue
#now plot the regression line 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#x-coordinate we will take as x-train, y-coordiante we will take as prediction of x-train no. of users of experience are the predicted salary of the x-train observation
plt.title('Salary vs Experience (Training set)')
#let me name as title salary and experinece, we considered only traine set & havent seen for test set yet
plt.xlabel('Years of Experience')
#xlable we will mention of years of experience 
plt.ylabel('Salary')
#ylabel we will mention as salary of only training data
plt.show()

#let me check the distinguis between real value and predicted value,red points are real value & blue line are the predicted value
#e.g if an employee have 4 yrs of experience actul sal is about 55k but per predicted sal we got as 60k & same thing projected in regression line
#clearly dependency b/w independent & dependent variable & our regression line is approchng quite well 
#when i talk about prediction then we have some accurate prediction & less accurate 
#now we can say our machine fit the correct slr model machine understanding very well & now we will see the test data or test observation
#now we will plot same kind of graph and lets plot the new observation point in test set & we will see ohow slr will predict the test points

#*****************************************************************************************************************

#VISUALISING THE TEST SET RESULTS

plt.scatter(X_test, y_test, color = 'red')
#insted of training set we will change to test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#in this code we dont need to change X_test because our regressor is trained in the training set thats why we will mention X_train only
#instead of X_train in first if you put X_test then you might get some new  point & if you check the regressor we just fit only train data not for test data
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
    
#lets find the new test set result blue line is predict result which we created from SLR model from the training set but now red points are new test points & before the observation are trainign set
#finally business goal is predicted salary is same as actual salary as per the graph& we can say the model is work well on new test set point
#SLR is perfect able to predict new observation & this is how we saw the how machine build the first simple linear regression using the data
#Real time we wont get this type of case type but for explanation we have start in this way

#*******************************************************************************************************************




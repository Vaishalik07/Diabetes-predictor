# Import libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from keras.regularizers import l2
from sklearn.externals import joblib
from sklearn import datasets
import tensorflow as tf
global graph,model

graph = tf.get_default_graph()


# Get the dataset
#dataset = 
url = "Dataset/diabetes_csv.csv"
df1 = pd.read_csv(url)
print(len(df1))
val01 = df1
'''

# Split the dataset into features and labels
X = list(val01.columns[0:8]) 
y = val01.columns[8] 

# Split the dataset into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = True)
'''

X1 = val01.iloc[:,0:8]                          # slicing: all rows and 0 to 8 cols
#print(X)

# store response vector in "y1"
y1 = val01.iloc[:,8]                            # slicing: all rows and 8th col


le = preprocessing.LabelEncoder()
le.fit(y1)
y1=le.transform(y1)

val01['EncodedClasses'] = y1

print('Classes:',le.classes_)
print('Response variable after encoding:',y1)
val01.tail(10)


X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0)
#x_train.shape,y_train.shape, x_test.shape,y_test.shape


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, prediction))

# Save the model to disk
joblib.dump(classifier, 'classifier.joblib')


#joblib.dump(model, 'classifier.joblib')

'''
model = Sequential()

#Hidden layer-1
#model.add(Dense(100, activation = 'relu', input_dim=8, kernel_regularizer=l2(0.01)))
model.add(Dense(8, activation = 'relu', input_dim=8, kernel_initializer="uniform"))

#Hidden layer-2
#model.add(Dense(100, activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(Dense(12, activation = 'relu', input_dim=8, kernel_initializer="uniform"))

#Output LayerLayer
model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))
#Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model_output = model.fit(x_train, y_train, epochs=20, batch_size=20, verbose=1, validation_data=(x_test, y_test))

print(model_output)
print('Training Accuracy:', np.mean(model_output.history["acc"]))
print('Validation ACcuracy:', np.mean(model_output.history["val_acc"]))

#predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
y_pred = model.predict(x_test)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded, dtype='int64')

print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

#Save the model
filename = 'finalized.sav'
joblib.dump(model, open(filename, 'wb'))

joblib.dump(model, 'classifier.joblib')
'''
'''import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, precision_score
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
import numpy as np
# import pickle
from sklearn.externals import joblib

#Load dataset
url = "Dataset/diabetes_csv.csv"
df1 = pandas.read_csv(url)
print(len(df1))
val01 = df1

features_columns = list(val01.columns[0:8])            #'preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age'
target_column = val01.columns[8]                      # class
print(features_columns)

print('Features:',features_columns)
print('Target:',target_column)

#store feature matrix in "X1"
X1 = val01.iloc[:,0:8]                          # slicing: all rows and 0 to 8 cols
#print(X)

# store response vector in "y1"
y1 = val01.iloc[:,8]                            # slicing: all rows and 8th col

print(y1.shape)
print(X1.shape)

le = preprocessing.LabelEncoder()
le.fit(y1)
y1=le.transform(y1)

val01['EncodedClasses'] = y1

print('Classes:',le.classes_)
print('Response variable after encoding:',y1)
val01.tail(10)


x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0)
x_train.shape,y_train.shape, x_test.shape,y_test.shape

model = Sequential()

#Hidden layer-1
model.add(Dense(100, activation = 'relu', input_dim=8, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape= None, seed=None))

#Hidden layer-2
model.add(Dense(100, activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape= None, seed= None))

#Output LayerLayer
model.add(Dense(1, activation='sigmoid'))
#Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model_output = model.fit(x_train, y_train, epochs=20, batch_size=20, verbose=1, validation_data=(x_test, y_test),)

print(model_output)
print('Training Accuracy:', np.mean(model_output.history["acc"]))
print('Validation ACcuracy:', np.mean(model_output.history["val_acc"]))

#predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
y_pred = model.predict(x_test)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded, dtype='int64')

print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

#Save the model
filename = 'finalized.sav'
joblib.dump(model, open(filename, 'wb'))

# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)
'''
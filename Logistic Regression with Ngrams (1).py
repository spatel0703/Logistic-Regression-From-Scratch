#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

diabetesdf = pd.read_csv('combined_dataset.csv', na_filter = False)

#since we only need the 3rd column(Code), we will remove the Date and Time columns
diabetesdf.drop(diabetesdf.columns[[0,1]], axis = 1, inplace=True)
diabetesdf.to_csv('diabetesdf.csv', index=False)
diabetesdf


# In[46]:


x = diabetesdf['Code']
y = diabetesdf['Value']
print(x)
print(y)


# In[47]:


#this is a scatterplot to show the relationship between x and y
plt.scatter(x.head(1000),y.head(1000))
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[133]:


from sklearn.feature_extraction.text import CountVectorizer

#get folder and file path
folder_path = r'C:\\Users\\shrey\\CSC 4850 - Machine Learning\\Homework 1 - KNN and K-Means'

#load the 70 data files
patients_data = []
max_length = 0

for i in range(1, 71):
    filename = f'{folder_path}\\data-{i:02d}' 
    with open(filename, 'r') as file:
        patient_sequence = file.read().strip()
        patients_data.append(patient_sequence)
        max_length = max(max_length, len(patient_sequence))

#define ngram range. The min and max values can be adjusted
ngram_range = (2, 4) 

#initialize countvectorizer
vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)

#this will convert each of the pateients sequences to ngrams
ngram_data = vectorizer.fit_transform(patients_data)

#this will turn the matrix above into a concise array
final_data_matrix = ngram_data.toarray()

print(final_data_matrix)


# In[134]:


#this is to turn the ngrams version into a dataframe
final_dataframe = pd.DataFrame(final_data_matrix)

#this is to rename 0 column of dataframe from arrays into 'Value'
final_dataframe = final_dataframe.rename(columns={0: 'Value'})

final_dataframe


# In[135]:


from sklearn.model_selection import train_test_split

#this splits the training-validation-test sets into a 60-10-30% split
train_data, temp_data = train_test_split(final_dataframe, test_size=0.4, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.75, random_state=42)

#this splits the features
X_train = train_data.drop('Value', axis=1)
X_valid = valid_data.drop('Value', axis=1)
X_test = test_data.drop('Value', axis=1)

#this splits the Value variable since it is the target
y_train = train_data['Value']
y_valid = valid_data['Value']
y_test = test_data['Value']

#prints out shape of sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_valid.shape, y_valid.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# In[136]:


#dataframe for training set
train_data


# In[137]:


valid_data


# In[138]:


test_data


# In[139]:


from tqdm import tqdm

def sigmoid_formula(instance):
    sigmoid_denominator = 1 + np.exp(-instance)
    return 1 / sigmoid_denominator

def log_regress(instance, output, iterations, learning_rate=0.00001):
    #initialize weights with zeros
    m = np.zeros(X_train.shape[1])
    b = 0

    gradients_m = []  #list to store gradient values after each iteration
    gradients_b = []

    progress_bar = tqdm(total=iterations, desc="Training Progress")

    for _ in range(iterations):
        predictions = sigmoid_formula(np.dot(instance, m) + b)
        error = predictions - output
        dm = (1/len(instance)) * np.dot(instance.T, error)
        db = (1/len(instance)) * np.sum(error)

        m -= learning_rate * dm
        b -= learning_rate * db

        gradients_m.append(dm)
        gradients_b.append(db)

        progress_bar.update(1)  #update progress bar
        progress_bar.set_postfix({'m': f'{m}', 'b': f'{b}'})

    progress_bar.close()
    return m, b, gradients_m, gradients_b



def predictor_function(instance, m, b):
    return (m * instance) + b
numeric_mask = pd.to_numeric(y_train, errors='coerce').notna()
X_train = X_train[numeric_mask]
y_train = y_train[numeric_mask]

#convert y_train to a numeric type
y_train = y_train.astype(float)


m, b, gm, gb = log_regress(X_train, y_train, 50)
print('m is:', m)
print('b is:', b)


# In[140]:


newvalidation_predictions = predictor_function(X_valid, m, b)
newvalidation_predictions


# In[141]:


#assuming newvalidation_predictions is your DataFrame
threshold = 1e-6  #you can adjust this threshold based on your data precision

newvalidation_predictions = newvalidation_predictions.applymap(lambda x: np.nan if abs(x - 0.144338) < threshold else x)
newvalidation_predictions


# In[142]:


#assuming new_df is your DataFrame with NaN values
#new_df
stacked_df = newvalidation_predictions.stack()

#create a new DataFrame with one column and drop NaN values
validfinal_df = pd.DataFrame({'Combined_Column': stacked_df.dropna().values})
validfinal_df


# In[143]:


validfinal_array = validfinal_df.values
validfinal_array


# In[144]:


y_testarray = y_test.values
y_testarray


# In[145]:


#applying sigmoid function
validprobabilities = 1 / (1 + np.exp(-validfinal_array))

print(validprobabilities)


# In[146]:


#applying sigmoid function
y_probabilities = 1 / (1 + np.exp(-y_testarray))

print(y_probabilities)


# In[147]:


print("Actual vs Predicted:")
for actual, predicted in zip(y_testarray, validfinal_array):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[148]:


print("Actual Probabilties vs Predicted Probabilities:")
for actualprob, predictedprob in zip(y_probabilities, validprobabilities):
    print(f"Actual Probabilitity: {actualprob}, Predicted Probability: {predictedprob}")


# In[149]:


#this will give the new test set predictions with gradient descent function
newtest_predictions = predictor_function(X_test, m, b)

newtest_predictions


# In[150]:


#assuming newvalidation_predictions is your DataFrame
threshold = 1e-6  

newtest_predictions = newtest_predictions.applymap(lambda x: np.nan if abs(x - 0.144338) < threshold else x)
newtest_predictions


# In[151]:


stacked_df = newtest_predictions.stack()

#create a new DataFrame with one column and drop NaN values
testfinal_df = pd.DataFrame({'Combined_Column': stacked_df.dropna().values})
testfinal_df


# In[152]:


testfinal_array = testfinal_df.values
testfinal_array


# In[153]:


#applying sigmoid function
testprobabilities = 1 / (1 + np.exp(-testfinal_array))

print(testprobabilities)


# In[154]:


print("Actual vs Predicted:")
for actual, predicted, prob in zip(y_testarray, testfinal_array, testprobabilities):
    print(f"Actual: {actual}, Predicted: {predicted}, Probability: {prob}")


# In[155]:


print("Actual Probabilties vs Predicted Probabilities:")
for actualprob, predictedprob in zip(y_probabilities, testprobabilities):
    print(f"Actual: {actualprob}, Predicted: {predictedprob}")


# In[156]:


threshold = 0.5

#apply threshold to classify as positive or negative
predicted_classes = (testprobabilities > threshold).astype(int)
actual_classes = (y_probabilities > threshold).astype(int)


# In[157]:


print("Actual vs Predicted:")
for actual_class, pred_class in zip(actual_classes, predicted_classes):
    print(f"Actual:Actual Class: {actual_class}  , Predicted Class: {pred_class}")


# In[158]:


#accuracy calculations for test predictions
correct = 0
for i in range(min(len(y_testarray), 21)):
    if y_testarray[i] == testfinal_array[i]:
        correct += 1

accuracy = correct / len(y_testarray)
print("Accuracy:", accuracy)


# In[159]:


#precision calculations 
true_positives = 0
predicted_positives = 0  
for i in range(min(len(y_testarray), 21)):
    if testfinal_array[i] == y_testarray[i]: 
        true_positives += 1
    predicted_positives += 1
    
precision = true_positives / predicted_positives
print("Precision:", precision)  


# In[160]:


#recall calculations
true_positives = 0  
actual_positives = 0
for i in range(min(len(y_testarray), 21)):
    if testfinal_array[i] == y_testarray[i]:
        true_positives += 1 
    actual_positives += 1
recall = true_positives / actual_positives  
print("Recall:", recall)


# In[161]:


#f1 score calculations
f1 = 2 * ( (precision * recall) / (precision + recall)) 
print("F1 Score:", f1)


# In[ ]:





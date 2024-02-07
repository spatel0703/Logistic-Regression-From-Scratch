#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

diabetesdf = pd.read_csv('combined_dataset.csv', na_filter = False)

#since we only need the 3rd column(Code), we will remove the Date and Time columns
diabetesdf.drop(diabetesdf.columns[[0,1]], axis = 1, inplace=True)
diabetesdf.to_csv('diabetesdf.csv', index=False)
diabetesdf


# In[2]:


x = diabetesdf['Code']
y = diabetesdf['Value']
print(x)
print(y)


# In[3]:


#this is a scatterplot to show the relationship between x and y
plt.scatter(x.head(1000),y.head(1000))
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[4]:


#this turns the original dataframe into a one-hot encoded one. Value remains same for each row.
#However, code only returns 1 if true for the Code column
diabetes_one_hot_data = pd.get_dummies(diabetesdf, columns = ['Code'])
diabetes_one_hot_data.to_csv('diabetes_one_hot_dataknn.csv', index=False)
diabetes_one_hot_data


# In[5]:


#Here I got rid of the Code_ in the column names as it was causing me issues for some reason
diabetes_one_hot_data.columns = [int(col.split('_')[1]) if isinstance(col, str) and col.startswith('Code_') else col for col in diabetes_one_hot_data.columns]
diabetes_one_hot_data


# In[6]:


from sklearn.model_selection import train_test_split

#this splits the training-validation-test sets into a 60-10-30% split
train_data, temp_data = train_test_split(diabetes_one_hot_data, test_size=0.4, random_state=42)
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


# In[7]:


#dataframe for training set
train_data


# In[8]:


valid_data


# In[9]:


test_data


# In[10]:


from tqdm import tqdm
#tqdm to keep a progress bar on the code running

#this sigmoid functions puts the values between 0 and 1
def sigmoid_formula(instance):
    sigmoid_denominator = 1 + np.exp(-instance)
    return 1 / sigmoid_denominator

#logistic regression function that uses gradient descent
def log_regress(instance, output, iterations, learning_rate=0.0001):
    #initialize weights with zeros
    m = np.zeros(X_train.shape[1])
    b = 0

    gradients_m = []  #list to store gradient values after each iteration
    gradients_b = []
    
    progress_bar = tqdm(total=iterations, desc="Training Progress")

    for _ in range(iterations):
        #after each iteration, the predictions are made using the sigmoid formula function
        predictions = sigmoid_formula(np.dot(instance, m) + b)
        error = predictions - output
        
        #here the gradients are calcuated for the weights and bias
        dm = (1/len(instance)) * np.dot(instance.T, error)
        db = (1/len(instance)) * np.sum(error)

        #the weights and bias are updates by using gradient descent
        m -= learning_rate * dm
        b -= learning_rate * db

        gradients_m.append(dm)
        gradients_b.append(db)

        progress_bar.update(1) 
        progress_bar.set_postfix({'m': f'{m}', 'b': f'{b}'})

    progress_bar.close()
    return m, b, gradients_m, gradients_b


#this function predicts new values using the weights and bias from the logistic regression function
def predictor_function(instance, m, b):
    return (m * instance) + b

numeric_mask = pd.to_numeric(y_train, errors='coerce').notna()
X_train = X_train[numeric_mask]
y_train = y_train[numeric_mask]

y_train = y_train.astype(float)


m, b, gm, gb = log_regress(X_train, y_train, 1000)
print('m is:', m)
print('b is:', b)


# In[173]:


plt.plot(range(1000), gm, label='Slope Gradient') 
plt.plot(range(1000), gb, label='Intercept Gradient')
plt.xlabel("Iterations")
plt.ylabel("Gradient Values") 
plt.title("Gradient Descent Convergence")
plt.legend()
plt.show()


# In[11]:


#making predictions for validation set
newvalidation_predictions = predictor_function(X_valid, m, b)
newvalidation_predictions


# In[12]:


#this removes all the values for the columns where there wasn't a one hot encoded 1
threshold = 1e-6 

newvalidation_predictions = newvalidation_predictions.applymap(lambda x: np.nan if abs(x - 7.8101) < threshold else x)
newvalidation_predictions


# In[13]:


#this will get rid of unnecessary columns
stacked_df = newvalidation_predictions.stack()

#creates a new DataFrame with one column and drop NaN values
validfinal_df = pd.DataFrame({'Combined_Column': stacked_df.dropna().values})
validfinal_df


# In[14]:


#just turned the df into an array
validfinal_array = validfinal_df.values
validfinal_array


# In[15]:


y_testarray = y_test.values
y_testarray


# In[16]:


#applying sigmoid function to find probabilities of validation predictions
probabilities = 1 / (1 + np.exp(-validfinal_array))

print(probabilities)


# In[17]:


#filter out non-numeric values
y_testarray_numeric = [float(value) for value in y_testarray if value.replace('.', '', 1).isdigit()]

#convert to NumPy array
y_testarray_numeric = np.array(y_testarray_numeric)

#calculate probabilities using the sigmoid function
y_probabilities = 1 / (1 + np.exp(-y_testarray_numeric))

print(y_probabilities)


# In[18]:


print("Actual vs Predicted vs prob:")
for actual, predicted in zip(y_testarray, validfinal_array):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[19]:


print("Actual Probabilties vs Predicted Probabilities:")
for actualprob, predictedprob in zip(y_probabilities, probabilities):
    print(f"Actual: {actualprob}, Predicted: {predictedprob}")


# In[20]:


#this will give the new test set predictions with gradient descent function
newtest_predictions = predictor_function(X_test, m, b)

newtest_predictions


# In[21]:


#this removes all the values for the columns where there wasn't a one hot encoded 1
threshold = 1e-6 

newtest_predictions = newtest_predictions.applymap(lambda x: np.nan if abs(x - 7.8101) < threshold else x)
newtest_predictions


# In[22]:


stacked_df = newtest_predictions.stack()

#creates a new DataFrame with one column and drop NaN values
testfinal_df = pd.DataFrame({'Combined_Column': stacked_df.dropna().values})
testfinal_df


# In[23]:


testfinal_array = testfinal_df.values
testfinal_array


# In[24]:


#applying sigmoid function to find probabilities of test predictions
testprobabilities = 1 / (1 + np.exp(-testfinal_array))

print(testprobabilities)


# In[25]:


threshold = 0.5

#apply threshold to classify as positive or negative
predicted_classes = (testprobabilities > threshold).astype(int)
actual_classes = (y_probabilities > threshold).astype(int)


# In[26]:


print("Actual vs Predicted Values:")
for actual, predicted in zip(y_testarray, testfinal_array):
    print(f"Actual: {actual}, Predicted: {predicted}")


# In[27]:


print("Actual Probabilties vs Predicted Probabilities:")
for actualprob, predictedprob in zip(y_probabilities, testprobabilities):
    print(f"Actual: {actualprob}, Predicted: {predictedprob}")


# In[28]:


print("Actual vs Predicted:")
for actual_class, pred_class in zip(actual_classes, predicted_classes):
    print(f"Actual:Actual Class: {actual_class}  , Predicted Class: {pred_class}")


# In[192]:


#plots original data points against regression line of the iterated test set predictions
plt.scatter(x.head(1000), y.head(1000), color='blue', label='Original Data points')
plt.plot(X_test.head(8787), testprobabilities, color='red', label='logistic line')
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
plt.title('Logistic Regression Model')
plt.legend()
plt.show()


# In[29]:


#accuracy calculations for test predictions
correct = 0
for i in range(min(len(y_testarray), 8787)):
    if y_testarray[i] == testfinal_array[i]:
        correct += 1

accuracy = correct / len(y_testarray)
print(correct)
print("Accuracy:", accuracy)


# In[30]:


#precision calculations 
true_positives = 0
predicted_positives = 0  
for i in range(min(len(y_testarray), 8787)):
    if testfinal_array[i] == y_testarray[i]: 
        true_positives += 1
    predicted_positives += 1
    
precision = true_positives / predicted_positives
print(true_positives)
print(predicted_positives)
print("Precision:", precision)  


# In[31]:


#recall calculations
true_positives = 0  
actual_positives = 0
for i in range(min(len(y_testarray), 8787)):
    if testfinal_array[i] == y_testarray[i]:
        true_positives += 1 
    actual_positives += 1
print(true_positives)
print(actual_positives)
recall = true_positives / actual_positives  
print("Recall:", recall)


# In[32]:


#f1 score calculations
f1 = 2 * ( (precision * recall) / (precision + recall)) 
print("F1 Score:", f1)


# In[39]:


# Convert arrays to numpy  
y_true = np.array(y_testarray)
y_pred = np.array(testfinal_array)

# Define loss 
def logistic_loss(y_true, y_pred):
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return loss

loss = logistic_loss(y_true, y_pred) 
print(loss)


# In[ ]:





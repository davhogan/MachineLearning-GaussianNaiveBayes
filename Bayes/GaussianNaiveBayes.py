#Authored By: David Hogan
#Email: dvahogan@pdsx.edu
#The purpose of this program is to use a guassian naive bayes classifier to evaluate if an email is spam or not
#The data used for this program is based on the dataset provided by UCI ML repository:
#https://archive.ics.uci.edu/ml/datasets/spambase

import pandas as pd
import numpy as np

#This function trains the data
#The train function takes in a dataframe
#The columns of the dataframe are then grouped into spam or not
#The mean and std of each column for each group is then calculted and stored
#They are stored in arrays where the ith value of the array represents the jth column of the dataframe
#The arrays of positive and negative means and stds are then returned
def train(df):
    #Arrays used to represent the mean or std for the xi value for positive and negative classification.
    cols = df.columns
    num_cols = cols.__len__()
    pos_means = [0.0]*(num_cols - 1)
    neg_means = [0.0]*(num_cols - 1)
    pos_std = [0.0]*(num_cols - 1)
    neg_std = [0.0]*(num_cols - 1)

    #Group the columns by if they are spam or not
    groups = df.groupby(['spam_or_not'])
    pos_group = groups.get_group(1)
    neg_group = groups.get_group(0)

    i = 0
    #Go throught the data frame by column
    #Filling the means and stds arrays for the classifiers
    #The ith value in the classifier array represent the corresonding jth column in the dataframe
    for col in cols:
        if col != 'spam_or_not':
            pos_means[i] = pos_group[col].agg(np.mean)
            neg_means[i] = neg_group[col].agg(np.mean)
            pos_std[i] = pos_group[col].agg(np.std)
            neg_std[i] = neg_group[col].agg(np.std)
            i += 1

    return (pos_means,neg_means,pos_std,neg_std)

#Represents the normal distrubution.
#Takes in the x value to be classified
#It also takes in the mean and standard deviation for that element's classification.
#Returns the probability of the classifier for the x value
def normalize(x, mean, std):
    if std <= 0:
        std = 0.001

    exp_pow = -((x-mean)*(x-mean))/(2*std*std)
    prob = np.exp(exp_pow)/(np.sqrt(2*np.pi*std))

    #Needed because the log of the result is taken.
    if prob <= 0:
        prob = 0.0001

    return prob

#Tests the naive bayes classifier
#Takes in the dataframe to be tested
#Also takes in arrays representing the means and stds of the classifiers
#The function goes through each row calculating the probablility for the classifier for each element in the row
#The function uses the mean and std for the column of the element being represented
#The probability is calculated by using the normal function to evaluate the probability
#Then a prediction of spam or not (1 or 0) is made based on the sum of the probabilities for each classifier
#The predicted value is compared to the actual value, if they are correct the number correct is incremented
#The confusion matrix is then updated based on the expected and actual results.
#The function returns the confusion matrix, accuracy, precision and recall
def test(df,pos_means,neg_means,pos_std,neg_std):
    #Convert the dataframe to a matrix for easier traversal
    test_matrix = df.values
    num_cols = df.columns.__len__()
    num_rows = df.__len__()
    prob_pos = 0
    prob_neg = 0
    correct = 0
    confusion_matrix = np.zeros(shape=(2,2))

    for i in range(0,num_rows-1):
        for j in range(0,num_cols-2):
            x = test_matrix[i][j]
            prob_pos = prob_pos + np.log(normalize(x,pos_means[j],pos_std[j]))
            prob_neg = prob_neg + np.log(normalize(x,neg_means[j],neg_std[j]))

        if prob_pos > prob_neg:
            expected = 1.0
        else:
            expected = 0.0

        actual = test_matrix[i][57]

        if actual == expected:
            correct += 1

        confusion_matrix[int(actual)][int(expected)] += 1

    #True Positive,False Positive, False Negative, True Negative
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return (confusion_matrix,accuracy,precision,recall)

#Read in the data
df = pd.read_csv("spambase.csv")

#Create training and testing dataframes
split = np.random.rand(len(df)) < 0.5
df_train = df[split]
df_test = df[~split]

(pos_means, neg_means, pos_stds, neg_stds) = train(df_train) #Train Data
(confusion_matrix,accuracy,precision,recall) = test(df_test,pos_means,neg_means, pos_stds, neg_stds) #Test Data
# Display Results
print("Confusion Matrix:")
print(confusion_matrix)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

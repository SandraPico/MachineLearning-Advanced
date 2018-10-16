import numpy as np
import math
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split
)
from sklearn import datasets

#-------------------------Load data--------------------------#

def ExtractData(dataset_name):
    
    #Datasets for Regression
    if(dataset_name == 'boston'):
        features_train,target_train,features_test,target_test = BostonDataset()
    elif(dataset_name == 'NoisyG'):
        features_train,target_train,features_test,target_test = NoisyDataset('g')
    elif(dataset_name == 'NoisyU'):
        features_train,target_train,features_test,target_test = NoisyDataset('u')
    elif(dataset_name == 'Friedman1'):
        features_train,target_train,features_test,target_test = FriedmanDataset_1()
    elif(dataset_name == 'Friedman2'):
        features_train,target_train,features_test,target_test = FriedmanDataset_2()
    elif(dataset_name == 'Friedman3'):
        features_train,target_train,features_test,target_test = FriedmanDataset_3()

    return features_train,target_train,features_test,target_test


"""
    Create the Friedman functions that is named in the paper.
"""
def FriedmanDataset_1():
    d_train = datasets.make_friedman1(240, 10, 1)
    d_test = datasets.make_friedman1(1000, 10)
        
    features_train = d_train[0] + np.random.normal(0, 1, 240).reshape((240,1))
    target_train = d_train[1]
    features_test = d_test[0]
    target_test = d_test[1]

    return features_train, target_train, features_test, target_test

def FriedmanDataset_2():
    d_train = datasets.make_friedman2(240, random_state=0)
    d_test = datasets.make_friedman2(1000, random_state=0)
    
    features_train = d_train[0]
    for i in range(240):
        features_train[i] += np.random.normal(0, features_train[i]/3)

    target_train = d_train[1]
    features_test = d_test[0]
    target_test = d_test[1]

    return features_train, target_train, features_test, target_test


def FriedmanDataset_3():
    d_train = datasets.make_friedman3(240, random_state=0)
    d_test = datasets.make_friedman3(1000, random_state=0)

    features_train = d_train[0]
    for i in range(240):
        features_train[i] += np.random.normal(0, features_train[i]/3)

    target_train = d_train[1]
    features_test = d_test[0]
    target_test = d_test[1]
    return features_train, target_train, features_test, target_test


"""
    Create the function specification in the paper.
"""
def NoisyDataset(noise):

    features_train = np.linspace(-10, 10, 100)
    target_train = np.sinc(features_train)
    if(noise =='g'):
        target_train +=  np.random.normal(0, 0.1, 100)
    elif (noise == 'u'):
        target_train +=  np.random.uniform(-0.1, 0.1, 100)
    
    
    features_test = np.linspace(-10, 10, 1000)
    target_test = np.reshape(np.sinc(features_test), (1000))
    
    return features_train, target_train, features_test,  target_test

"""
    To analyze all the data from the Boston dataset, we should:
    
        1. Analize and extract the data from the dataset.
        2. Divide the data into: training set and the test set. Given the requierements from the paper:
            486/506 examples randomly chosen for the training
            25/506 examples randomly chosen for the test
        3. Above the 14 features, the first 13 will be the features and the 14th will be the target.
        
"""
def BostonDataset():

    #Analyze all Boston's file data.
    file = open("datasets/Boston.txt","r")
    info = []
    counter = 0
    new_Data = 1
    for line in file:
        i = 0
        counter = counter + 1
        aux =  line
        aux2 = []
        #Analyzing the file
        if (new_Data == 1):
            data_line = []
            new_Data = 0
        y = 0
        while(aux[i] != '\n'):
            #We have a number there..
            if(aux[i]!= ' '):
                y = y - i
                aux2.append(aux[i])
                i = i + 1
            else:
                if (aux2):
                    data_line.append(aux2)
                aux2 = []
                i = i + 1
        data_line.append(aux2)
        #If it is an even line..
        if (counter %2 == 0):
            data = np.zeros(len(data_line))
            for i in range(0,len(data_line)):
                data[i] = float(''.join(map(str,data_line[i])))
            info.append(data)
            new_Data = 1
    #Shuffle the whole dataset.
    np.random.shuffle(info)
    #Divide into test and training sets.
    features_train = np.zeros((486,13))
    target_train = np.zeros((486,1))
    features_test =  np.zeros((25,13))
    target_test = np.zeros((25,1))

    #Extract features and targets for the training.
    for i in range(0,486):
        for j in range(0,14):
            if (j != 13):
                features_train[i][j] = float(info[i][j])
            else:
                target_train[i][j-13] = float(info[i][j])

    #Extract features and targets for the test.
    for i in range(486,506):
        for j in range(0,14):
            if( j != 13):
                features_test[i-486][j] = float(info[i][j])
            else:
                target_test[i-486][j-13] = float(info[i][j])

    return features_train,target_train.reshape(486),features_test,target_test.reshape(25)

#---------------------RVM Regression------------------#

## PARAMETERS
alphaThreshold = 1e9

#optimize hyperparameters
def rvm_regression_train(X, y, alphaThreshold, g_value):

    N_samples = len(y)

    phi = np.zeros((N_samples, N_samples+1))
    phi[:,1:] = rbf_kernel(X, X, gamma=g_value)
    phi[:,0] = 1 #first column is bias

    alpha = np.ones((len(X) +1, 1))
    A = np.diagflat(alpha)
    index =  np.ones(len(X)+1, dtype=bool)
    mu = np.zeros(len(X) + 1)
    sigma = 50

    for t in range(1000): 
        index = rvm_prune(alpha, alphaThreshold)
        S = np.linalg.inv(A[index][:,index] + sigma * np.dot(phi[:,index].T, phi[:,index]))
        mu[index] = sigma * np.dot(S, np.dot(phi[:,index].T, y))
        gamma = 1 - (alpha[index]).T * np.diag(S)
        alpha[index] = ( gamma / np.array(mu[index]**2) ).T
        sigma = ( N_samples - np.sum(gamma) ) / np.linalg.norm( y - np.dot(phi[:,index], mu[index]) )
        A = np.diagflat(alpha)
    
    X_rel = X[index[1::]] #bias is not taken into consideration
    y_rel = y[index[1::]]
    
    #print('Number of mu:',mu[index].size,' Number of relevant vectors:',X_rel.size)
    
    return mu[index], sigma, alpha[index], X_rel, y_rel

def rvm_prune(alpha, alphaThreshold):
    pruned_index = np.squeeze(np.abs(alpha) < alphaThreshold)
    return pruned_index


def rvm_regression_predict(X_rel, X_new, mu):
    
    if(mu.size == X_rel.shape[0]+1):  #mu corresponding to bias is kept! 
        phi = np.zeros((X_rel.shape[0]+1, X_new.shape[0])) #add one dimension full of ones
        phi[1:,:] = rbf_kernel(X_rel, X_new)
        phi[0,:] = 1 #first column is bias
    else:
        phi = np.zeros((X_rel.shape[0], X_new.shape[0]))
        phi = rbf_kernel(X_rel, X_new)
   
    return np.dot(phi.T, mu)


def rmse(y_est, y_true):
    return math.sqrt(mean_squared_error(y_est, y_true))

  #-----------------Test Regression--------------------#

#Concatenate all the vectors in order to achieve a good train/test classification.
def ConcatenateCross(data,index):
    
    new_vector = []
    for i in range(0,len(data)):
        if (i != index):
            new_vector.append(data[i])
    ab_vector = np.concatenate((new_vector[0],new_vector[1]))
    abc_vector = np.concatenate((ab_vector,new_vector[2]))
    abcd_vector = np.concatenate((abc_vector,new_vector[3]))

    return abcd_vector

regression_datasets = ['NoisyU', 'NoisyG','Friedman2', 'Friedman3', 'boston'] 
svm_result = np.zeros((len(regression_datasets), 6)) #(rmse, number of support vectors)
test_split = [0.91,0.91,0.806,0.806,0.806,0.05 ] #specify splitting percentages for testing set for each dataset

for i in range(len(regression_datasets)):
    #load dataset
    data = ExtractData(regression_datasets[i])
    features_all = np.concatenate((data[0], data[2]), axis=0) #concatenate all features
    targets_all = np.concatenate((data[1], data[3]), axis=0)  #concatenate all targets
   

    repetitions_result = np.zeros((10,6)) # store (accuracy, number of support vectors) for every iteration
    for r in range(10): #repeat for different train-test datasets
    
        #X_train: training features, y_train: training labels
        X_train, X_test, y_train, y_test = train_test_split(features_all, targets_all, test_size=test_split[i], random_state=0)
        if(len(X_train.shape) == 1): #only one feature, has to be 2 dimensional
            X_train = np.reshape(X_train, (X_train.shape[0], 1))
       
        #SVM Regression - training
        g = range(1,10) #specify different parameters for the Gaussian kernel to test

        #Cross validation to define the kernel parameter 
        CrossV_scores = np.zeros(len(g))
        for j in range(len(g)):
            clf = svm.SVR(C=1, kernel='rbf', gamma=g[j]) 
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error') #(training_data, target values)
            CrossV_scores[j] = np.mean(np.absolute(scores))
        
        index_min = np.argmin(CrossV_scores) #find gamma value with the minimum score
        best_gamma = g[index_min]
         
        #SVM Regression - testing
        best_clf = svm.SVR(C=1, kernel='rbf', gamma = best_gamma)
        best_clf.fit(X_train, y_train) #create svm model for training data
        
        if(len(X_test.shape) == 1): #only one feature, has to be 2 dimensional
            X_test = np.reshape(X_test, (X_test.shape[0], 1))
       
        res = best_clf.predict(X_test)
        repetitions_result[r,0] = rmse(y_test, res)
        repetitions_result[r,1] = len(best_clf.support_)
        repetitions_result[r,4] = best_gamma
        
        
        #RVM Cross-Validation
        #Create a numpy array in order to compute the rvm.
        X_train_array = np.array(X_train)
        X_test_array = np.array(X_test)
        y_train_array = np.array(y_train)
        y_test_array = np.array(y_test)

        #To use for Cross-validation in order to find the proper parameter for the gaussian kernel.
        X_crossV = np.array_split(X_train_array,5)
        y_crossV = np.array_split(y_train_array,5)
        
        rvm_scores = np.zeros(len(g)) #keep 5CV score for each kernel parameter
        #Errors will store the error value for each possible value of the kernel parameter.
        errors = []
        #rv will store the relevant vectors values for each value of the kernel parameter.
        rv = []
        
        for j in range(0,len(g)):
            error_number = 0
            rv_number = 0
            #print("We are going to compute RVM with the dataset: " + str(regression_datasets[i])+str(r)+ " using the gamma parameter: " + str(g[j]))
            for y in range(0,5):
                #Now the test will be:
                test_index = y
                X_test_crossV = X_crossV[y]
                y_test_crossV = y_crossV[y]
                
                #Define the training sets for the cross validation concatanting.
                X_train_crossV = ConcatenateCross(X_crossV,test_index)
                y_train_crossV = ConcatenateCross(y_crossV,test_index)
                
                #find relevant vectors for the specific fold
                mu, sigma, alpha, X_rel, y_rel = rvm_regression_train(X_train_crossV,y_train_crossV,alphaThreshold,g[j]) #RVM Regression!
                #use relevant vectors to predict on the test set for the specific fold
                y_pred = rvm_regression_predict(X_rel, X_test_crossV, mu)
                
                #compute the rmse for the specific fold
                error_number += rmse(y_pred, y_test_crossV)
                rv_number = rv_number + X_rel.shape[0]
        
            #Compute mean error and number of relevant vectors after 5CV
            errors.append((error_number/5))
            rv.append((rv_number/5))
            
            #minimum number for the g.
            #print("The total errors for each gamma and dataset: "+str(regression_datasets[i])+str(r)+ " is: ")
            #print(errors)
            rvm_index_min = np.argmin(errors)
            rvm_best_gamma = g[rvm_index_min]
            #print("The best gamma is: ",rvm_best_gamma)
            
            #RVM Classification (Testing)
            #Build the model
            #find relevant vectors for the specific fold
            mu_final, sigma_final, alpha_final, X_rel_final, y_rel_final = rvm_regression_train(X_train_array, y_train_array,alphaThreshold,rvm_best_gamma) #RVM Regression!
            #use relevant vectors to predict 
            y_pred_test = rvm_regression_predict(X_rel_final, X_test_array, mu_final)
            error_final =  rmse(y_pred_test, y_test_array)
            rv_final = X_rel_final.shape[0]
            print("With the gamma: " + str(rvm_best_gamma) + "and the dataset: " +str(regression_datasets[i])+str(r)+ "we achieve an error that is: " + str(error_final) + "with a number of relevance vectors like: " + str(rv_final))
            #Error for the testing.
            repetitions_result[r,2] = error_final  #rmse
            repetitions_result[r,3] = rv_final     #number of relevant vectors
            repetitions_result[r,5] = rvm_best_gamma
            
    #Compute mean (mse, number of support vectors) for dataset i
    svm_result[i,0] = round(np.mean(repetitions_result[:,0]), 3)     #error SVM
    #svm_result[i,0] = svm_result[i,0]*100                            #error SVM (%)
    svm_result[i,1] = np.mean(repetitions_result[:,1])               #support vectors for SVM
    svm_result[i,2] = round(np.mean(repetitions_result[:,2]),3)      #error for RVM
    #svm_result[i,2] = svm_result[i,2]*100                             #error RVM(%)
    svm_result[i,3] = np.mean(repetitions_result[:,3])               #relevance vectors for RVM
    svm_result[i,4] = np.mean(repetitions_result[:,4])               #SVM gamma
    svm_result[i,5] = np.mean(repetitions_result[:,5])               #RVM gamma

print("The results: ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print("Dataset:"," ", "SVM: Mean error(%)", " ", "SVM : Number of SupVec", " " ,"RVM: Mean error(%)", " "," RVM: Number of Relevant Vectors"," ","SVM gamma", " ", "RVM gamma")
#Print the result for the dataset.
print(np.column_stack((np.array(regression_datasets),svm_result)))
print("######################################################################")


      


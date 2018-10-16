import numpy as np
import math
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    cross_val_score,
    train_test_split
)

def DrawError(gammas,errors):
    
    #First we need to plot the data points.
    plt.figure(figsize=(7,6))
    plt.title("Gamma parameter analysis for the Breast-Cancer dataset.")
    plt.plot(gammas,errors,'r')
    plt.ylabel('Error rate')
    plt.xlabel('Gamma parameter')
    plt.show()

#--------Load Datasets--------------#

def ExtractData(dataset_name, i=-1):
    
    #Datasets for Classification
    if (dataset_name == 'pima'):
        features_train,target_train,features_test,target_test = PimaDataset()
    elif(dataset_name == 'breastcancer'):
        features_train,target_train,features_test,target_test = BreastCancerDataset(i)
    elif(dataset_name == 'titanic'):
        features_train,target_train,features_test,target_test = TitanicDataset(i)
    elif(dataset_name == 'waveform'):
        features_train,target_train,features_test,target_test = WaveformDataset(i)
    elif(dataset_name == 'german'):
        features_train,target_train,features_test,target_test = German(i)
    elif(dataset_name == 'image'):
        features_train,target_train,features_test,target_test = ImageDataset(i)
    elif(dataset_name == 'banana'):
        features_train,target_train,features_test,target_test = Banana(i)

    return (features_train, target_train, features_test, target_test)

#We already have the dataset divided into train and test.
def PimaDataset():
    train = []
    test = []
    train_file = open('datasets/pima/pima_train.csv')
    test_file = open('datasets/pima/pima_test.csv')

    train_file.readline()
    for i in train_file.readlines():
        train.append([i for i in i.split(';')])
    train = np.array(train)
    cols = len(train[0])
    features_train = (train[:, 0:cols - 1]).astype('float')
    target_train = [(0, 1)[i[0][0] == 'Y'] for i in train[:, cols - 1]]

    test_file.readline()
    for i in test_file.readlines():
        test.append([i for i in i.split(';')])
    test = np.array(test)
    cols_test = len(test[0])
    features_test = (test[:, 0:cols_test - 1]).astype('float')
    target_test = [(0, 1)[i[0][0] == "Y"] for i in test[:, cols_test - 1]]

    train_file.close()
    test_file.close()
    return features_train, target_train, features_test, target_test

#As in paper we use the ten first datasets. To import the asc files we use np.loadtxt
def BreastCancerDataset(i): #specify file id
    
    features_train = []
    target_train = []
    features_test = []
    target_test = []
    
    x = np.loadtxt("datasets/breast-cancer/breast-cancer_train_data_" + str(i+1) + ".asc")
    features_train += x.tolist()
    x_test = np.loadtxt("datasets/breast-cancer/breast-cancer_test_data_" + str(i+1) + ".asc")
    features_test += x_test.tolist()
    
    t = np.loadtxt("datasets/breast-cancer/breast-cancer_train_labels_" + str(i+1) + ".asc")
    target_train = [(0, 1)[i == 1] for i in t]
    
    t_test = np.loadtxt("datasets/breast-cancer/breast-cancer_test_labels_" + str(i+1) + ".asc")
    target_test = [(0, 1)[i == 1] for i in t_test]
    
    return features_train, target_train, features_test, target_test

def TitanicDataset(i):

    features_train = []
    target_train = []
    features_test = []
    target_test = []

    x = np.loadtxt("datasets/titanic/titanic_train_data_" + str(i+1) + ".asc")
    features_train += x.tolist()
    x_test = np.loadtxt("datasets/titanic/titanic_test_data_" + str(i+1) + ".asc")
    features_test += x_test.tolist()
    
    t = np.loadtxt("datasets/titanic/titanic_train_labels_" + str(i+1) + ".asc")
    target_train = [(0, 1)[i == 1] for i in t]

    t_test = np.loadtxt("datasets/titanic/titanic_test_labels_" + str(i+1) + ".asc")
    target_test = [(0, 1)[i == 1] for i in t_test]
    
    return features_train, target_train, features_test, target_test

def WaveformDataset(i):
    
    features_train = []
    target_train = []
    features_test = []
    target_test = []
    
    x = np.loadtxt("datasets/waveform/waveform_train_data_" + str(i+1) + ".asc")
    features_train += x.tolist()
    x_test = np.loadtxt("datasets/waveform/waveform_test_data_" + str(i+1) + ".asc")
    features_test += x_test.tolist()
    
    t = np.loadtxt("datasets/waveform/waveform_train_labels_" + str(i+1) + ".asc")
    target_train = [(0, 1)[i == 1] for i in t]
    
    t_test = np.loadtxt("datasets/waveform/waveform_test_labels_" + str(i+1) + ".asc")
    target_test = [(0, 1)[i == 1] for i in t_test]

    return features_train, target_train, features_test, target_test

def German(i):
    
    features_train = []
    target_train = []
    features_test = []
    target_test = []
    
    x = np.loadtxt("datasets/german/german_train_data_" + str(i+1) + ".asc")
    features_train += x.tolist()
    x_test = np.loadtxt("datasets/german/german_test_data_" + str(i+1) + ".asc")
    features_test += x_test.tolist()
    
    t = np.loadtxt("datasets/german/german_train_labels_" + str(i+1) + ".asc")
    target_train = [(0, 1)[i == 1] for i in t]
    
    t_test = np.loadtxt("datasets/german/german_test_labels_" + str(i+1) + ".asc")
    target_test = [(0, 1)[i == 1] for i in t_test]
    
    return features_train, target_train, features_test, target_test

def Banana(i):
    
    features_train = []
    target_train = []
    features_test = []
    target_test = []
    
    x = np.loadtxt("datasets/banana/banana_train_data_" + str(i+1) + ".asc")
    features_train += x.tolist()
    x_test = np.loadtxt("datasets/banana/banana_test_data_" + str(i+1) + ".asc")
    features_test += x_test.tolist()
    
    t = np.loadtxt("datasets/banana/banana_train_labels_" + str(i+1) + ".asc")
    target_train = [(0, 1)[i == 1] for i in t]
    
    t_test = np.loadtxt("datasets/banana/banana_test_labels_" + str(i+1) + ".asc")
    target_test = [(0, 1)[i == 1] for i in t_test]
    
    return features_train, target_train, features_test, target_test

def ImageDataset(i):
    
    features_train = []
    target_train = []
    features_test = []
    target_test = []
    
    x = np.loadtxt("datasets/image/image_train_data_" + str(i+1) + ".asc")
    features_train += x.tolist()
    x_test = np.loadtxt("datasets/image/image_test_data_" + str(i+1) + ".asc")
    features_test += x_test.tolist()
    
    t = np.loadtxt("datasets/image/image_train_labels_" + str(i+1) + ".asc")
    target_train = [(0, 1)[i == 1] for i in t]
    
    t_test = np.loadtxt("datasets/image/image_test_labels_" + str(i+1) + ".asc")
    target_test = [(0, 1)[i == 1] for i in t_test]
    
    return features_train, target_train, features_test, target_test



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


#Compute the phi matrix
def buildPhi(r, x1, x2):
    
    N = len(x1)
    K = len(x2)
    Phi = np.zeros((N, K))
    for m in range(0, N):
        for n in range(0, K):
            Phi[m, n] = gaussianKernel(r, x1[m], x2[n])
    return (Phi)

#Execute the gaussian kernel for each of the phi elements
def gaussianKernel(r, x_m, x_n):
    return (math.exp(-(r**(-2) * np.linalg.norm(x_m-x_n)**2)))

#Define the sigmoid function to compute the likelihood in the Classification.
def sigma(y):
    return (1 / (1 + math.exp(-y)))

#RVM function for Classification
def rvm_Classification(x, t, r):
    
    LAMBDA_MIN = 2 ** (-10)
    GRAD_STOP = 10 ** (-6)
    ALPHA_THRESHOLD = 10 ** 9
    N = len(x)
    
    w = np.zeros(N + 1)
    alpha = N ** (-2) * np.ones(N+1)
    sigmoid = np.vectorize(sigma)
    
    Complete_Phi = np.ones((N, N + 1))
    Complete_Phi[:, 1:] = buildPhi(r, x, x)
    
    indices = [i for i in range(0, N + 1)]
    K = len(indices)
    
    iteration_count = 0
    
    while (iteration_count <= 500):
        
        alpha_used = alpha[indices]
        w_used = w[indices]
        Phi = Complete_Phi[:, indices]
        
        A = alpha_used * np.identity(K)
        Phiw = np.dot(Phi, w_used)
        Phiw[Phiw < -500] = -500
        Y = sigmoid(Phiw)
        Y[Y == 1.0] = 1.0 - (1e-16)
        zero_class_log = [math.log(Y[k]) for k in range(0, N) if t[k] == 1]
        one_class_log = [math.log(1 - Y[k]) for k in range(0, N) if t[k] == 0]
        data_term = -(sum(zero_class_log) + sum(one_class_log))
        regulariser = 0.5 * np.dot(np.dot(w_used, A), w_used)
        err_old = data_term + regulariser
        
        for i in range(0, 25):
            
            Phiw = np.dot(Phi, w_used)
            Phiw[Phiw < -500] = -500
            Y = sigmoid(Phiw)
            Y[Y == 1.0] = 1.0 - (1e-16)
            beta = [Y[n] * (1 - Y[n]) for n in range(0, N)]
            B = beta * np.identity((N))
            Hessian = np.dot(np.dot(np.transpose(Phi), B), Phi) + A
            Sigma = np.linalg.inv(Hessian)
            
            e = np.subtract(t, Y)
            g = np.dot(np.transpose(Phi), e) - np.array(alpha_used) * np.array(w_used)
            
            delta_w = np.dot(Sigma, g)
            
            lamb = 1
            while lamb > LAMBDA_MIN:
                w_new = w_used + lamb * np.array(delta_w)
                Phiw = np.dot(Phi, w_new)
                Phiw[Phiw < -500] = -500
                Y = sigmoid(Phiw)
                Y[Y == 1.0] = 1.0 - (1e-16)
                zero_class_log = [math.log(Y[k]) for k in range(0, N) if t[k] == 1]
                one_class_log = [math.log(1 - Y[k]) for k in range(0, N) if t[k] == 0]
                data_term = -(sum(zero_class_log) + sum(one_class_log)) / N
                w_squared = [k ** 2 for k in w_new]
                regulariser = np.dot(alpha_used, w_squared) / (2 * N)
                err_new = data_term + regulariser
                
                if err_new > err_old:
                    lamb = lamb / 2
                else:
                    break
            w_used = w_new
            err_old = err_new
            if np.linalg.norm(g) / K < GRAD_STOP:
                break
    
        w[indices] = w_used
        
        gamma = [(1 - alpha_used[i] * Sigma[i, i]) for i in range(0, K)]
        old_alpha = list(alpha)
        alpha[indices] = np.array([gamma[i] * (w_used[i] ** (-2)) for i in range(0, K)])
        
        indices = [k for k in range(0, N + 1) if alpha[k] < ALPHA_THRESHOLD]
        K = len(indices)
        
        not_used_indices = list(set(range(0, N)) - set(indices))
        w[not_used_indices] = 0
        
        #if (iteration_count % 50 == 0):
        #    print("Status: Iteration: " + str(iteration_count) + " Useful indices: " + str(K))
        
        iteration_count = iteration_count + 1
    
    #print("Optimization for this training set has finished.")
    w_used = w[indices]
    indices = np.array(indices) - 1
    
    return (w_used, indices)


#---------SVM and RVM Classification tests -------------#

classification_datasets = ['pima', 'banana', 'breastcancer', 'titanic', 'waveform', 'german', 'image']
svm_result = np.zeros((len(classification_datasets), 6)) #(classification error, number of support vectors)

for i in range(len(classification_datasets)):
    
    #load dataset
    if(classification_datasets[i] == 'pima'): #only one set of training, test files
        data = ExtractData(classification_datasets[i])  #data[0]=train, data[1]=test 
        features_all = np.concatenate((data[0], data[2]), axis=0) #concatenate all features
        targets_all = np.concatenate((data[1], data[3]), axis=0)  #concatenate all targets
   
    repetitions_result = np.zeros((10,6)) # store (accuracy, number of support vectors) for every iteration

    for r in range(10): #repeat for different train-test datasets
        
        #read data from different files in each iteration
        if(classification_datasets[i] != 'pima'):
            if (r != 3):
                data = ExtractData(classification_datasets[i], r)  #data[0]=train, data[1]=test
                X_train = data[0]
                X_test = data[2]
                y_train = data[1]
                y_test = data[3]
            else:
                data = ExtractData(classification_datasets[i], 4)  #data[0]=train, data[1]=test
                X_train = data[0]
                X_test = data[2]
                y_train = data[1]
                y_test = data[3]
        else:
            #split randomly the training and testing file -  Prima dataset
            #X_train: training features, y_train: training labels
            X_train, X_test, y_train, y_test = train_test_split(features_all, targets_all, test_size=0.624, random_state=0)
        
        
        #SVM Classification - training
        g = np.zeros(10) #Store different values for the gamma parameter.
        if (classification_datasets[i] != 'pima'):
            value = 0
        else:
            value = 120
        for u in range(0,10):
            value = value + 0.5
            g[u] = value
        #Now g is an array with different values that we want to test.

        #Cross validation to define the kernel parameter 
        CrossV_scores = np.zeros(len(g))
        
        #SVM Cross-Validation
        for j in range(len(g)):
            clf = svm.SVC(C=1, kernel='rbf', gamma=g[j])
            #Scores will be an array that store the accuracy for each of the splits.
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') #(training_data, target values)
            CrossV_scores[j] = np.mean(np.absolute(scores))

        index_min = np.argmax(CrossV_scores) #find gamma value with the maximum score.
        best_gamma = g[index_min]
    
        #SVM Classification - testing
        best_clf = svm.SVC(C=1, kernel='rbf', gamma = best_gamma)
        best_clf.fit(X_train, y_train) #create svm model for training data
        res = best_clf.predict(X_test)
        repetitions_result[r,0] = accuracy_score(y_test, res, normalize='True')
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

        rvm_scores = np.zeros(len(g))

        #Errors will store the error value for each possible value of the kernel parameter.
        errors = []
        #rv will store the relevant vectors values for each value of the kernel parameter.
        rv = []
        for j in range(0,len(g)):
            error_number = 0
            rv_number = 0
            print("We are going to compute RVM with the dataset: " + str(classification_datasets[i])+str(r)+ " using the gamma parameter: " + str(g[j]))
            for y in range(0,5):
                #Now the test will be:
                test_index = y
                X_test_crossV = X_crossV[y]
                y_test_crossV = y_crossV[y]
                
                
                #Define the training sets for the cross validation concatanting.
                X_train_crossV = ConcatenateCross(X_crossV,test_index)
                y_train_crossV = ConcatenateCross(y_crossV,test_index)
                
                #Now we have to compute the RVM.
                y_train_crossV_list = []
                for val in range(0,len(y_train_crossV)):
                    y_train_crossV_list.append(y_train_crossV[val])
                
                weights, indices = rvm_Classification(X_train_crossV,y_train_crossV_list,g[j])
                Phi = buildPhi(g[j],X_test_crossV,X_train_crossV[indices])
                y_pred = np.dot(Phi, weights)
                y_test_crossV_list = []
                for val in range(0,len(y_test_crossV)):
                    y_test_crossV_list.append(y_test_crossV[val])
                
                check = [(y_pred[k] > 0) == y_test_crossV_list[k] for k in range(0, len(y_test_crossV_list))]
                classification_rate = float(sum(check)) / len(y_test_crossV_list)
                error_number = error_number + (1-classification_rate)
                rv_number = rv_number + len(indices)
            print("The average error for gamma = " + str(g[j]) + "and the dataset: " + str(classification_datasets[i])+str(r)+ "is: " +   str(error_number/5) )
            print("The number of the relevance vectors for gamma = " + str(g[j]) + "and the dataset: " + str(classification_datasets[i])+str(r)+ "is: " + str(rv_number/5))
            errors.append((error_number/5))
            rv.append((rv_number/5))

        #minimum number for the g.
        print("The total errors for each gamma and dataset: "+str(classification_datasets[i])+str(r)+ " is: ")
        print(errors)
        rvm_index_min = np.argmin(errors)
        #Print the gamma's/error dataset only one time.
        if ( i == 0):
            DrawError(g,errors)
        rvm_best_gamma = g[rvm_index_min]
        print("The best gamma is: ",rvm_best_gamma)

        #RVM Classification (Testing)
        #Build the model
        y_train_list = []
        for val in range(0,len(y_train_array)):
            y_train_list.append(y_train_array[val])


        weights,indices = rvm_Classification(X_train_array,y_train_list,rvm_best_gamma)
        Phi = buildPhi(rvm_best_gamma,X_test_array,X_train_array[indices])
        y_pred = np.dot(Phi, weights)
        y_test_list = []
        for val in range(0,len(y_test_array)):
            y_test_list.append(y_test_array[val])
        check = [(y_pred[k] > 0) == y_test_list[k] for k in range(0, len(y_test_list))]
        classification_rate = float(sum(check)) / len(y_test_list)
        #Error for the testing.
        error_final = 1 - classification_rate
        rv_final = len(indices)
        print("With the gamma: " + str(rvm_best_gamma) + "and the dataset: " +str(classification_datasets[i])+str(r)+ "we achieve an error that is: " + str(error_final) + "with a number of relevance vectors like: " + str(rv_final))
        repetitions_result[r,2] = error_final
        repetitions_result[r,3] = rv_final
        repetitions_result[r,5] = rvm_best_gamma
        print("Now the errors that we obtain until now are: " +str(repetitions_result[:,2]))
        print("and the relevance vectors that we obtain until now are: " + str(repetitions_result[:,3]))

    #Compute mean (classification error, number of support vectors) for dataset i
    svm_result[i,0] = round(1 - np.mean(repetitions_result[:,0]), 3) #error SVM
    svm_result[i,0] = svm_result[i,0]*100                            #error SVM (%)
    svm_result[i,1] = np.mean(repetitions_result[:,1])               #support vectors for SVM
    svm_result[i,2] = round(np.mean(repetitions_result[:,2]),3)      #error for RVM
    svm_result[i,2]= svm_result[i,2]*100                             #error RVM(%)
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
print(np.column_stack((np.array(classification_datasets),svm_result)))
print("######################################################################")

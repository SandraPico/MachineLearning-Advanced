import numpy as np
import math
import random as random
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    cross_val_score,
    train_test_split
)

#-----------Plot Synthetic dataset and banana dataset functions----------#
#Features are defined as X1 and X2.
#Class 1 will be represented with red dots. (X1_class0, X2_class0)
#Class 2 will be represented with blue dots. (X1_class0,X1_class1)
#Relevance vectors positions. (X1_rv, X2_rv) Represented with yellow dots.

def Draw_RVM(X1_class0,X2_class0,X1_class1,X2_class1,X1_rv,X2_rv):
    
    #First we need to plot the data points.
    plt.figure(figsize=(7,6))
    plt.title("Synthetic dataset using RVM")
    plt.plot(X1_class0,X2_class0,'bx',MarkerSize=4.5,label = "Samples Class 1")
    plt.plot(X1_class1,X2_class1,'ro',MarkerSize=4.5,label = "Samples Class 2")
    #Plot the relevance vectors.
    plt.scatter(X1_rv, X2_rv, s=50, marker='o', c='black', label="Relevance vectors", edgecolor='black', linewidth=2)
    #Plot the classification boundary
    plt.legend()


#--------Load Datasets--------------#

#Plot meshgrid.
def make_meshgrid(x, y, h=.02):
    
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    return xx, yy

#Plot contour.
def plot_contours(clf, xx, yy, **params):
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def BoundarySVM(X_train,models,y_train):
    
    X0 = X_train[:,0]
    X1 = X_train[:,1]

    xx, yy = make_meshgrid(X0, X1)

    plt.figure(figsize=(7,6))
    plt.title("Synthetic dataset using SVM")
    
    print(models)
    plot_contours(models, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    #plt.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')


def SupportVectors(data,indices):
    
    X1_rel = []
    X2_rel = []
    for i in range(0,len(indices)):
        index = indices[i]
        X1_rel.append(data[index][0])
        X2_rel.append(data[index][1])
    
    return X1_rel,X2_rel


def RelevanceVectors(data,indices):
    
    X1_rel = []
    X2_rel = []
    for i in range(0,len(indices)):
        index = indices[i]
        X1_rel.append(data[index][0])
        X2_rel.append(data[index][1])
    return X1_rel,X2_rel


def DecisionBoundary(x,y,weights):
    
    steps = 50
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    h_x = (x_max-x_min)/steps
    h_y = (y_max-y_min)/steps
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),np.arange(y_min, y_max, h_y))
    x_grid = np.c_[xx.ravel(), yy.ravel()]
                         
    # Calculate values for each grid point
    PhiGrid = buildPhi(0.5,x_grid,x[indices])
    y_grid = np.dot(PhiGrid, weights)
    sigmoid = np.vectorize(sigma)
    # apply sigmoid for probabilites
    p_grid = sigmoid(y_grid)
    p_grid = p_grid.reshape(xx.shape)
    
    CS = plt.contour(xx,yy,p_grid,levels=[0.25,0.5,0.75],linewidths=2,labels=["p=0.25","p=0.5","p=0.75"], colors=['yellow', 'black', 'green'],linestyles=['--','solid','--'])
    labels = ['p=0.25', 'p=0.5','p=0.75']
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    plt.legend(loc='upper left')
    plt.show(block = False)
    plt.xlim([xx.min(),xx.max()])
    plt.ylim([yy.min(),yy.max()])
    plt.show()


#Def Draw_SVM
def Draw_SVM(X1_class0,X2_class0,X1_class1,X2_class1,X1_support,X2_support):
    
    #First we need to plot the data points.
    #Plotting the samples
    
    plt.plot(X1_class0,X2_class0,'bx',MarkerSize=4.5,label = "Samples Class 1")
    plt.plot(X1_class1,X2_class1,'ro',MarkerSize=4.5,label = "Samples Class 2")
    #Plotting the support vectors.
    print("The number of support vectors: ",len(X1_support))
    plt.scatter(X1_support, X2_support, s=50, marker='o', c='black', label= "Support vectors", edgecolor='black', linewidth=2)
    #Plotting the boundary for classification.
    plt.legend()

def ExtractData(dataset_name):
    #Datasets for Classification
    if (dataset_name == 'synth'):
        features_train,target_train,features_test,target_test = SyntheticDataset()

    return (features_train, target_train, features_test, target_test)


#Synthetic data.
def SyntheticDataset():
    
    train = []
    test = []
    #Training file
    train_file = open('datasets/synthetic/synthetic_train.csv')
    #Test file
    test_file = open('datasets/synthetic/synthetic_test.csv')
    
    #Training 100-example randomly chosen from Ripley's original of 250.
    train_file.readline()
    for i in train_file.readlines():
        train.append([i for i in i.split(';')])
    train = np.array(train)
    cols = len(train[0])
    features_train_aux = (train[:, 0:cols - 1]).astype('float')
    target_train_aux = [(0, 1)[i[0][0] == "0"] for i in train[:, cols - 1]]

    #Indexes that we want
    index = []
    for ind in range(0,100):
        index.append(int(random.uniform(0,244)))

    features_train = features_train_aux[index]
    target_train_aux_array = np.array(target_train_aux)
    target_train = target_train_aux_array[index]

    #The test has to be 1000 elements so it is ok!
    test_file.readline()
    for i in test_file.readlines():
        test.append([i for i in i.split(';')])
    test = np.asarray(test)
    cols_test = len(test[0])
    features_test = (test[:, 0:cols_test - 1]).astype('float')
    target_test = [(0, 1)[i[0][0] == "0"] for i in test[:, cols_test - 1]]
    train_file.close()
    test_file.close()
    return features_train, target_train, features_test, target_test


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


def Datapoints(data,target):
    X1_class1 = []
    X2_class1 = []
    X1_class2 = []
    X2_class2 = []
    print("Datapoints length: ",len(data))
    print("Datapoints target:", len(target))
    for i in range(0,len(target)):
        if(target[i] == 0):
            X1_class1.append(data[i][0])
            X2_class1.append(data[i][1])
        else:
            X1_class2.append(data[i][0])
            X2_class2.append(data[i][1])

    return X1_class1,X2_class1, X1_class2, X2_class2


#---------SVM and RVM Classification tests -------------#

#Results will be:
#SVM error , SVM number of vectors, RVM error, RVM number of relevance vectors.
results = np.zeros((1, 4)) #(classification error, number of support vectors)
print("Synthetic dataset for Ryple's synthetic data." )

#We are going to make an average for 10 datasets.
svm_error_average = []
svm_vectors_average = []
rvm_error_average = []
rvm_vectors_average = []

for ww in range(0,10):
    
    #load dataset (random training)
    X_train,y_train,X_test,y_test = ExtractData('synth')

    #The gamma parameter should be: 0.5
    g = 0.5
    
    #We should set the values of C with cross-validation
    #We are going to test 10 values from 0.5 to 5.0
    C = np.zeros(100)
    value = 0
    for u in range(0,100):
        value = value + 0.5
        C[u] = value

    #SVM Classification - training
    #Cross validation to define the C value
    CrossV_scores = np.zeros(len(C))
        
    #SVM Cross-Validation
    for j in range(len(C)):
        clf = svm.SVC(C[j], kernel='rbf', gamma=0.5)
        #Scores will be an array that store the accuracy for each of the splits.
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') #(training_data, target values)
        CrossV_scores[j] = np.mean(np.absolute(scores))

    index_max = np.argmax(CrossV_scores) #find value with the maximum accuracy.
    best_C = C[index_max]
    print("The best C is :" , best_C)

    #SVM Classification - testing
    best_clf = svm.SVC(best_C, kernel='rbf', gamma = 0.5)
    models = best_clf.fit(X_train, y_train) #create svm model for training data

    res = best_clf.predict(X_test)
    svm_error_average.append(accuracy_score(y_test, res, normalize='True'))
    svm_vectors_average.append(len(best_clf.support_))
    #Plotting the SVM

    if (ww == 1):
        X1_class1,X2_class1,X1_class2,X2_class2 = Datapoints(X_train,y_train)
        X1_rel,X2_rel = SupportVectors(X_train,best_clf.support_)
        BoundarySVM(X_train,models,y_train)
        Draw_SVM(X1_class1,X2_class1,X1_class2,X2_class2,X1_rel,X2_rel)

    #RVM
    weights, indices = rvm_Classification(X_train,y_train,g)
    Phi = buildPhi(g,X_test,X_train[indices])
    y_pred = np.dot(Phi, weights)
    check = [(y_pred[k] > 0) == y_test[k] for k in range(0, len(y_test))]
    error_number = 0
    rv_number = 0
    classification_rate = float(sum(check)) / len(y_test)
    error_number = error_number + (1-classification_rate)
    rv_number = rv_number + len(indices)
    rvm_error_average.append(error_number)
    rvm_vectors_average.append(rv_number)
    #Plot only the last case.
    #Plotting RVM
    if (ww == 1):
        #Then, after that we are going to plot :)
        X1_class1,X2_class1,X1_class2,X2_class2 = Datapoints(X_train,y_train)
        X1_rel,X2_rel = RelevanceVectors(X_train,indices)
        Draw_RVM(X1_class1,X2_class1,X1_class2,X2_class2,X1_rel,X2_rel)
        DecisionBoundary(X_train,y_train,weights)

svm_error = 0
svm_vector = 0
rvm_error = 0
rvm_vector = 0

for o in range(0,10):
    svm_error = svm_error + svm_error_average[o]
    svm_vector = svm_vector + svm_vectors_average[o]
    rvm_error = rvm_error + rvm_error_average[o]
    rvm_vector = rvm_vector + rvm_vectors_average[o]

results[0,0] = svm_error/10
results[0,1] = svm_vector/10
results[0,2] = rvm_error/10
results[0,3] = rvm_vector/10

#Print the result for the dataset.
print("SVM error: ",1-results[0,0])
print("SVM number of vectors: ",results[0,1])
print("RVM error: ",results[0,2])
print("RVM number of vectors: ",results[0,3])




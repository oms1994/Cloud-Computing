import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment

def digit_2_convertlist(word):
    #print(word)
    name_number = {'1':'a','2':'b','3':'c','4':'d','5':'e','6':'f','7':'g','8':'h','9':'i','10':'j','11':'k','12':'l','13':'m','14':'n','15':'o','16':'p','17':'q','18':'r','19':'s','20':'t','21':'u', '22':'v', '23':'w','24':'x','25':'y','26':'z'}
    
     #classid=[];
     #print(word)           
     #print(name_number[i])
     #print(classid)
     #print(name_number[1])
    return name_number[word]

def letter_2_digit_convertlist(word):
     name_number = {'a':1, 'b':2, 'c':3, 'd':4,'e':5, 'f':6, 'g':7, 'h':8,'i':9, 'j':10, 'k':11, 'l':12,'m':13, 'n':14, 'o':15, 'p':16,'q':17, 'r':18, 's':19, 't':20,'u':21, 'v':22, 'w':23,'x':24,'y':25, 'z':26}
     classid=[];
     #print(word)            
     for i in word:     
        classid.append(name_number[i])
     #print(classid)
     return classid

def init_Data_Value():
  ip_Fname = "HandWrittenLetters.txt"
  Data_Value = pd.read_csv(ip_Fname, sep=',', header=None)
  samples = 39
  #option = "omarsbhe" #[]
  #class_ids = letter_2_digit_convertlist(option)
  #print("SFF")
  #print(class_ids)
  class_labels =set(Data_Value.iloc[0])
  train_Data_Value_vals = 33 # number of training instances
  test_Data_Value_vals = 6   # number of test instances
  k = 26 #int(len(list(class_labels)))  # number of clusters
  X_dtrain, Y_dtrain, X_dtest, Y_dtest = Data_Value_picker(ip_Fname=ip_Fname, Nclass=list(class_labels), samples=samples, trainingInstance=train_Data_Value_vals, testingInstance=test_Data_Value_vals)
  return(k,X_dtrain, Y_dtrain, X_dtest, Y_dtest)


def Data_Value_picker(ip_Fname, Nclass, samples, trainingInstance, testingInstance):
  Data_Value = pd.read_csv(ip_Fname, header = None).values
  y_Data_Value = np.transpose(Data_Value[0, :])
  x_Data_Value = np.transpose(Data_Value[1:, :])
 
  Nx = Data_Value.shape[0] - 1
  
  Y_dtrain, Y_dtest = [], []

  X_dtest = np.zeros((1, Nx))
  X_dtrain = np.zeros((1, Nx))

  
  for k in Nclass:
    i = k - 1
    no_of_samples=(samples * i)
    total_samples=no_of_samples + trainingInstance
    X_dtrain = np.vstack((X_dtrain, x_Data_Value[no_of_samples:(total_samples), :]))
    Y_dtrain = np.hstack((Y_dtrain, y_Data_Value[no_of_samples:(total_samples)]))
    X_dtest = np.vstack((X_dtest, x_Data_Value[(total_samples):(total_samples + testingInstance), :]))
    Y_dtest = np.hstack((Y_dtest, y_Data_Value[(total_samples):(total_samples + testingInstance)]))
  
  X_dtrain = X_dtrain[1:, :]
  X_dtest = X_dtest[1:, :]
  
  return X_dtrain, Y_dtrain, X_dtest, Y_dtest 

def clustering(k, X_dtrain, Y_dtrain, X_dtest, Y_sdtest):
  cluster_kmean_vals = KMeans(n_clusters=k)
  cluster_kmean_vals.fit(X_dtrain)#, Y_dtrain) 
  obj_val = getobjective_function(cluster_kmean_vals,X_dtrain)
  print("The Lowest obejective value",obj_val)
  cluster_kmean_vals.inertia_= obj_val
  predicted_y = cluster_kmean_vals.predict(X_dtrain) + 1
  c = confusion_matrix(Y_dtrain, predicted_y)
  cm = c.T 

  idx = linear_assignment(-cm)
  return(cm,idx,c,predicted_y)

def getobjective_function(cluster_kmean_vals,XYMatrix):
    obj_val=[];
    print("The Objective values for 10 iterations")
    for i in range(0,10):
      obj_val.append([-cluster_kmean_vals.fit(XYMatrix).score(XYMatrix)])
      print ([-cluster_kmean_vals.fit(XYMatrix).score(XYMatrix)])
    #min_ind = np.argmin(score)
    return(min(obj_val))
def get_accuracy(idx,c,predicted_y):
  f = open("log1.txt", "a")
  
  #f.close()
  #orignal_mat = c[:, cm[:, 1]]
  #Predicted_labels=[]
  #for x in predicted_y:
   # Predicted_labels.append((digit_2_convertlist(str(x))))
  #print(Predicted_labels)  
  #percent_opt = np.trace(c)/np.sum(c)
  print("\nConfusion Matrix - Original: \n", c)
  #f.write("\nConfusion Matrix - Original: \n", c)
  #print("\nAccuracy: {:2.2f}%\n".format(percent_opt*100))
  f.write("\nConfusion Matrix - Original: \n"+str(c))
  confu_opt = cm[:, idx[:, 1]]
  percent_opt = np.trace(confu_opt)/np.sum(confu_opt)
    
  print("\nReordered Confusion Matrix : \n"+str(confu_opt))
  f.write("\nReordered Confusion Reordered Matrix-: \n"+str(confu_opt))
  print("\nAccuracy: {:2.2f}%\n".format(percent_opt*100))
  f.write("\nAccuracy: {:2.2f}%\n"+str(format(percent_opt*100)))

k,X_dtrain, Y_dtrain, X_dtest, Y_dtest =init_Data_Value()
#for J in range(0,10):
cm,idx,c,predicted_y=clustering(k, X_dtrain, Y_dtrain, X_dtest, Y_dtest)
get_accuracy(idx,c,predicted_y)

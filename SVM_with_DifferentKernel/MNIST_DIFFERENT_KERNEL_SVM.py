from scipy.io import loadmat
import numpy as np
data_path = "./mnist.mat"
data_raw = loadmat(data_path)
images = data_raw['data'].T
label = data_raw['label'][0]


import matplotlib.pyplot as plt
import random
plt.figure(figsize=(20,10))
for i in range(10,20):
    plt.subplot(2, 5, i-9)
    t = random.randint(0,70000)
    plt.imshow(np.reshape(images[t,:], (28,28)), cmap = plt.cm.gray)
    plt.title('Digit %i\n' %label[t], fontsize = 20)
    
from sklearn.model_selection import train_test_split
X_new, X_unused, Y_new, Y_unused = train_test_split(images, label, test_size = 0.9, random_state = 1000)

# split the dataset into training and testing sets
X_trn, X_tst, Y_trn, Y_tst = train_test_split(X_new, Y_new, test_size = 0.3, random_state = 1000) 
X_trn = X_trn/256
X_tst = X_tst/256


# Resize the image data
# as the data is two large
# Choose the first 1000 data
from skimage.transform import rescale, resize, downscale_local_mean
m, n = X_trn.shape
n_new = n//4
X_trn_new = np.zeros((m,n_new))
for i in range(m):
    image = np.reshape(X_trn[i,:], (28,28))
    image_rescaled = rescale(image, 1.0 / 2.0, anti_aliasing=False)
    X_trn_new[i,:] = np.reshape(image_rescaled, n_new)

m2 = X_tst.shape[0]
X_tst_new = np.zeros((m2,n_new))
for i in range(m2):
    image = np.reshape(X_tst[i,:], (28,28))
    image_rescaled = rescale(image, 1.0 / 2.0, anti_aliasing=False)
    X_tst_new[i,:] = np.reshape(image_rescaled, n_new)
    
print('The new training set has size: '+ str(X_trn_new.shape))
print('The new testing set has size: '+ str(X_tst_new.shape))


#### Linear Kernel SVM####
from sklearn.svm import SVC
n = np.array(range(20))
C = 2**n
accuracy_tst = []
accuracy_trn = []

for i in C:
    svc = SVC(C = i,kernel= 'linear')
    svc.fit(X_trn_new, Y_trn)
    accuracy_trn.append(svc.score(X_trn_new, Y_trn))
    accuracy_tst.append(svc.score(X_tst_new,Y_tst ))

max_acc = max(accuracy_tst)

plt.semilogx(C, accuracy_tst)
plt.semilogx(C, accuracy_trn)
plt.title("Linear SVM")
plt.xlabel('C')
plt.ylabel('accuracy')
plt.show()
print('The maximum testing accuracy achieved with Linear SVM is: ' + str(max_acc))


#### Poly Kernel SVM with different degrees####
D = [2, 3, 4]
n = np.array(range(20))
C = 2**n
max_acc = np.zeros(3)
for i in range(3):
    accuracy_tst = []
    accuracy_trn = []

    for j in C:
        svc = SVC(C= j, kernel = 'poly',degree = d)
        svc.fit(X_trn_new, Y_trn)
        accuracy_trn.append(svc.score(X_trn_new, Y_trn))
        accuracy_tst.append(svc.score(X_tst_new, Y_tst))
    
    max_acc[i] += max(accuracy_tst)

    plt.semilogx(C, accuracy_tst)
    plt.semilogx(C, accuracy_trn)
    plt.title("Polynomial Kernel SVM, degree %i" %d)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.show()
    print('The maximum testing accuracy achieved with Polynomial Kernel SVM of degree ' + str(d) + ' is: ' + str(max_acc[i]))


#### Gasussian Kernel( Radial Basis Function Kernel) SVM####  
accuracy_tst = []
accuracy_trn = []
n = np.array(range(20))
C = 2**n

for i in C:
    svc = SVC(C = i, kernel = 'rbf')
    svc.fit(X_trn_new, Y_trn)
    accuracy_trn.append(svc.score(X_trn_new, Y_trn))
    accuracy_tst.append(svc.score(X_tst_new, Y_tst))
    
max_acc = max(accuracy_tst)


plt.semilogx(C, accuracy_tst)
plt.semilogx(C, accuracy_trn)
plt.title("SVM with Gaussian kernel")
plt.xlabel('C')
plt.ylabel('accuracy')
plt.show()
print('The maximum testing accuracy achieved with SVM with Gaussian kernel is: ' + str(max_acc))

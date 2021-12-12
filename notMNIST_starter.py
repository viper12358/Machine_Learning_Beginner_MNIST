import numpy as np
import network
import time

# to run the code: python notMNIST_starter.py
# the program will train a neural net with specified parameters, and output the net in a file called `part2.pkl`. The binary


# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)

    
#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    # CODE GOES HERE
    train_list = []
    test_list = []
    for i in range(len(train_features)):
        flatIsJutice = train_features[i].flatten().reshape(784,1)
        train_list.append(flatIsJutice)
    
    for i in range(len(train_list)):
        train_list[i] = train_list[i]/255.0
        
    for j in range(len(test_features)):
        flatIsJutice = test_features[j].flatten().reshape(784,1)
        test_list.append(flatIsJutice)
    
    for j in range(len(test_list)):
        test_list[j] = test_list[j]/255.0
        
    trainingLabelsEncode = [onehot(label, 10) for label in train_labels]
    trainingData = zip(train_list, trainingLabelsEncode)
    testingData = zip(test_list, test_labels)
       
    return (trainingData, testingData)
    
###################################################################


trainingData, testingData = prepData()
start = time.time()

# features, hidden neurons, output neurons 
net = network.Network([784, 30, 15, 10])

# epochs, batch_size, step
net.SGD(trainingData, 7, 5, 0.7886326, test_data = testingData)
end = time.time()

finish = end - start
print("Run time: ", finish)
network.saveToFile(net, "part2.pkl")








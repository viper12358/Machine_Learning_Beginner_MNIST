import numpy as np
import idx2numpy
import network
import time

# to run the code: python MNIST_starter.py
# the program will train a neural net with specified parameters, and output the net in a file called `part1.pkl`. The binary
# To extract pictures that the program failed to identify, please refer to extract.py
# Note: multiple run might need to be done to get desirable results (Accuracy > 95%)

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


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "data/train-images.idx3-ubyte"
trainingLabelFile = "data/train-labels.idx1-ubyte"

# testing data
testingImageFile = "data/t10k-images.idx3-ubyte"
testingLabelFile = "data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    images = idx2numpy.convert_from_file(imagefile) 
    
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    # CODE GOES HERE
    
    flatten_list = []
    
    for i in range(len(images)):
        flatIsJutice = images[i].flatten().reshape(784,1)
        flatten_list.append(flatIsJutice)
        
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    # CODE GOES HERE
    for i in range(len(flatten_list)):
        flatten_list[i] = flatten_list[i]/255.0
   
    # return features
    return flatten_list


# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    ntrain, train_labels = getLabels(trainingLabelFile)
    ntest, test_labels = getLabels(testingLabelFile)
    # CODE GOES HERE
    
    trainingLabelsEncode = [onehot(label, 10) for label in train_labels]
    
    trainingFeatures = getImgData(trainingImageFile)
    trainingData = zip(trainingFeatures, trainingLabelsEncode)
    
    testingFeatures = getImgData(testingImageFile)
    testingData = zip(testingFeatures, test_labels)
    
    return (trainingData, testingData)
    

###################################################


trainingData, testingData = prepData()

# features, hidden neurons, output neurons 
start = time.time()
net = network.Network([784,42,10])

# epochs, mini_batch_size, step_size
net.SGD(trainingData, 9, 3, 0.69, test_data = testingData)
end = time.time()

finish = end - start
print("Run time in seconds: ", finish)
# saving the neural net
network.saveToFile(net, "part1.pkl")




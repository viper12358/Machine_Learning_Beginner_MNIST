import csv
import numpy as np
import network
import time

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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

def getDataFromSample(sample):
    # [1]               min		max		average		std.dev
    # sbp				101		218		138.3		20.5
    sbp_mean = 138.3
    sbp_stdDev = 20.5
    sbp = cv([standardize(float(sample[1]), sbp_mean, sbp_stdDev)])
    
    # tobacco			0		31.2	3.64		4.59        [2]
    tobacco_mean = 3.64
    tobacco_stdDev = 4.59
    tobacco = cv([standardize(float(sample[2]), tobacco_mean, tobacco_stdDev)])
    
    # ldl				.98		15.33	4.74		2.07        [3]
    ldl_mean = 4.74
    ldl_stdDev = 2.07
    ldl = cv([standardize(float(sample[3]), ldl_mean, ldl_stdDev)])
    
    # adiposity		   6.74	    42.49   25.4		7.77        [4]
    adiposity_mean = 25.4
    adiposity_stdDev = 7.77
    adiposity = cv([standardize(float(sample[4]), adiposity_mean, adiposity_stdDev)])
    
    # famhist			binary value (1 = Present, 0 = Absent)  [5]
    if (sample[5] == "Present"):
        famhist = cv([1])    
    elif (sample[5] == "Absent"):
        famhist = cv([0])
    else:
        print("Data processing error. Exiting program.")
        quit()
    
    # typea			    13		78		53.1		9.81        [6]
    typea_mean = 53.1
    typea_stdDev = 9.81
    typea = cv([standardize(float(sample[6]), typea_mean, typea_stdDev)])
    
    # obesity			14.7	46.58	26.0		4.21        [7]
    obesity_mean = 53.1
    obesity_stdDev = 9.81
    obesity = cv([standardize(float(sample[7]), obesity_mean, obesity_stdDev)])
    
    # alcohol			0		147.19	17.0		24.5        [8]
    alcohol_mean = 17.0
    alcohol_stdDev = 24.5
    alcohol = cv([standardize(float(sample[8]), alcohol_mean, alcohol_stdDev)])
    
    # age				15		64		42.8		14.6	    [9]
    age_int = int(sample[9])
    age = cv([float(age_int/64.0)])
    
    features = np.concatenate((sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age), axis=0)
    
    label = int(sample[10])
    
    return features, label

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    # CODE GOES HERE
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        next(reader, None)  # skip the header row
        
        n = 0
        features = []
        labels = []
        
        for row in reader:
            featureVec, label = getDataFromSample(row)
            features.append(featureVec)
            labels.append(label)
            n = n + 1

    print(f"Number of data points read: {n}")
    
    return (n, features, labels)
  


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')

    # CODE GOES HERE
    ntrain = int(n * 5/6)    
    ntest = n - ntrain
    trainingFeatures = features[:ntrain]
    trainingLabels = [onehot(label, 2) for label in labels[:ntrain]]    # training labels should be in onehot form

    print(f"Number of training samples: {ntrain}")

    testingFeatures = features[ntrain:]
    testingLabels = labels[ntrain:]
    print(f"Number of testing samples: {ntest}")

    trainingData = zip(trainingFeatures, trainingLabels)
    testingData = zip(testingFeatures, testingLabels)
    return (trainingData, testingData)



###################################################

start = time.time()
trainingData, testingData = prepData()

net = network.Network([9,10,5,2])
net.SGD(trainingData, 8, 5, 0.2, test_data = testingData)

end = time.time()
network.saveToFile(net, "part3.pkl")
print("Time taken in seconds: ", end-start)


       
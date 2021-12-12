import MNIST_starter
import network

# to run the code: python extract.py
# the program will read from the pkl file and train the network
# then it will print out the first 3 images that neural net cannot identify

# load the neural net
neuralNet = network.loadFromFile("part1.pkl")
trainingData, testingData = MNIST_starter.prepData()
row, col = neuralNet.getFailedData(testingData)
# print the first 3 images: their indices and labels
for i in range(3):
    print("Failed image's index: ", row[i])
    print("False label: ", col[i])

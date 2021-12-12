import MNIST_starter
import network

# to run the code: python extract.py
# the program will read from the pkl file and train the network
# then it will print out the first 3 images that neural net cannot identify



# load the neural net
model = network.loadFromFile("part1.pkl")
trainingData, testingData = MNIST_starter.prepData()
row, col = model.getFailedData(testingData)
# print the first 3 images: their indices and labels
print(row[0], row[1], row[2])
print(col[0], col[1], col[2])
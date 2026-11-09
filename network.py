import random


# simple neural network
class Level:
    def __init__(self, inputCount, outputCount):
        self.inputs = [0] * inputCount
        self.outputs = [0] * outputCount
        self.biases = [0] * outputCount

        self.weights= []
        for i in range(inputCount):
            self.weights.append([0]*outputCount)

        self.randomize()

    def randomize(self):
        for i in range(len(self.inputs)):
            for j in range(len(self.outputs)):
                self.weights[i][j] = random.uniform(-1,1)
        
        for j in range(len(self.outputs)):
            self.biases[j]= random.uniform(-1,1)


    def feed_forward(self, inputs):
        self.inputs = inputs

        for j in range(len(self.outputs)):
            total = 0 
            for i in range(len(self.inputs)):
                total += self.inputs[i] * self.weights[i][j]
            
            if total > self.biases[j]:
                self.outputs[j] = 1
            else:
                self.outputs[j] = 0
        
        return self.outputs



class NeuralNetwork:
    def __init__(self,sizes):
        self.levels = []
        for i in range(len(sizes)-1):
            self.levels.append(Level(sizes[i],sizes[i+1]))
    
    def feed_forward(self,inputs):
        outputs= inputs
        for level in self.levels:
            outputs = level.feed_forward(outputs)
        return outputs


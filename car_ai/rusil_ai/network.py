import copy
import math
import random


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
            total = self.biases[j]
            for i in range(len(self.inputs)):
                total += self.inputs[i] * self.weights[i][j]

            # tanh keeps the real strength of the signal (-1 to 1) instead of just 0 or 1
            self.outputs[j] = math.tanh(total)

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
        # only the final buttons are on/off: pressed when the output is positive
        return [1 if value > 0 else 0 for value in outputs]

    def mutate(self, amount, strength=0.2):
        # each weight and bias has an `amount` chance to get a small random nudge
        for level in self.levels:
            for j in range(len(level.biases)):
                if random.random() < amount:
                    level.biases[j] += random.gauss(0, strength)
            for i in range(len(level.weights)):
                for j in range(len(level.weights[i])):
                    if random.random() < amount:
                        level.weights[i][j] += random.gauss(0, strength)

def breed(mom, dad):
    # Child brain: every weight and bias is picked from mom or dad at random
    child = copy.deepcopy(mom)
    for c_level, d_level in zip(child.levels, dad.levels):
        for j in range(len(c_level.biases)):
            if random.random() < 0.5:
                c_level.biases[j] = d_level.biases[j]
        for i in range(len(c_level.weights)):
            for j in range(len(c_level.weights[i])):
                if random.random() < 0.5:
                    c_level.weights[i][j] = d_level.weights[i][j]
    return child


if __name__ == "__main__":
    brain = NeuralNetwork([8, 6, 4])
    print(len(brain.levels))                                        
    print(brain.feed_forward([0.5, 0.8, 1.0, 0.3, 1.0, 0.6, 0.2, 0.5]))  

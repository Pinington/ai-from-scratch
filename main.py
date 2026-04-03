from math import exp
import random

class BasicAI:
    THRESHOLD = 0.5
    LEARNING_RATE = 0.015
    LOSS = 0.05

    def __init__(self):
        # Weights and biases for layer 1:
            # Weights and bias for neuron 1
        self.w1x, self.w1y, self.b1 = random.uniform(1,4), random.uniform(-2,1), random.uniform(-0.1,0.1)

            # Weights and bias for neuron 2
        self.w2x, self.w2y, self.b2 = random.uniform(-2,2), random.uniform(-1,2), random.uniform(-0.1,0.1)

            # Weights and bias for neuron 3
        self.w3x, self.w3y, self.b3 = random.uniform(-4,-1), random.uniform(-2,1), random.uniform(-0.1,0.1)

        # Weights and biases for layer 2 (output layer)
        self.wO1, self.wO2, self.wO3, self.b = random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1), random.uniform(-0.1,0.1)


    @staticmethod
    def sigmoid(val):
        if val < -709: 
            return 0.0
        elif val > 709:
            return 1.0
        return 1 / (1 + exp(-val))
    

    @staticmethod
    def true_value(x, y):
        if (y >= 0):
            if (y > 2 * x and y > -2 * x):
                return 1 #BLUE
            else:
                return 0 #RED
        else:
            if (y < 2 * x and y < -2 * x):
                return 0 #RED
            else:
                return 1 #BLUE
    
    
    def reinitialize_weights(self):
        def init_weight():
            return random.gauss(0, 0.5)
        
        self.w1x, self.w1y, self.b1 = init_weight(), init_weight(), init_weight()
        self.w2x, self.w2y, self.b2 = init_weight(), init_weight(), init_weight()
        self.w3x, self.w3y, self.b3 = init_weight(), init_weight(), init_weight()


    def loss(self, x, y):
        output_neuron = self.layer_two(x, y)
        true_value = self.true_value(x, y)
        
        return (output_neuron - true_value)**2
    

    @property
    def get_loss(self):
        loss = 0
        for i in range(5000):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            loss += self.loss(x, y)
        loss /= 5000
        return loss


    def train(self):
        training_x = random.uniform(-50, 50)
        training_y = random.uniform(-50, 50)
        predicted = self.layer_two(training_x, training_y)

        delta = predicted - self.true_value(training_x, training_y)
        n1, n2, n3 = self.layer_one(training_x, training_y)

        # Output layer partial derivatives
        grad_w1 = 2 * delta * predicted * (1 - predicted) * n1
        grad_w2 = 2 * delta * predicted * (1 - predicted) * n2
        grad_w3 = 2 * delta * predicted * (1 - predicted) * n3
        grad_b = 2 * delta * predicted * (1 - predicted)

        # Output layer update
        self.wO1 -= self.LEARNING_RATE * grad_w1
        self.wO2 -= self.LEARNING_RATE * grad_w2
        self.wO3 -= self.LEARNING_RATE * grad_w3
        self.b -= self.LEARNING_RATE * grad_b


        # Hidden layer partial derivatives
        grad_w1x = 2 * delta * predicted * (1 - predicted) * self.wO1 * n1 * (1 - n1) * training_x
        grad_w1y = 2 * delta * predicted * (1 - predicted) * self.wO1 * n1 * (1 - n1) * training_y
        grad_b1 =  2 * delta * predicted * (1 - predicted) * self.wO1 * n1 * (1 - n1)

        grad_w2x = 2 * delta * predicted * (1 - predicted) * self.wO2 * n2 * (1 - n2) * training_x
        grad_w2y = 2 * delta * predicted * (1 - predicted) * self.wO2 * n2 * (1 - n2) * training_y
        grad_b2 =  2 * delta * predicted * (1 - predicted) * self.wO2 * n2 * (1 - n2)

        grad_w3x = 2 * delta * predicted * (1 - predicted) * self.wO3 * n3 * (1 - n3) * training_x
        grad_w3y = 2 * delta * predicted * (1 - predicted) * self.wO3 * n3 * (1 - n3) * training_y
        grad_b3 =  2 * delta * predicted * (1 - predicted) * self.wO3 * n3 * (1 - n3)

        # Hidden layer update
        self.w1x -= self.LEARNING_RATE * grad_w1x
        self.w1y -= self.LEARNING_RATE * grad_w1y
        self.b1 -= self.LEARNING_RATE * grad_b1

        self.w2x -= self.LEARNING_RATE * grad_w2x
        self.w2y -= self.LEARNING_RATE * grad_w2y
        self.b2 -= self.LEARNING_RATE * grad_b2

        self.w3x -= self.LEARNING_RATE * grad_w3x
        self.w3y -= self.LEARNING_RATE * grad_w3y
        self.b3 -= self.LEARNING_RATE * grad_b3
        

    def training(self, n):
        i = 0
        loss = 10
        while (loss > self.LOSS and i < n): # LOSS is an experimental value
            if i % 10000 == 0:
                loss = self.get_loss
                print(f"Step {i}, Loss: {self.get_loss:.4f}")
            if i != 0 and i % 23000 == 0: # 23000 is an experimental value
                self.reinitialize_weights()
            self.train()
            i += 1
    

    # Returns neuron activation at layer one
    def layer_one(self, x, y):
        z1 = self.w1x * x + self.w1y * y + self.b1
        z2 = self.w2x * x + self.w2y * y + self.b2
        z3 = self.w3x * x + self.w3y * y + self.b3

        return (self.sigmoid(z1), self.sigmoid(z2), self.sigmoid(z3))
    

    # Returns neuron activation at layer two (output layer)
    def layer_two(self, x, y):
        n1, n2, n3 = self.layer_one(x, y)
        z = self.wO1 * n1 + self.wO2 * n2 + self.wO3 * n3 + self.b

        return self.sigmoid(z)
    

    def classify(self, x, y):
        output_neurones = self.layer_two(x, y)
        if output_neurones > self.THRESHOLD:
            return "BLUE"
        else:
            return "RED"


def get_slope(w_x, w_y):
    if abs(w_y) < 1e-6: 
        return float('inf') if w_x > 0 else float('-inf')
    return round(-w_x / w_y, 2)


if __name__ == "__main__":
    basicAI = BasicAI()
    basicAI.training(1000000)
    loss = basicAI.get_loss

    print()

    w1 = get_slope(basicAI.w1x, basicAI.w1y)
    w2 = get_slope(basicAI.w2x, basicAI.w2y)
    w3 = get_slope(basicAI.w3x, basicAI.w3y)

    val_list = [w1, w2, w3]
    smallest = min(val_list)

    val_list.remove(smallest)
    middle = min(val_list) 
    
    biggest = max(val_list)

    print(f"The slopes used: -2, 0, 2\nThe deduced weights and biases: {smallest}, {middle}, {biggest}")

    prediction_one = basicAI.classify(-10, 3)
    prediction_two = basicAI.classify(0, 10)
    prediction_three = basicAI.classify(10, 3)
    prediction_four = basicAI.classify(-10, -3)
    prediction_five = basicAI.classify(0, -10)
    prediction_six = basicAI.classify(10, -3)

    print(f"    The pixel at (-10, 3) is in the {prediction_one} zone, should be RED ({prediction_one} : RED)")
    print(f"    The pixel at (0, 10) is in the {prediction_two} zone should be BLUE ({prediction_two} : BLUE)")
    print(f"    The pixel at (10, 3) is in the {prediction_three} zone should be RED ({prediction_three} : RED)")
    print(f"    The pixel at (-10, -3) is in the {prediction_four} zone should be BLUE ({prediction_four} : BLUE)")
    print(f"    The pixel at (0, -10) is in the {prediction_five} zone should be RED ({prediction_five} : RED)")
    print(f"    The pixel at (10, -3) is in the {prediction_six} zone should be BLUE ({prediction_six} : BLUE)")

    print(f"Average loss is {loss:.2}")
    print()

import math

# This function read the max value, min value, hidden layers and output layer from lander.txt
def load_model(filename):
    min_vals = []
    max_vals = []
    hidden_layer = []
    output_layer = []

    with open(filename, "r") as f:      #Open the file
        for line in f:
            line = line.strip()     #remove spaces or newline
            if not line or line.startswith("#"):        #If the line is empty or start with a # skip it
                continue

            # If line starts with min; split the min and convert each value to float.
            if line.startswith("min:"):
                parts = line.split("min:")[1].strip().split(",")
                min_vals = [float(p) for p in parts]

            # If line starts with max; split the max and convert each value to float.
            elif line.startswith("max:"):
                parts = line.split("max:")[1].strip().split(",")
                max_vals = [float(p) for p in parts]

            # If line starts with h
            elif line.startswith("h"):
                parts = line.split(":")[1].strip().split(",")
                # All values are weight except bias
                weights = [float(h) for h in parts[:-1]]
                bias = float(parts[-1])
                #Creat neuron with the numebr of weights
                neuron = Neuron(len(weights))
                # Write weight and bias values to created neuron
                neuron.weights = weights[:]
                neuron.bias = bias
                hidden_layer.append(neuron)

            #This section same with the hidden layer but this is our output layer
            elif line.startswith("o"):
                parts = line.split(":")[1].strip().split(",")
                weights = [float(o) for o in parts[:-1]]
                bias = float(parts[-1])
                neuron = Neuron(len(weights))
                neuron.weights = weights[:]
                neuron.bias = bias
                output_layer.append(neuron)
    return min_vals, max_vals, hidden_layer, output_layer


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [0.0] * num_inputs
        self.bias = 0.0
        self.activation = 0.0

    def sigmoid(self, x):
        # This is our sigmoid activation function.
        return 1.0 / (1.0 + math.exp(-x))

    def compute_activation(self, inputs):
        total = 0.0
        # Calculate weighted sum (inputs * weights + bias)
        for w, inp in zip(self.weights, inputs):
            total += w * inp
        total += self.bias

        self.activation = self.sigmoid(total)       #Apply the sigmoid activation
        return self.activation


def feedforward(inputs, hidden_layer, output_layer):
    # Calculation of hidden layer outputs
    hidden_outputs = []
    for neuron in hidden_layer:
        hidden_layer_outputs = neuron.compute_activation(inputs)
        hidden_outputs.append(hidden_layer_outputs)

    # Calculation of output layer outputs.
    final_outputs = []
    for neuron in output_layer:
        output_layer_outputs = neuron.compute_activation(hidden_outputs)
        final_outputs.append(output_layer_outputs)

    return final_outputs

#Normalize x and y values to [0,1]
def normalize_inputs(x_pos, y_pos, min_vals, max_vals):
    x_norm = (x_pos - min_vals[0]) / (max_vals[0] - min_vals[0])
    y_norm = (y_pos - min_vals[1]) / (max_vals[1] - min_vals[1])
    return [x_norm, y_norm]

# This is denormalization to get real outputs for the gameloop file
def denormalize_outputs(normalized_outputs, min_vals, max_vals):

    thruster_norm = normalized_outputs[0]
    turning_norm = normalized_outputs[1]

    thruster = thruster_norm * (max_vals[2] - min_vals[2]) + min_vals[2]
    turning = turning_norm * (max_vals[3] - min_vals[3]) + min_vals[3]

    return thruster, turning

class NeuralNetHolder:
    #This class first loads our txt file and create predictions with predict function which calling in gameloop.
    def __init__(self):
        try:
            self.min_vals, self.max_vals, self.hidden_layer, self.output_layer = load_model("lander.txt")
            print("Model Loaded")
            print(f"   Min vals: {self.min_vals}")
            print(f"   Max vals: {self.max_vals}")
            print(f"   Hidden neurons: {len(self.hidden_layer)}")
            print(f"   Output neurons: {len(self.output_layer)}")
        except Exception as e:
            print(f"Error {e}")

    def predict(self, input_row):

        # parse the input row to string.
        parts = input_row.split(",")
        x_position_to_target = float(parts[0])
        y_position_to_target = float(parts[1])

        # normalizations
        normalized_inputs = normalize_inputs(
            x_position_to_target,
            y_position_to_target,
            self.min_vals,
            self.max_vals
        )
        # These lines make predictions.
        outputs = feedforward(normalized_inputs, self.hidden_layer, self.output_layer)      #Feedforward line for the prediction
        thruster, turning = denormalize_outputs(outputs, self.min_vals, self.max_vals)      #We need to denormalize outputs. If we do not denormalize it, our lander can't land to target.

        # GameLoop is waiting for the [x_vel, y_vel] format from us
        # However this is our outputs [thruster (y_vel), turning (x_vel)]
        # we must change our outputs to [turning (x_vel), thruster (y_vel)] for gameloop
        return [turning, thruster]


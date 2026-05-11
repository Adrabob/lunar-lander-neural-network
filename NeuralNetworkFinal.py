import random
import math


hidden_neurons = 8
num_input_per_neuron = 2
epochs = 150

#This is our csv load file
def load_csv(filepath):
    data = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()         #This line for removing linespace
        if line == "":              #If line is empty skip the line
            continue

        split_line = line.split(",")     #Split every column with comma
        if len(split_line) != 4:         #Ensure that we have 4 columns 
            continue

        x_pos = float(split_line[0])        # x position to target
        y_pos = float(split_line[1])        # y position to target
        thruster = float(split_line[2])  # y_velocity(thruster)
        turning = float(split_line[3])  # x_velocity(turning)
    
        data.append([x_pos, y_pos, thruster, turning])
    return data

# This func is for the finding min and max of each column
def min_max_norm(data):
    #This is our infinity for min and -infinity for max
    min_values = [float("inf")] * 4
    max_values= [float("-inf")] * 4

    for row in data:
        for i in range(4):
            value = row[i]
            if value < min_values[i]:
                min_values[i] = value
            if value > max_values[i]:
                max_values[i] = value
    return min_values, max_values


def data_normalization(data, min_values, max_values):
    normalized = []
    for row in data:
        new_row = []
        for i in range(4):
            min_v = min_values[i]
            max_v = max_values[i]
            if max_v - min_v == 0:
                normalized_values = 0.0
            else:
                #Normalize the all data to [0,1]
                normalized_values = (row[i] - min_v) / (max_v - min_v)
            new_row.append(normalized_values)
        normalized.append(new_row)
    return normalized

#This func split the data to train data and validation data
def split_dataset(data):
    total_length = len(data)

    train_end = int(total_length * 0.70)    # %70
    val_end = int(total_length * 0.85)      # %70 + %15 = %85

    train_data = data[:train_end]     #This is our train data
    validation_data = data[train_end:val_end]        #This is our validation data
    test_data = data[val_end:]      #This is our test data

    return train_data, validation_data, test_data

# This is a single neuron. It handles weight, bias and gradient
class Neuron:

    def __init__(self, num_inputs):
        # We use random.random for picking random numbers between -1.0 and 1.0 for weights and bias.
        rand_val = 2 * random.random() - 1

        self.weights = [rand_val for i in range(num_inputs)]        #Create weights with random values between -1.0 and 1.0.
        self.bias = rand_val        #Create bias with random values between -1.0 and 1.0.
        self.activation = 0.0
        self.gradient = 0.0

        self.prev_delta_weights = [0.0 for i in range(num_inputs)]      #Storage for momentum. It stores previous weight changes.
        self.prev_delta_bias = 0.0

    def sigmoid(self, x):
        # This is our sigmoid activation function.
        return 1.0 / (1.0 + math.exp(-x))

    def sigmoid_derivative(self, output):
        # This is derivative of the sigmoid function. We use this for calculating gradients in backpropagation.
        return output * (1.0 - output)

    def compute_activation(self, inputs):
        total = 0.0
        # Calculate weighted sum (inputs * weights + bias)
        for w, inp in zip(self.weights, inputs):
            total += w * inp
        total += self.bias

        self.activation = self.sigmoid(total)       #Apply the sigmoid activation
        return self.activation


#Creates a layer containing the specified number of neurons.
def create_layer(num_neurons, num_inputs_per_neuron):
    layer = []
    for n in range(num_neurons):
        neuron = Neuron(num_inputs_per_neuron)
        layer.append(neuron)
    return layer


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

    return final_outputs, hidden_outputs

# This is our backpropagation function. We update weights and biases in this function.
def backpropagation(inputs, targets, hidden_layer, output_layer, learning_rate, momentum):
    # Calculate gradient for output layer. Error = target - output
    for i, neuron in enumerate(output_layer):
        output = neuron.activation
        error = targets[i] - output
        neuron.gradient = error * neuron.sigmoid_derivative(output)     # neuron gradient = Error * Derivative of activation

    # Calculate gradient for hidden layer
    for h_index, h_neuron in enumerate(hidden_layer):
        sum_gradients = 0.0

        for o_neuron in output_layer:
            sum_gradients += o_neuron.gradient * o_neuron.weights[h_index]      #Multiply output gradient with the weight

        h_neuron.gradient = sum_gradients * h_neuron.sigmoid_derivative(h_neuron.activation)

    # Update output layer weights and biases
    for o_neuron in output_layer:
        for w_index in range(len(o_neuron.weights)):
            #
            delta_w = (learning_rate * o_neuron.gradient * hidden_layer[w_index].activation) + (momentum * o_neuron.prev_delta_weights[w_index])

            o_neuron.prev_delta_weights[w_index] = delta_w      # Store changes for next momentum step and update weight
            o_neuron.weights[w_index] += delta_w

        # Update Bias
        delta_b = (learning_rate * o_neuron.gradient) + (momentum * o_neuron.prev_delta_bias)
        o_neuron.prev_delta_bias = delta_b
        o_neuron.bias += delta_b

    # Update hidden layer weights and biases
    for h_neuron in hidden_layer:
        for w_index in range(len(h_neuron.weights)):
            # This is same formula with the output layers update but inputs come from dataset's inputs
            delta_w = (learning_rate * h_neuron.gradient * inputs[w_index]) + (momentum * h_neuron.prev_delta_weights[w_index])

            h_neuron.prev_delta_weights[w_index] = delta_w
            h_neuron.weights[w_index] += delta_w

        # Bias update
        delta_b = (learning_rate * h_neuron.gradient) + (momentum * h_neuron.prev_delta_bias)
        h_neuron.prev_delta_bias = delta_b
        h_neuron.bias += delta_b

# This function calculates the rmse on validation and test dataset.
def rmse_calculation(validation_test_data, hidden_layer, output_layer):
    total_error = 0.0
    num_samples = len(validation_test_data)
    num_outputs = 2

    for row in validation_test_data:
        inputs = row[:2]
        targets = row[2:]

        outputs, hidden_outputs = feedforward(inputs, hidden_layer, output_layer)

        for i in range(num_outputs):
            error = targets[i] - outputs[i]
            total_error += error * error

    mse = total_error / (num_samples * num_outputs)     #Calcualtion of MSE
    rmse = math.sqrt(mse)       #Calculation of rmse (math.sqrt(mse))
    return rmse


def train_network(train_data, validation_data, hidden_layer, output_layer, epochs, learning_rate, momentum, early_stopping_patience=10):

    length_of_train_data = len(train_data)
    num_outputs = 2

    print(f"Train samples: {length_of_train_data}, Val samples: {len(validation_data)}")
    print(f"Epochs: {epochs}, Learning rate: {learning_rate}, Momentum: {momentum}")
    print(f"Hidden neurons: {len(hidden_layer)}, Output neurons: {len(output_layer)}")
    print("=" * 70)

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        random.shuffle(train_data)      #This line shuffle the training data to prevent the network from memorizing the order of the data
        total_error = 0.0

        # THis is our training loop for every epoch feedforward -> calculate error -> backpropagation -> calculate rmse
        for row in train_data:
            inputs = row[:2]  # [x_pos_norm, y_pos_norm]
            targets = row[2:]  # [thruster_norm, turning_norm]

            # Feedforward step
            outputs, hidden_outputs = feedforward(inputs, hidden_layer, output_layer)

            # Calculate the error for every epoch
            for i in range(num_outputs):
                error = targets[i] - outputs[i]
                total_error += error * error

            # Backpropagation step
            backpropagation(inputs, targets, hidden_layer, output_layer,
                          learning_rate, momentum)

        # Calculation of trainin rme
        mse = total_error / (length_of_train_data * num_outputs)
        train_rmse= math.sqrt(mse)

        # Print the calculated valdation rmse every 5 epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_rmse = rmse_calculation(validation_data, hidden_layer, output_layer)
            print(f"[Epoch {epoch + 1:4d}/{epochs}] Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f}")

            # Early stopping for the learning
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping is activated: Validation RMSE {early_stopping_patience * 5}")
                print(f"Best Validation RMSE: {best_val_rmse:.6f}")
                break
        else:
            print(f"[Epoch {epoch + 1:4d}/{epochs}] Train RMSE: {train_rmse:.6f}")

    print(f"Training finished")
    print(f"Final Train RMSE: {train_rmse:.6f}")
    print(f"Final Val RMSE: {rmse_calculation(validation_data, hidden_layer, output_layer):.6f}")


# save model parameters to text file. This file will load by the NeuralNetHolder.
def save_model(filename, min_values, max_values, hidden_layer, output_layer):
    with open(filename, "w") as f:
        # Save min and max values for normalization
        f.write("# MIN-MAX VALUES\n")
        f.write("min: " + ",".join(str(v) for v in min_values) + "\n")
        f.write("max: " + ",".join(str(v) for v in max_values) + "\n\n")

        #This is for saving hidden layer weights.
        f.write("# HIDDEN LAYER WEIGHTS\n")
        for idx, neuron in enumerate(hidden_layer):
            line = [str(w) for w in neuron.weights] + [str(neuron.bias)]
            f.write(f"h{idx}: " + ",".join(line) + "\n")
        f.write("\n")

        #This save output layer weights.
        f.write("# OUTPUT LAYER WEIGHTS\n")
        for idx, neuron in enumerate(output_layer):
            line = [str(w) for w in neuron.weights] + [str(neuron.bias)]
            f.write(f"o{idx}: " + ",".join(line) + "\n")

    print(f"Model successfully saved = {filename}")


# Main Program
if __name__ == "__main__":
    # Load the data
    dataset = load_csv("ce889_dataCollection.csv")
    print(f"Total {len(dataset)} data loaded.\n")

    # Calculation of min max of data
    min_values, max_values = min_max_norm(dataset)
    print("Statistics:")
    print(f"   X_pos: [{min_values[0]:.2f}, {max_values[0]:.2f}]")
    print(f"   Y_pos: [{min_values[1]:.2f}, {max_values[1]:.2f}]")
    print(f"   Thruster (Y_vel): [{min_values[2]:.2f}, {max_values[2]:.2f}]")
    print(f"   Turning (X_vel): [{min_values[3]:.2f}, {max_values[3]:.2f}]\n")

    normalized_data = data_normalization(dataset, min_values, max_values)       #Normalize the data to [0, 1]
    train_data, validation_data, test_data = split_dataset(normalized_data)        # Split the data to train and validation data

    # Creating the network layers
    print("Neural Network Layers Creating")
    hidden_layer = create_layer(hidden_neurons, num_input_per_neuron)
    output_layer = create_layer(num_input_per_neuron, hidden_neurons)
    print("Neural Network Layers Created")
    print(f"Architecture: 2 → {hidden_neurons} → 2\n")

    # Train the data
    train_network(
        train_data=train_data,
        validation_data=validation_data,
        hidden_layer=hidden_layer,
        output_layer=output_layer,
        epochs=epochs,
        learning_rate=0.01,
        momentum=0.8,
        early_stopping_patience=20
    )

    # Calculate RMSE on the Test set
    test_rmse = rmse_calculation(test_data, hidden_layer, output_layer)
    print(f" Test RMSE: {test_rmse:.6f}")

    # Saving of the model
    save_model(
        "lander.txt",
        min_values=min_values,
        max_values=max_values,
        hidden_layer=hidden_layer,
        output_layer=output_layer
    )
    print("\nModel Saved.")


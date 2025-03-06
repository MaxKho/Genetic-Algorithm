# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import librosa
import scipy.io
from scipy.signal.windows import hann
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
population_size = 10
num_generations = 4
num_parents = 4
mutation_rate = 0.2
conv_chance = 0.8
noise_level = 0.1
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 8

def load_and_process_pcvc_data(directory='.', train_size=0.8, random_seed=42):
    """
    Load, process, and split the PCVC dataset from .mat files into training and validation sets,
    with randomized window slicing for the training data to expand the dataset size.

    Instructions for data preparation:
    1. Visit the Kaggle dataset page at https://www.kaggle.com/sabermalek/pcvcspeech
    2. Download the dataset by clicking on the 'Download' button.
    3. Extract the downloaded zip file.
    4. Ensure all .mat files from the dataset are in the same directory as this script or specify the directory.

    Parameters:
    - directory: str, the directory where .mat files are located (default is the current directory).
    - train_size: float, the proportion of the dataset to include in the train split.
    - random_seed: int, the seed used for random operations to ensure reproducibility.

    Returns:
    - tr_data: np.array, training dataset.
    - tr_labels: np.array, training labels.
    - vl_data: np.array, validation dataset.
    - vl_labels: np.array, validation labels.
    """

    # List all .mat files in the specified directory
    all_mats = [file for file in os.listdir(directory) if file.endswith('.mat')]
    raw_data = []
    num_vowels = 6
    ndatapoints_per_vowel = 299
    labels = []

    for idx, mat_file in enumerate(all_mats):
        mat_path = os.path.join(directory, mat_file)
        mat_data = np.squeeze(scipy.io.loadmat(mat_path)['x'])
        raw_data.append(mat_data)
        labels.append(np.repeat(np.arange(num_vowels)[np.newaxis], mat_data.shape[0], axis=0))

    # Concatenate and reshape all data
    raw_data, labels = np.concatenate(raw_data, axis=1), np.concatenate(labels, axis=1)
    nreps, nvow, nsamps = raw_data.shape
    raw_data = np.reshape(raw_data, (nreps * nvow, nsamps), order='F')
    labels = np.reshape(labels, (nreps * nvow), order = 'F')

    # Split data into training and validation sets
    tr_data, vl_data, tr_labels, vl_labels = train_test_split(
        raw_data, labels, train_size=train_size, random_state=random_seed, stratify=labels)

    # Define window size and function
    window_size = 10000
    window = hann(window_size)

    # Define noise variance for data augmentation
    noise_std = np.std(tr_data) * noise_level

    # Process Training Data with random slicing
    tr_data_processed = []
    tr_labels_processed = []
    for j in range(10): # repeat the tr data 10 times
      for i, d in enumerate(tr_data):
          start = np.random.randint(0, nsamps - window_size)
          end = start + window_size
          sliced = d[start:end] * window
          resampled = librosa.resample(sliced, orig_sr=48000, target_sr=16000)
          noise = np.random.normal(loc=0, scale=noise_std, size=resampled.shape)
          resampled += noise
          tr_data_processed.append(resampled)
          tr_labels_processed.append(tr_labels[i])
    tr_data = np.array(tr_data_processed)
    tr_labels= np.array(tr_labels_processed)

    # Process Validation Data with fixed slicing
    vl_data = vl_data[:, 5000:15000] * window
    vl_data = np.array([librosa.resample(d, orig_sr=48000, target_sr=16000) for d in vl_data])

    # One-hot encode labels
    tr_labels = np.eye(num_vowels)[tr_labels]
    vl_labels = np.eye(num_vowels)[vl_labels]

    return tr_data, tr_labels.astype('float'), vl_data, vl_labels.astype('float')

class Net(nn.Module):
    """
    Defines a neural network architecture dynamically based on a specified genome configuration.

    The `Net` class constructs a neural network where each layer's configuration is dictated by the genome.
    The network will always end with a linear layer with an output size of `K`, meant
    to match the number of classes in the dataset.

    Parameters:
    - genome (list of dicts): Specifies the architecture of the neural network. Each dictionary in the list
      represents a layer in the network and should include keys for 'num_neurons' (int), 'activation' (str),
      and optionally 'dropout_rate' (float).
    - D (int): The dimensionality of the input data. Defaults to 3.
    - K (int): The number of output classes. Defaults to 4.

    Attributes:
    - network (nn.Sequential): The sequential container of network layers as specified by the genome.
    """
    def __init__(self, genome, D=3, K=4, input_channels=1):
        super().__init__()
        layers = []
        conv_output_size = D
        activation_map = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'swish': nn.SiLU()}

        for gene in genome:
            if 'conv_filters' in gene:
                gene['kernel_size'] = min([gene['kernel_size'], conv_output_size])
                gene['stride'] = min([gene['stride'], max([1, conv_output_size // 2])])

                conv_output_size = (conv_output_size - gene['kernel_size'] + 2 * gene['padding']) // gene['stride'] + 1
                gene['pool_size'] = max([1, min([gene['pool_size'], conv_output_size])])
                conv_output_size = conv_output_size // gene['pool_size']
                conv_output_size = max(1, conv_output_size)

                layers.append(nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=gene['conv_filters'],
                    kernel_size=gene['kernel_size'],
                    stride=gene['stride'],
                    padding=gene['padding']
                ))
                layers.append(activation_map[gene['activation']])
                layers.append(nn.MaxPool1d(kernel_size=gene['pool_size']))
                input_channels = gene['conv_filters']

        layers.append(nn.Flatten())
        fc_input_size = input_channels * conv_output_size

        for gene in genome:
            if 'neurons' in gene:
                layers.append(nn.Linear(fc_input_size, gene['neurons']))
                layers.append(activation_map[gene['activation']])
                layers.append(nn.Dropout(gene['dropout']))
                fc_input_size = gene['neurons']

        layers.append(nn.Linear(fc_input_size, K))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
        - x (Tensor): The input data tensor.

        Returns:
        - Tensor: The output of the network after processing the input tensor through all the layers defined
          in the `network` attribute.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.network(x)

def generate_initial_population(size, blueprint):
    """
    Generates an initial population of neural network architectures based on a flexible blueprint.

    Each individual in the population (or 'genome') consists of a randomly constructed neural network architecture.
    The architecture is determined by randomly selecting from possible configurations specified in the blueprint.

    Parameters:
    - size (int): The number of neural network architectures to generate in the population.
    - blueprint (dict): A dictionary specifying the possible configurations for neural network layers.
      The blueprint can contain keys such as:
      - 'n_layers' (int): The number of layers to include in each network architecture.
      - 'neurons' (list): Possible numbers of neurons per layer.
      - 'activations' (list): Possible activation functions.
      - 'dropout' (list): Possible dropout rates, including None if dropout is not to be applied.
      - convolutional hyperparameters for convolutional layers

      Each layer in a generated architecture randomly selects from these lists, promoting a diverse initial population.

    Returns:
    - population (list of list of dicts): A list of neural network architectures, where each architecture
      is represented as a list of dictionaries. Each dictionary defines the configuration of one layer.

    Example:
    >>> population = generate_initial_population(10, blueprint)
    >>> len(population)
    10
    """

    population = []
    for i in range(size):
      individual = []
      for j in range(random.randint(1, blueprint['max_n_layers'])):
        if random.random() < conv_chance:
          layer = {
              'conv_filters': random.choice(blueprint['conv_filters']),
              'kernel_size': random.choice(blueprint['kernel_size']),
              'stride': random.choice(blueprint['stride']),
              'padding': random.choice(blueprint['padding']),
              'pool_size': random.choice(blueprint['pool_size']),
              'activation': random.choice(blueprint['activations'])}
        else:
          layer = {'neurons': random.choice(blueprint['neurons']),
                  'activation': random.choice(blueprint['activations']),
                  'dropout': random.choice(blueprint['dropout'])}
        individual.append(layer)

      population.append(sorted(individual, key = lambda x: 0 if 'conv_filters' in x else 1))

    return population

def selection(population, fitnesses, num_parents):
    """
    Selects the top-performing individuals from the population based on their fitness scores.

    This function sorts the population by fitness in descending order and selects the top `num_parents`
    individuals to form the next generation's parent group. This selection process ensures that individuals
    with higher fitness have a higher probability of reproducing and passing on their genes.

    Parameters:
    - population (list of list of dicts): The population from which to select top individuals. Each individual
      in the population is represented as a genome, which is a list of dictionaries where each dictionary
      details the configuration of a neural network layer.
    - fitnesses (list of floats): A list of fitness scores corresponding to each individual in the population.
      Each fitness score should be a float indicating the performance of the associated individual.
    - num_parents (int): The number of top-performing individuals to select for the next generation.

    Returns:
    - list of list of dicts: A list containing the genomes of the top-performing individuals selected from the
      population.

    Example:
    >>> population = [[{'num_neurons': 32, 'activation': 'relu'}], [{'num_neurons': 16, 'activation': 'sigmoid'}]]
    >>> fitnesses = [0.95, 0.88]
    >>> selected = selection(population, fitnesses, 1)
    >>> len(selected)
    1
    """
    sorted_population = sorted(population, key=lambda x: fitnesses[population.index(x)], reverse=True)
    return sorted_population[:num_parents]

def crossover(parent1, parent2):
    """
    Combines two parent genomes to create a new child genome through a crossover process.

    Parameters:
    - parent1 (list of dicts): The genome of the first parent.
    - parent2 (list of dicts): The genome of the second parent.

    Returns:
    - list of dicts: The genome of the child, formed by combining genes from the two parents.
    """
    child = []
    for i in range(min(len(parent1), len(parent2))):
        layer1 = parent1[i]
        layer2 = parent2[i]
        if 'conv_filters' in layer1 and 'conv_filters' in layer2:
            child_layer = {
                'conv_filters': random.choice([layer1['conv_filters'], layer2['conv_filters']]),
                'kernel_size': random.choice([layer1['kernel_size'], layer2['kernel_size']]),
                'stride': random.choice([layer1['stride'], layer2['stride']]),
                'padding': random.choice([layer1['padding'], layer2['padding']]),
                'pool_size': random.choice([layer1['pool_size'], layer2['pool_size']]),
                'activation': random.choice([layer1['activation'], layer2['activation']])}
        else:
            child_layer = {
                'neurons': random.choice([layer1.get('neurons', 64), layer2.get('neurons', 64)]),
                'activation': random.choice([layer1.get('activation', 'relu'), layer2.get('activation', 'relu')]),
                'dropout': random.choice([layer1.get('dropout', 0.1), layer2.get('dropout', 0.1)])}
        child.append(child_layer)

    if len(parent1) > len(parent2):
        child += parent1[len(parent2):]
    elif len(parent2) > len(parent1):
        child += parent2[len(parent1):]

    return sorted(child, key = lambda x: 0 if 'conv_filters' in x else 1)

def mutate(genome):
    genome_mutated = []
    # Randomly sample the number of layers from a discretised normal distribution
    n_layers = max(1, round(np.random.normal(len(genome), 3 * mutation_rate)))

    for j in range(n_layers):
        if j >= len(genome):
            # Randomly create convolutional layers with probability defined by the conv_chance hyperparameter
            if random.random() < conv_chance:
              layer = {
                  'conv_filters': random.choice(blueprint['conv_filters']),
                  'kernel_size': random.choice(blueprint['kernel_size']),
                  'stride': random.choice(blueprint['stride']),
                  'padding': random.choice(blueprint['padding']),
                  'pool_size': random.choice(blueprint['pool_size']),
                  'activation': random.choice(blueprint['activations'])}
            else:
              layer = {'neurons': random.choice(blueprint['neurons']),
                      'activation': random.choice(blueprint['activations']),
                      'dropout': random.choice(blueprint['dropout'])}
            genome_mutated.append(layer)
        else:
            # Create additional layers if the number of layers generated exceeds the current layer count
            layer = genome[j].copy()
            if 'neurons' in layer:
                error = round(np.random.normal(0, 5 * mutation_rate))
                ind = blueprint['neurons'].index(layer['neurons']) + error
                if ind >= 0 and ind < len(blueprint['neurons']):
                    layer['neurons'] = blueprint['neurons'][ind]
                if random.random() < mutation_rate:
                    layer['activation'] = random.choice(blueprint['activations'])
                if 'dropout' in layer and random.random() < mutation_rate:
                    layer['dropout'] = random.choice(blueprint['dropout'])
            if 'conv_filters' in layer:
                if random.random() < mutation_rate:
                  layer['conv_filters'] = random.choice(blueprint['conv_filters'])
                if random.random() < mutation_rate:
                    layer['kernel_size'] = random.choice([blueprint['kernel_size']])
                if random.random() < mutation_rate:
                    layer['stride'] = random.choice(blueprint['stride'])
                if random.random() < mutation_rate:
                    layer['padding'] = random.choice(blueprint['padding'])
                if random.random() < mutation_rate:
                    layer['pool_size'] = random.choice(blueprint['pool_size'])
                if random.random() < mutation_rate:
                    layer['activation'] = random.choice(blueprint['activations'])
            genome_mutated.append(layer)

    return sorted(genome_mutated, key = lambda x: 0 if 'conv_filters' in x else 1)

def compute_fitness(genome, train_loader, test_loader, criterion, lr=0.01, epochs=5, D=None, K=None):

    # Create the model from the genome
    model = Net(genome, D, K, X_tensor.size(1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / total_batches
        print(f'Epoch {epoch + 1}/{epochs} complete. Average Training Loss: {average_epoch_loss:.4f}')
        total_loss += epoch_loss

    print(f'Training complete.')

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f'Evaluation complete. Accuracy: {accuracy:.4f} ({correct}/{total} correct)\n')

    return accuracy

# Load and process the PCVC dataset into training and validation sets.
# This function will split the raw data, apply preprocessing like windowing and resampling,
# and return processed training data (X, labels) and validation data (X_val, labels_val).
X, labels, X_val, labels_val = load_and_process_pcvc_data()

# Determine the number of samples and classes from the shape of the training data and labels.
# This will help in setting up the network architecture later.
Nsamps, Nclasses = X.shape[-1], labels.shape[-1]

# Convert numpy arrays to PyTorch tensors. 
X_tensor, X_val_tensor = torch.FloatTensor(X).to(device), torch.FloatTensor(X_val).to(device)
y_tensor, y_val_tensor = torch.FloatTensor(labels).to(device), torch.FloatTensor(labels_val).to(device)
X_tensor = X_tensor.unsqueeze(1).to(device)
X_val_tensor = X_val_tensor.unsqueeze(1).to(device)

# Wrap tensors in a TensorDataset.
dataset = TensorDataset(X_tensor, y_tensor)
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

# DataLoader is used to efficiently load data in batches, which is necessary for training neural networks.
# `shuffle=True` ensures that the data is shuffled at every epoch to prevent the model from learning
# any order-based biases in the dataset.
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)


# Initial population definition
blueprint = {
    'max_n_layers': 6,           # Define the (maximum) number of layers in each neural network
    'neurons': [2**i for i in range(3,10)],     # Possible neuron counts per layer
    'activations': ['sigmoid', 'relu', 'leaky_relu', 'swish'], # Possible activation functions
    'dropout': [0.1, 0.2, 0.25],            # Possible dropout rates
    'conv_filters': [8, 16, 32, 64],  # Number of filters for conv layers
    'kernel_size': [3, 5, 7],  # Kernel sizes for conv layers
    'stride': [1, 2],  # Stride for conv layers
    'padding': [0, 1],  # Padding for conv layers
    'pool_size': [2, 3],  # Max pooling size
}

population = generate_initial_population(population_size, blueprint)

# Initialise best performance tracking
best_overall_fitness = float('-inf')
best_overall_architecture = None

for generation in range(num_generations):

    # Evaluate fitnesses
    fitnesses = []
    total_genomes = len(population)
    for idx, genome in enumerate(population):
        # Compute the fitness for each genome
        try:
            fitness = compute_fitness(genome, train_loader, val_loader, nn.CrossEntropyLoss(), lr=LR, epochs=EPOCHS, D=Nsamps, K=Nclasses)
        except:
            # The fitness of genomes with faulty genes is defaulted to 0
            fitness = 0
        fitnesses.append(fitness)
        print(f'    Genome {idx + 1}/{total_genomes} evaluated. "Fitness" (i.e. accuracy): {fitness:.4f}.\n')
    print(f"All genomes in generation {generation} have been evaluated.")

    parents = selection(population, fitnesses, num_parents)

    # Track the best architecture in this generation
    max_fitness_idx = fitnesses.index(max(fitnesses))
    best_fitness_this_gen = fitnesses[max_fitness_idx]
    best_architecture_this_gen = population[max_fitness_idx]

    # Update overall best if the current gen has a new best
    if best_fitness_this_gen > best_overall_fitness:
        best_overall_fitness = best_fitness_this_gen
        best_overall_architecture = best_architecture_this_gen

    print(f"Generation {generation + 1}, Best Fitness: {best_fitness_this_gen}")
    print("Best Architecture:", best_architecture_this_gen, '\n')

    # Generate next generation
    next_generation = parents[:]
    while len(next_generation) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        next_generation.append(child)
    population = next_generation

# Final summary at the end of all generations
print("\nFinal Summary")
print("Best Overall Fitness:", best_overall_fitness)
print("Best Overall Architecture:", best_overall_architecture)

# Inform about the beginning of the re-training process
print("\nStarting the re-training of the best model found by the genetic algorithm (corroborate reproducibility)")

# Re-build the best model based on the architecture determined to be most effective during the genetic algorithm.
# This model is built from scratch using the best configuration parameters (genome) found.
best_model = Net(best_overall_architecture, D=Nsamps, K=Nclasses)
best_model = best_model.to(device)

# Set up the loss function and the optimiser. The optimiser is configured to optimise the weights of our neural network,
# and the learning rate is set as per earlier specification.
best_model_criterion = nn.CrossEntropyLoss()
best_model_optimizer = optim.Adam(best_model.parameters(), lr=LR)

# Training loop: This process involves multiple epochs where each epoch goes through the entire training dataset.
for epoch in range(EPOCHS):
    best_model.train()  # Set the model to training mode
    total_loss = 0
    total_batches = len(train_loader)

    # Process each batch of data
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        best_model_optimizer.zero_grad()  # Clear previous gradients
        output = best_model(data)  # Compute the model's output
        loss = best_model_criterion(output, target)  # Calculate loss
        loss.backward()  # Compute gradients
        best_model_optimizer.step()  # Update weights
        total_loss += loss.item()  # Accumulate the loss

    average_epoch_loss = total_loss / total_batches
    print(f'Epoch {epoch + 1}/{EPOCHS} complete. Average Training Loss: {average_epoch_loss:.4f}')

# After training, switch to evaluation mode for testing
best_model.eval()
correct = 0
total = 0

# Disable gradient computation for validation, as it isn't needed and saves memory and computation
with torch.no_grad():
    # Process each batch from the validation set
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = best_model(data)  # Compute the model's output
        pred = output.argmax(dim=1, keepdim=True)  # Find the predicted class
        target = target.argmax(dim=1, keepdim=True)  # Actual class
        correct += pred.eq(target).sum().item()  # Count correct predictions
        total += target.size(0)  # Total number of items

validation_accuracy = correct / total  # Calculate accuracy
print(f'Evaluation on validation set complete. Accuracy: {validation_accuracy:.4f} ({correct}/{total} correct)')

# Save the trained model's weights for future use
torch.save(best_model.state_dict(), 'best_net.pth')
print("Saved the best model's weights to 'best_net.pth'")
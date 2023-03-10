# -*- coding: utf-8 -*-
"""C FeedForwardNN_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/148iPQhbU_lmzO__g3xcVmxzw3vyYEHtw
"""

# CONNECTING GOOGLE DRIVE FOR DATASET
# from google.colab import drive
# drive.mount("/content/drive/")
DATASET_PATH = "classification/data/"

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

np.random.seed(42)
NUM_FEATS = 90  # Number of features

def read_data(file_name):
  data = pd.read_csv(f"{DATASET_PATH}{file_name}", header=0)
  train_Y = data['1'] # First column is the labels column
  data = data.drop(['1'], axis=1)

  scaler = MinMaxScaler()
  scaler.fit(data)
  train_X = pd.DataFrame(scaler.transform(data))*2-1 # Min Max Normalization
  return train_X, pd.DataFrame(train_Y)

def apply_pca(train_X):
  current_features = train_X.T
  cov_matrix = np.cov(current_features)
  values, vectors = np.linalg.eig(cov_matrix)
  explained_variances = []
  for i in range(len(values)):
      explained_variances.append(values[i] / np.sum(values))
  
  current_features = train_X.T
  cov_matrix = np.cov(current_features)
  values, vectors = np.linalg.eig(cov_matrix)
  explained_variances = []
  for i in range(len(values)):
      explained_variances.append(values[i] / np.sum(values))

  baseline_coverage = 0.9
  current_coverage = 0
  reduced_feature_count = 0
  for ev in range(len(explained_variances)):
    current_coverage += explained_variances[ev]
    if current_coverage > baseline_coverage:
      reduced_feature_count = ev
      break
  
  NUM_FEATS = reduced_feature_count
  reduced_features_X = pd.DataFrame()
  for i in range(reduced_feature_count):
    reduced_features_X[f'Feature{i}'] = train_X.dot(vectors.T[i])
  
  return reduced_features_X, vectors

class Net(object):
  def __init__(self, num_layers, num_units):
    self.num_layers = num_layers
    self.num_units = num_units

    self.biases = []
    self.weights = []

    for i in range(num_layers):
      if i == 0:  # Input Layer
        self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, num_units)))
      else: # Hidden Layer
        self.weights.append(np.random.uniform(-1, 1, size=(num_units, num_units)))
      self.biases.append(np.random.uniform(-1, 1, size=(num_units, 1)))

    # Output Layer
    self.weights.append(np.random.uniform(-1, 1, size=(num_units, 4)))
    self.biases.append(np.random.uniform(-1, 1, size=(4, 1)))

  def __call__(self, train_X):
    self.aggregates = list()
    self.activations = list()
    
    layer_input = train_X
    for layer in range(self.num_layers):
      # print("lay:",layer_input.shape) 
      # print("Sw:",self.weights[layer].shape) 
      # print("Sb:",self.biases[layer].T.shape)
      aggregate = np.dot(layer_input, self.weights[layer]) + self.biases[layer].T
      activation = self.relu(aggregate)
      self.aggregates.append(aggregate)
      self.activations.append(activation)
      layer_input = activation
    # Output layer
    aggregate = np.dot(layer_input, self.weights[layer+1]) + self.biases[layer+1].T
    aggregate = np.exp(aggregate) / np.sum(np.exp(aggregate), axis=1, keepdims=True)
    self.aggregates.append(aggregate)
    self.activations.append(aggregate)
    
    return aggregate

  def relu(self, input_matrix):
    return np.maximum(input_matrix, 0)

  def relu_grad(self, input_matrix):
    return input_matrix > 0

  def backward(self, x, y, y_hat, lamda):
    weight_gradients = [None]*(self.num_layers + 1)
    node_gradients = [None]*(self.num_layers + 1)

    sum = 0.0
    for w in self.weights:
      sum += np.sqrt(np.sum(np.square(w)))

    # Calculating node gradients
    for layer in range(self.num_layers, -1, -1):
      if layer == self.num_layers:
        node_gradients[layer] = np.subtract(y_hat, y)
      else:
        temp = np.dot(node_gradients[layer+1], self.weights[layer+1].T)
        node_gradients[layer] = np.multiply(temp, self.relu_grad(self.aggregates[layer]))
      
    for layer in range(self.num_layers, 0, -1):
      weight_gradients[layer] = np.einsum('ij,ik->ijk', self.activations[layer-1], node_gradients[layer])

    layer = layer - 1
    weight_gradients[layer] = np.einsum('ij,ik->ijk', x, node_gradients[layer])
    
    for layer in range(self.num_layers, -1, -1):
      weight_gradients[layer] = np.mean(weight_gradients[layer], axis = 0)

    for layer in range(self.num_layers, -1, -1):
      node_gradients[layer] = np.sum(node_gradients[layer], axis = 0)
      if type(node_gradients[layer]) != np.ndarray:
        node_gradients[layer] = node_gradients[layer].to_numpy()
    node_gradients[0] = node_gradients[0].reshape(128,1)
    node_gradients[1] = node_gradients[1].reshape(4,1)
    # node_gradients[2] = node_gradients[2].reshape(1,1)
    return weight_gradients, node_gradients

# n[0].shape, n[1].shape, n[2].shape

# w[0].shape, w[1].shape, w[2].shape

class Optimizer(object):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.m = []
    self.v = []
    self.t = 1
    # self.learning_rate = learning_rate

  def step(self, weights, delta_weights, biases, delta_biases, beta1 = 0.9,beta2 = 0.999):
    # for layer in range(len(weights)):
    #   # print(type(delta_weights[layer]))
    #   delta_wt = self.learning_rate*delta_weights[layer]
    #   weights[layer] = np.subtract(weights[layer], delta_wt)

    # for layer in range(len(weights)):
    #   delta_bias = self.learning_rate*delta_biases[layer]
    #   # print(delta_bias.shape)
    #   biases[layer] = np.subtract(biases[layer], delta_bias)
    """ Adam optimizer, bias correction is implemented. """
    if(self.t==1):
      for layer in range(len(weights)):
        self.m.append(np.random.uniform(0.0, 0.0, size=delta_weights[layer].shape))
        self.v.append(np.random.uniform(0.0, 0.0, size=delta_weights[layer].shape))
        # updated_params = []
        
    for  layer in range(len(weights)):    
      self.m[layer] = beta1 * self.m[layer] + (1-beta1) * delta_weights[layer]       
      self.v[layer] = beta2 * self.v[layer] + (1-beta2) * delta_weights[layer] **2
      m_corrected = self.m[layer] / (1-beta1**self.t)
      v_corrected = self.v[layer] / (1-beta2**self.t)
      weights[layer] += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
        
    self.t +=1
        
    for layer in range(len(weights)):
      delta_bias = self.learning_rate*delta_biases[layer]
     
    biases[layer] = np.subtract(biases[layer], delta_bias)
    return weights, biases
  def loss_regularization(weights, biases):
    # '''
    # Compute l2 regularization loss.
    # Parameters
    # ----------
    # 	weights and biases of the network.
    # Returns
    # ----------
    # 	l2 regularization loss 
    # '''
    l2=[]
    # l2 = (biases**2)/len(weights)
    # for i in range(len(weights)):
    # 	l2.append((biases[i]**2)/len(weights[i]))
    # return np.sum(l2)	
    # raise NotImplementedError
    sum = 0.0
    for w in weights:
      sum += np.sqrt(np.sum(np.square(w)))
    return sum
  def cross_entropy_loss(self,y, y_hat):
    '''
    Compute cross entropy loss
    Parameters
    ----------
      y : targets, numpy array of shape m x 1
      y_hat : predictions, numpy array of shape m x 1
    Returns
    ----------
      cross entropy loss
    '''
    m = y.shape[1]              
    cost = - (1 / m) * np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1 - y, np.log(1 - y_hat)))

    return cost

def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	pred = net(inputs)
	new_pred = []
	key_list = list(dict.keys())
	val_list = list(dict.values())
	for i in range(len(pred)):
		for j in range(4):
			if(pred[i,j] == pred[i].max()):
				new_pred.append(key_list[val_list.index(j)])
	
	f = open('output.csv', 'w')
	writer = csv.writer(f)

	# write a row to the csv file
	writer.writerow(["Id","Predictions"])
	for i in range(len(new_pred)):
		writer.writerow([i+1, new_pred[i]])


	# close the file
	f.close()


def loss_rmse(y, y_hat):
  return (((np.subtract(y, y_hat)**2).sum())/y.shape[0])**(0.5)

import random
train_X, train_Y = read_data('train.csv')
# train_X, v = apply_pca(train_X)
train_Y = pd.get_dummies(train_Y)
FFNN = Net(1, 128) # 2 hidden layers, 64 nodes each
learning_rate = 10**-6
lamda = 0.1
optimizer = Optimizer(learning_rate)
no_of_epochs = 200
best_weights = []
best_biases = []
best_error = 10**9
batch_size = 32
flag = 0
mse_error = 0.0
no_of_samples = train_X.shape[0]
l = [x for x in range(no_of_samples)]

for i in range(no_of_epochs):
  np.random.shuffle(l)

  y_pred = pd.DataFrame(FFNN(train_X.iloc[l[:batch_size]]))
  mse_error = optimizer.cross_entropy_loss(train_Y.iloc[l[:batch_size]], y_pred).iloc[0]
  
  if mse_error < best_error:
    best_weights = FFNN.weights 
    best_biases = FFNN.biases
    best_error = mse_error

  if flag == 0 and i > 300:
    learning_rate = 10**-10
    optimizer = Optimizer(learning_rate)
    flag = 1

  # if flag == 1 and i > 1500:
  #   learning_rate = 10**-30
  #   optimizer = Optimizer(learning_rate)
  #   flag = 2
  
  if mse_error < 8 and i>1000:
    break

  del_w, del_b = FFNN.backward(train_X.iloc[l[:batch_size]], train_Y.iloc[l[:batch_size]], y_pred, lamda)
  new_w, new_b = optimizer.step(FFNN.weights, del_w, FFNN.biases, del_b)

  print(f"EPOCH {i} : CE Loss : {mse_error}")

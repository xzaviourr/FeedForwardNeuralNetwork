import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 29
DATASET_PATH = "22m0758/regression/data/"

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		self.aggregates = list()
		self.activations = list()
		
		layer_input = X
		for layer in range(self.num_layers):
			aggregate = np.dot(layer_input, self.weights[layer]) + self.biases[layer].T
			activation = self.relu(aggregate)
			self.aggregates.append(aggregate)
			self.activations.append(activation)
			layer_input = activation
		
		# Output layer
		aggregate = np.dot(layer_input, self.weights[layer+1]) + self.biases[layer+1].T
		self.aggregates.append(aggregate)
		self.activations.append(aggregate)
		
		self.y_hat = pd.DataFrame(aggregate)
		return aggregate

	def relu(self, input_matrix):
		return np.maximum(input_matrix, 0)

	def relu_grad(self, input_matrix):
		return input_matrix > 0

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		'''
		weight_gradients = [None]*(self.num_layers + 1)
		node_gradients = [None]*(self.num_layers + 1)

		# Calculating node gradients
		for layer in range(self.num_layers, -1, -1):
			if layer == self.num_layers:
				node_gradients[layer] = 2*np.subtract(self.y_hat, y) 
			else:
				temp = np.dot(node_gradients[layer+1], self.weights[layer+1].T)
				node_gradients[layer] = np.multiply(temp, self.relu_grad(self.aggregates[layer])) 
		
		for layer in range(self.num_layers, 0, -1):
			weight_gradients[layer] = np.einsum('ij,ik->ijk', self.activations[layer-1], node_gradients[layer]) + 2 * lamda * self.weights[layer]

		layer = layer - 1
		weight_gradients[layer] = np.einsum('ij,ik->ijk', X, node_gradients[layer]) + 2 * lamda * self.weights[layer]
		
		for layer in range(self.num_layers, -1, -1):
			weight_gradients[layer] = np.mean(weight_gradients[layer], axis = 0)

		for layer in range(self.num_layers, -1, -1):
			node_gradients[layer] = np.sum(node_gradients[layer], axis = 0)
			if type(node_gradients[layer]) != np.ndarray:
				node_gradients[layer] = node_gradients[layer].to_numpy()

		for l in range(len(node_gradients)):
			node_gradients[l] = node_gradients[l].reshape(-1,1)
		return weight_gradients, node_gradients

class Optimizer(object):
    def __init__(self, learning_rate):
        self.m = []
        self.v = []
        self.t = 1
        self.learning_rate = learning_rate
        
    def step(self, weights, biases, delta_weights , delta_biases, beta1 = 0.9, beta2 = 0.999):
        if(self.t==1):
          for layer in range(len(weights)):
            self.m.append(np.random.uniform(0.0, 0.0, size=delta_weights[layer].shape))
            self.v.append(np.random.uniform(0.0, 0.0, size=delta_weights[layer].shape))
        
        for  layer in range(len(weights)):
          self.m[layer] = beta1 * self.m[layer] + (1-beta1) * delta_weights[layer]       
          self.v[layer] = beta2 * self.v[layer] + (1-beta2) * delta_weights[layer] **2
          m_corrected = self.m[layer] / (1-beta1**self.t)
          v_corrected = self.v[layer] / (1-beta2**self.t)
          weights[layer] += (-self.learning_rate * m_corrected) / (np.sqrt(v_corrected) + 1e-8)
        
        self.t +=1
        
        for layer in range(len(weights)):
          delta_bias = self.learning_rate*delta_biases[layer]
     
          biases[layer] = np.subtract(biases[layer], delta_bias)
        return weights,biases

def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	return (((np.subtract(y, y_hat)**2).sum())/y.shape[0]).iloc[0]

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	total_no_of_weights = 0
	square_sum = 0
	for w in weights:
		square_sum += np.sum(np.square(w))
		total_no_of_weights += w.shape[0] * w.shape[1]
	lr_loss = square_sum/(2*total_no_of_weights)
	return lr_loss

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	loss = loss_mse(y, y_hat) + lamda*loss_regularization(weights, biases)
	return loss

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	return float((((np.subtract(y, y_hat)**2).sum())/y.shape[0])**(0.5))

def cross_entropy_loss(y, y_hat):
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

def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss

			# print(e, i, rmse(batch_target, pred), batch_loss)

		print(e, epoch_loss)

		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)

	print(dev_rmse)
	print('RMSE on dev data: {:.5f}'.format(dev_rmse))

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
	tst = net(inputs)
	df = pd.DataFrame(tst)
	df.index = df.index+1
	df.to_csv('part2.csv', header=['Predictions'], index=True, index_label='Id')
	return tst

def minmaxscaler(data):
	data=(data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
	return pd.DataFrame(data)

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

    baseline_coverage = 0.92
    current_coverage = 0
    reduced_feature_count = 0
    for ev in range(len(explained_variances)):
        current_coverage += explained_variances[ev]
        if current_coverage > baseline_coverage:
            reduced_feature_count = ev
            break
  
    NUM_FEATS = reduced_feature_count
    return vectors

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	def read_d(file_name):
		data = pd.read_csv(f"{file_name}", header=0)
		train_Y = data['1'] # First column is the labels column
		data = data.drop(['1'], axis=1)

		train_X = minmaxscaler(data)
		return train_X, pd.DataFrame(train_Y)

	raw_test_X = np.array(pd.read_csv(DATASET_PATH + "test.csv"))
	raw_test_X = minmaxscaler(raw_test_X)

	raw_train_X, train_Y = read_d(DATASET_PATH + 'train.csv')
	raw_dev_X, dev_Y = read_d(DATASET_PATH + "dev.csv")

	columns = [x for x in range(1, 91)]
	columns = columns
	raw_test_X.columns.values[:] = columns
	raw_train_X.columns.values[:] = columns
	raw_dev_X.columns.values[:] = columns

	db = pd.concat([raw_train_X, raw_dev_X, raw_test_X], axis=0)

	vectors = apply_pca(db)

	train_X = pd.DataFrame()
	for i in range(NUM_FEATS):
		train_X[f'Feature{i}'] = raw_train_X.dot(vectors.T[i])

	dev_X = pd.DataFrame()
	for i in range(NUM_FEATS):
		dev_X[f'Feature{i}'] = raw_dev_X.dot(vectors.T[i])

	test_X = pd.DataFrame()
	for i in range(NUM_FEATS):
		test_X[f'Feature{i}'] = raw_test_X.dot(vectors.T[i])

	return train_X, train_Y, dev_X, dev_Y, test_X

def main():
	# Hyper-parameters 
	max_epochs = 500
	batch_size = 16
	learning_rate = 10**-3
	num_layers = 1
	num_units = 256
	lamda = 0.0005 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)

if __name__ == '__main__':
	main()

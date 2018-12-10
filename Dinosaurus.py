#!/usr/bin/env python
# coding: utf-8

# ## RNN and LSTM
# #### By MMA

# In[1]:


import numpy as np
from random import shuffle
import matplotlib.pyplot as plt


# ## Define requirements

# In[2]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print('%s' % (txt,), end='')


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    b = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

    return parameters


def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)  # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by)

    return a_next, p_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']
    daraw = (1 - a * a) * da
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


def rnn_forward(X, Y, a0, parameters, vocab_size=27):
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}

    a[-1] = np.copy(a0)
    loss = 0

    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if (X[t] != None):
            x[t][X[t]] = 1

        # One step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])

        # Update the loss
        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, a, x)

    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])

    return gradients, a


# #### Dataset loading

# In[3]:


data = open('data/dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('Total Characters :  %d , Number of unique characters : %d' % (data_size, vocab_size))


# #### Creating dictionaries 

# In[4]:


char_to_index = {ch: i for i, ch in enumerate(sorted(chars))}
index_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
print(index_to_char)


# #### Clipping Gradient

# In[5]:


def clip_gradient(gradients, maxValue):
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'],                                gradients['db'], gradients['dby']
   
    # clipping
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, a_min=-1 * maxValue, a_max=maxValue, out=gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


# #### Sampling Function

# In[6]:


def sample(parameters, char_to_index, seed):

    # fetch parameters
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'],                            parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # create one-hot vector x
    x = np.zeros(vocab_size)
    x = np.reshape(x, (vocab_size, 1))

    a_old = np.zeros(n_a)
    a_old = np.reshape(a_old, (n_a, 1))
    
    indices = []

    counter = 0
    newline_char = char_to_index['\n']

    index = -1
    # forward phase
    while index != newline_char and counter != 50:
        a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_old) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)
        y = y.flatten()

        np.random.seed(counter + seed)

        # sampling from index set
        index = np.random.choice(27, p=y)

        indices.append(index)
        x = np.zeros(27)
        x = np.reshape(x, (vocab_size, 1))
        x[index] = 1
        a_old = a

        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_index['\n'])

    return indices


# In[7]:


def optimize(X, Y, a_old, parameters, learning_rate = 0.01):
    # Forward phase
    loss, cache = rnn_forward(X, Y, a_old, parameters, vocab_size)
    
    # Backpropagation
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clipping
    gradients = clip_gradient(gradients, 5)
    
    # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X)-1]


# ### Training the model 

# In[ ]:





# In[8]:


def model(data, index_to_char, char_to_index, num_iterations=16000, n_a=50, dino_names=10, vocab_size=27):
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    a_old = np.zeros((n_a, 1))

    with open("data/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    shuffle(examples)

    # Optimization loop
    iteration = []
    losses = []
    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_index[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_index["\n"]]

        curr_loss, gradients, a_prev = optimize(X, Y, a_old, parameters, learning_rate=0.01)
        loss = smooth(loss, curr_loss)

        if j % 100 == 0:
            losses.append(loss)
            iteration.append(j)
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            seed = 0
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_index, seed)
                print_sample(sampled_indices, index_to_char)
                seed += 1

            print('\n')

    return parameters, iteration, losses


# In[9]:


parameters, iteration, losses = model(data, index_to_char, char_to_index)
plt.plot(iteration, losses)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.show()


# In[ ]:





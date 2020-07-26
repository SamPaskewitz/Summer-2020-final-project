import numpy as np
from numpy.random import rand, choice
from scipy.special import softmax

def stoch_matrix(n_row = 3, n_col = 3, noise_factor = 1):
    '''Creates a random(right) stochastic matrix.
    In other words, each row sums to 1.'''
    matrix = 1 + noise_factor*rand(n_row, n_col)
    for i in range(n_row):
        row_sum = np.sum(matrix[i, :])
        matrix[i, :] /= row_sum      
    return matrix

def epsilon_greedy(n_act, epsilon, q_val):
    '''Epsilon-greedy action selection.'''
    if np.all(q_val == q_val[0]):
        act_probs = n_act*[1/n_act]
    else:
        act_probs = n_act*[epsilon/(n_act - 1)]
        act_probs[np.argmax(q_val)] = 1 - epsilon
    action = choice(range(n_act), p = act_probs) # action selection
    return action

def softmax_choice(n_act, phi, q_val):
    '''Softmax action selection.'''
    act_probs = softmax(phi * q_val[obs, :].squeeze()).squeeze() # softmax action probabilities
    action = choice(range(n_act), p = act_probs) # action selection
    return action
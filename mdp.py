import numpy as np
from numpy.random import choice, randint
from misc import stoch_matrix

class mdp:
    '''Class for Markov decision processes (MDPs).'''
    def __init__(self, A, R):
        self.n_state = A.shape[0] # number of states
        self.n_act = A.shape[2] # number of actions
        self.A = A # transition array
        self.R = R # reward array
        
    def sample(self, action, old_state):
        '''Sample the next observation, reward and state.'''
        state = choice(range(self.n_state), p = self.A[old_state, :, action].squeeze()) # move to the next state
        state = state # fully observable -> state = observation
        reward = self.R[old_state, state, action] # determine reward
        return state, reward
    
def make_random_mdp(n_state = 5, n_act = 2):
    A = np.zeros((n_state, n_state, n_act)) # transition array
    R = np.zeros((n_state, n_state, n_act)) # reward array
    for i in range(n_act):
        A[:, :, i] = stoch_matrix(n_row = n_state, n_col = n_state) # transition probabilities for action i
        R[:, :, i] = randint(low = -1, high = 2, size = (n_state, n_state)) # rewards for action i (either -1, 0 or 1)
    return mdp(A, R)
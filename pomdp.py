import numpy as np
from numpy.random import choice, randint, dirichlet, normal
from misc import stoch_matrix

class pomdp:
    '''Class for partially observable Markov decision processes (POMDPs).'''
    def __init__(self, A, B, R):
        self.n_state = B.shape[0] # number of states
        self.n_obs = B.shape[1] # number of observables
        self.n_act = B.shape[2] # number of actions
        self.A = A # transition array
        self.B = B # observation array
        self.R = R # reward array
        
    def sample(self, action, old_state):
        '''Sample the next observation, reward and state.'''
        state = choice(range(self.n_state), p = self.A[old_state, :, action].squeeze()) # move to the next state
        reward = self.R[old_state, state, action] # determine reward
        obs = choice(range(self.n_obs), p = self.B[state, :, action].squeeze()) # sample observation
        return obs, reward, state
    
    def update_belief(self, obs, action, old_belief):
        '''Update belief state.'''
        likelihood = self.B[:, obs, action]
        oblf = np.array(old_belief).reshape((1, self.n_state))
        prior = np.matmul(oblf, self.A[:, :, action])
        numerator = likelihood*prior
        belief = (numerator/np.sum(numerator))
        return belief
    
def make_random_pomdp(n_state = 5, n_obs = 3, n_act = 2):
    A = np.zeros((n_state, n_state, n_act)) # transition array
    B = np.zeros((n_state, n_obs, n_act)) # observation array
    R = np.zeros((n_state, n_state, n_act)) # reward array
    for i in range(n_act):
        A[:, :, i] = stoch_matrix(n_row = n_state, n_col = n_state) # transition probabilities for action i
        B[:, :, i] = stoch_matrix(n_row = n_state, n_col = n_obs) # observation probabilities for action i
        R[:, :, i] = randint(low = -1, high = 2, size = (n_state, n_state)) # rewards for action i (either -1, 0 or 1)
    return pomdp(A, B, R)

def add_noise_pomdp(old_pomdp, A_noise = 2, B_noise = 2, R_noise = 0.01):
    '''Add noise to a POMDP's parameters.'''
    # FIGURE OUT A BETTER WAY TO PERTURB THE TRANSITION AND OBSERVATION MATRICES.
    A = np.zeros((old_pomdp.n_state, old_pomdp.n_state, old_pomdp.n_act)) # transition array
    B = np.zeros((old_pomdp.n_state, old_pomdp.n_obs, old_pomdp.n_act)) # observation array
    for i in range(old_pomdp.n_state):
        for j in range(old_pomdp.n_act):
            A[i, :, j] = dirichlet(alpha = A_noise + old_pomdp.A[i, :, j])
            B[i, :, j] = dirichlet(alpha = B_noise + old_pomdp.B[i, :, j])
    R_noise = normal(loc = 0, scale = R_noise, size = (old_pomdp.n_state, old_pomdp.n_state, old_pomdp.n_act))
    R = old_pomdp.R + R_noise # reward array
    return pomdp(A, B, R)
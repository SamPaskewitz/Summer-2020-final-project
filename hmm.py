import numpy as np
from numpy.random import choice, uniform
from misc import stoch_matrix

class hmm:
    '''Class for hidden Markov models (HMM).'''
    def __init__(self, A, B):
        self.n_state = B.shape[0] # number of states
        self.n_obs = B.shape[1] # number of observables
        self.A = A # transition array
        self.B = B # observation array
        
    def sample(self, old_state):
        '''Sample the next observation and state.'''
        state = choice(range(self.n_state), p = self.A[old_state, :].squeeze()) # move to the next state
        obs = choice(range(self.n_obs), p = self.B[state, :].squeeze()) # sample observation
        return obs, state
    
    def update_belief(self, obs, old_belief):
        '''Update belief state.'''
        likelihood = self.B[:, obs]
        oblf = np.array(old_belief).reshape((1, self.n_state))
        prior = np.matmul(oblf, self.A[:, :])
        numerator = likelihood*prior
        belief = (numerator/np.sum(numerator))
        return belief
    
def make_random_hmm(n_state = 5, n_obs = 3):
    A = stoch_matrix(n_row = n_state, n_col = n_state) # transition matrix
    B = stoch_matrix(n_row = n_state, n_col = n_obs) # observation matrix
    return hmm(A, B)

class learn_hmm:
    '''
    Online HMM learning based on Mongillo and Deneve (2008).
    
    Dimensions of phi (array of sufficient statistics):
    0 = x_{t-1}, 1 = x_{t}, 2 = y_{t}, 3 = x_{T}
    
    # FINISH.
    '''
    def __init__(self, hmm, n_state = 5, eta = 0.0005, a0 = None, b0 = None):
        self.hmm = hmm # HMM whose parameters are to be estimated
        self.n_state = n_state # number of states
        self.n_obs = hmm.n_obs # number of observations
        self.eta = eta # discount factor hyperparameter (essentially a learning rate)
        #a0 = (self.n_state**2)*[1/self.n_state]
        #self.a = np.array(a0).reshape((self.n_state, self.n_state)) # estimated transition probabilities
        #b0 = (self.n_state*self.n_obs)*[1/self.n_obs]
        #self.b = np.array(b0).reshape((self.n_state, self.n_obs)) # estimated observation probabilities
        if a0 is None:
            self.a = stoch_matrix(self.n_state, self.n_state, noise_factor = 0.5)
        else:
            self.a = a0
        if b0 is None:
            self.b = stoch_matrix(self.n_state, self.n_obs, noise_factor = 0.5)
        else:
            self.b = b0
        
    def learn(self, n_t = 1000):
        # THERE'S AN ERROR SOMEWHERE.
        # NUMERICAL ERRORS?  TRY TAKING LOGS
        # THE DIFFERENCE IN COMPUTING GAMMA IS VERY SMALL.
        q0 = 1 + 0.5*uniform(low = 0, high = 1, size = self.n_state) # initial state probs
        q0 = q0/np.sum(q0)
        q = np.array(q0).reshape((self.n_state, 1)) # state estimates (essentially belief states)
        phi = np.zeros((self.n_state, self.n_state, self.n_obs, self.n_state)) # sufficient statistics
        state = choice(range(self.n_state))
        for t in range(n_t):
            obs, state = self.hmm.sample(old_state = state) # sample observation and state
            # I've checked that the gamma computation is correct.
            # NOT THIS PART.
            gamma0 = (self.a*self.b[:, obs]) / np.squeeze(q.transpose()@self.a@self.b[:, obs]) # compute gamma (Equation 2.12)
            log_gamma = np.log((self.a*self.b[:, obs])) - np.log(np.squeeze(q.transpose()@self.a@self.b[:, obs]))
            gamma = np.exp(log_gamma)
            #print(np.abs(gamma0 - gamma)/gamma)
            # Try to remove these 'for' loops at some point.
            for i in range(self.n_state):
                for j in range(self.n_state):
                    # update phi (Equation 2.11)
                    # This is actually a slightly different form of the equation.
                    # Trying to figure out that g_{ij}(l, h) business was driving me crazy.
                    foo = phi[i, j, :, :]@gamma # NOT THIS PART.
                    bar = np.zeros((self.n_obs, self.n_state))
                    bar[obs, :] = gamma[i, j]*q[j]  # NOT THIS PART
                    #bar = gamma[i, j]*delta@q.transpose() # AN ERROR WAS FOUND HERE.
                    phi[i, j, :, :] = foo + self.eta*(bar - foo)
            # NOT THIS PART.
            q = gamma.transpose()@q # update q (Equation 2.13); I've confirmed that one needs to transpose gamma.
            if t > 100:
                # NOT THIS PART.
                self.a = phi.sum((2, 3)) / phi.sum((1, 2, 3)).reshape((self.n_state, 1)) # update A (transition matrix)
                self.b = phi.sum((0, 3)) / phi.sum((0, 2, 3)).reshape((self.n_state, 1)) # update B (observation matrix)
        
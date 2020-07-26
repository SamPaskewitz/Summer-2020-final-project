import numpy as np
import pandas as pd
from misc import epsilon_greedy
from numpy.random import choice
from plotnine import *

class qlearn:
    
    def __init__(self, env, learning_rate = 0.1, epsilon = 0.05, gamma = 0.99):
        self.env = env
        self.n_obs = env.observation_space.n # number of observables (= number of states for basic q-learning)
        self.n_act = env.action_space.n # number of actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.zeros((self.n_obs, self.n_act))
        self.rwd_list = []
        self.obs_list = []
        
    def learn(self, n_episodes = 100): 
        for eps in range(n_episodes):
            obs = self.env.reset()
            for t in range(10000):
                self.obs_list += [obs]
                action = epsilon_greedy(self.n_act, self.epsilon, self.q[obs, :]) # action selection
                new_obs, reward, done, info = self.env.step(action) # interact with environment
                self.rwd_list += [reward] # record reward
                # learning
                q_max = np.max(self.q[new_obs, :].squeeze()) # maximum q value for next observation (observed state)
                delta = reward + self.gamma*q_max - self.q[obs, action] # prediction error
                self.q[obs, action] += self.learning_rate*delta # q update
                # update observation (observed state)
                obs = new_obs
                if done:
                    break # end episode if criterion met
        self.rwd_list = pd.Series(self.rwd_list, dtype = 'float64')
        self.obs_list = pd.Series(self.obs_list)
        self.env.close()
    
    def plot(self, span = 0.2, obs = None):
        '''Plots a learning curve (average reward against time).'''
        if obs is None:
            rwd_to_plot = self.rwd_list
        else:
            rwd_to_plot = self.rwd_list.loc[self.obs_list.isin(obs)]
        df = pd.DataFrame({'rwd' : rwd_to_plot, 't' : range(len(rwd_to_plot))})
        p = (ggplot(df, aes('t', 'rwd')) + geom_smooth(span = span, method = 'lowess', se = False))
        p.draw()
    
    #def save(self):
    
    
    #def load(self):

class po_qlearn:
    
    def __init__(self, env, pomdp, learning_rate = 0.1, epsilon = 0.05, gamma = 0.99):
        self.env = env
        self.pomdp = pomdp
        self.n_state = pomdp.A.shape[0]
        self.n_obs = env.observation_space.n # number of observables (= number of states for basic q-learning)
        self.n_act = env.action_space.n # number of actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.zeros((self.n_state, self.n_act))
        self.rwd_list = [] # record of (real) rewards received
        self.obs_list = []
        
    def learn(self, n_episodes = 100):
        rwd_list = []
        for eps in range(n_episodes):
            belief = np.array(self.n_state*[1/self.n_state]).reshape((1, self.n_state))
            obs = self.env.reset()
            for t in range(10000):
                # prediction and action
                self.obs_list += [obs]
                Q = np.matmul(belief, self.q).squeeze() # compute Q
                action = epsilon_greedy(self.n_act, self.epsilon, Q) # action selection
                obs, reward, done, info = self.env.step(action) # interact with environment
                self.rwd_list += [reward] # record reward
                # Q-learning
                new_belief = self.pomdp.update_belief(obs, action, belief).reshape((1, self.n_state)) # next belief state
                new_Q = np.matmul(new_belief, self.q).squeeze() # Q values for next belief state
                Q_max = np.max(new_Q) # maximum Q value for next belief state
                delta = reward + self.gamma*Q_max - Q[action] # prediction error
                self.q[:, action] += belief.squeeze()*self.learning_rate*delta # q update
                belief = new_belief # update belief state
                if done:
                    break # end episode if criterion met
        self.rwd_list = pd.Series(self.rwd_list, dtype = 'float64')
        self.obs_list = pd.Series(self.obs_list)
        self.env.close()
    
    def plot(self, span = 0.2, obs = None):
        '''Plots a learning curve (average reward against time).'''
        if obs is None:
            rwd_to_plot = self.rwd_list
        else:
            rwd_to_plot = self.rwd_list.loc[self.obs_list.isin(obs)]
        df = pd.DataFrame({'rwd' : rwd_to_plot, 't' : range(len(rwd_to_plot))})
        p = (ggplot(df, aes('t', 'rwd')) + geom_smooth(span = span, method = 'lowess', se = False))
        p.draw()        
    
    #def save(self):
    
    
    #def load(self):

class dynaq:
    '''
    This is tabular Dyna-Q, roughly as described in the Sutton-Barto book (p. 135, 2nd edition).
    This implementation differs in two ways from this, namely
    1) We assume the agent already has a full MDP model, and
    2) The agent chooses actions via the ordinary distribution during simulated experience.
    '''
    
    def __init__(self, env, mdp, learning_rate = 0.1, epsilon = 0.05, gamma = 0.99, n_dyna = 1):
        self.env = env
        self.mdp = mdp
        self.n_obs = env.observation_space.n # number of observables (= number of states for basic q-learning)
        self.n_act = env.action_space.n # number of actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_dyna = n_dyna # number of Dyna-style model-based updates per time step
        self.q = np.zeros((self.n_obs, self.n_act))
        self.rwd_list = [] # record of (real) rewards received
        self.obs_list = []
        
    def learn(self, n_episodes = 100): 
        for eps in range(n_episodes):
            obs = self.env.reset()
            for t in range(10000):
                # prediction and action
                self.obs_list += [obs]
                action = epsilon_greedy(self.n_act, self.epsilon, self.q[obs, :]) # action selection
                new_real_obs, reward, done, info = self.env.step(action) # interact with environment
                self.rwd_list += [reward] # record real reward
                # learning from direct experience
                q_max = np.max(self.q[new_real_obs, :].squeeze()) # maximum q value for next observation (observed state)
                delta = reward + self.gamma*q_max - self.q[obs, action] # prediction error
                self.q[obs, action] += self.learning_rate*delta # q update
                # learning from simulated experience
                for j in range(self.n_dyna):
                    # prediction and action
                    obs = choice(range(self.n_obs)) # starting state (= observation) chosen uniformly
                    action = epsilon_greedy(self.n_act, self.epsilon, self.q[obs, :]) # action selection
                    new_sim_obs, reward = self.mdp.sample(action = action, old_state = obs) # interact with environment
                    # q-learning update
                    q_max = np.max(self.q[new_sim_obs, :].squeeze()) # maximum q value for next observation (observed state)
                    delta = reward + self.gamma*q_max - self.q[obs, action] # prediction error
                    self.q[obs, action] += self.learning_rate*delta # q update
                # update real-world observation (observed state)
                obs = new_real_obs
                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break # end episode if criterion met
        self.rwd_list = pd.Series(self.rwd_list, dtype = 'float64')
        self.obs_list = pd.Series(self.obs_list)
        self.env.close()
        
    def plot(self, span = 0.2, obs = None):
        '''Plots a learning curve (average reward against time).'''
        if obs is None:
            rwd_to_plot = self.rwd_list
        else:
            rwd_to_plot = self.rwd_list.loc[self.obs_list.isin(obs)]
        df = pd.DataFrame({'rwd' : rwd_to_plot, 't' : range(len(rwd_to_plot))})
        p = (ggplot(df, aes('t', 'rwd')) + geom_smooth(span = span, method = 'lowess', se = False))
        p.draw()  
    
    #def save(self):
    
    
    #def load(self):
    
class po_dynaq:
    def __init__(self, env, pomdp, learning_rate = 0.1, epsilon = 0.05, gamma = 0.99, n_dyna = 1):
        self.env = env
        self.pomdp = pomdp
        self.n_state = pomdp.A.shape[0]
        self.n_obs = env.observation_space.n # number of observables (= number of states for basic q-learning)
        self.n_act = env.action_space.n # number of actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_dyna = n_dyna
        self.q = np.zeros((self.n_state, self.n_act))
        self.rwd_list = [] # record of (real) rewards received
        self.obs_list = []
        
    def learn(self, n_episodes = 100): 
        rwd_list = []
        for eps in range(n_episodes):
            belief = np.array(self.n_state*[1/self.n_state]).reshape((1, self.n_state))
            obs = self.env.reset()
            for t in range(10000):       
                # prediction and action
                self.obs_list += [obs] # record real observation
                Q = np.matmul(belief, self.q).squeeze() # compute Q
                action = epsilon_greedy(self.n_act, self.epsilon, Q) # action selection
                obs, reward, done, info = self.env.step(action) # interact with environment
                self.rwd_list += [reward] # record real reward
                # Q-learning
                new_real_belief = self.pomdp.update_belief(obs, action, belief).reshape((1, self.n_state)) # next belief state
                new_Q = np.matmul(new_real_belief, self.q).squeeze() # Q values for next belief state
                Q_max = np.max(new_Q) # maximum Q value for next belief state
                delta = reward + self.gamma*Q_max - Q[action] # prediction error
                self.q[:, action] += belief.squeeze()*self.learning_rate*delta # q update   
                # learning from simulated experience
                for j in range(self.n_dyna):
                    # prediction and action
                    sim_state = choice(range(self.n_state), p = belief.squeeze()) # starting state chosen based on belief probs
                    Q = np.matmul(belief, self.q).squeeze() # compute Q
                    action = epsilon_greedy(self.n_act, self.epsilon, Q) # action selection
                    obs, reward, new_state = self.pomdp.sample(action = action, old_state = sim_state) # simulated environment
                    # q-learning update
                    new_sim_belief = self.pomdp.update_belief(obs, action, belief).reshape((1, self.n_state)) # next belief state
                    new_Q = np.matmul(new_sim_belief, self.q).squeeze() # Q values for next belief state
                    Q_max = np.max(new_Q) # maximum Q value for next belief state
                    delta = reward + self.gamma*Q_max - Q[action] # prediction error
                    self.q[:, action] += belief.squeeze()*self.learning_rate*delta # q update   
                if done:
                    break # end episode if criterion met
                belief = new_real_belief # update belief state based on real observation
        self.rwd_list = pd.Series(self.rwd_list, dtype = 'float64')
        self.obs_list = pd.Series(self.obs_list)
        self.env.close()
    
    def plot(self, span = 0.2, obs = None):
        '''Plots a learning curve (average reward against time).'''
        if obs is None:
            rwd_to_plot = self.rwd_list
        else:
            rwd_to_plot = self.rwd_list.loc[self.obs_list.isin(obs)]
        df = pd.DataFrame({'rwd' : rwd_to_plot, 't' : range(len(rwd_to_plot))})
        p = (ggplot(df, aes('t', 'rwd')) + geom_smooth(span = span, method = 'lowess', se = False))
        p.draw()  
    
    #def save(self):
    
    
    #def load(self):
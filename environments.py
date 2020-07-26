import gym
from gym import spaces
import numpy as np
from numpy.random import rand, randint, choice
from mdp import mdp, make_random_mdp
from pomdp import pomdp, make_random_pomdp

class tiger(gym.Env):
    '''
    This is the tiger problem described by Kaelbling, Littman and Cassandra (1998).
    
    Action labels:
    0 = open left door, 1 = open right door, 2 = listen
    
    State labels:
    0 = tiger behind left door, 1 = tiger behind right door
    
    Observation labels:
    0 = hear tiger behind left door, 1 = hear tiger behind right door
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self, duration = 50):
        super(tiger, self).__init__()
        self.n_state = 2 # number of states
        self.n_obs = 2 # number of observables
        self.n_act = 3 # number of actions
        
        # transition array
        A = np.zeros((2, 2, 3))
        A[:, :, [0, 1]] = 0.5 # transition after opening door
        A[0, 0, 2] = 1 # listening does not change the state
        A[1, 1, 2] = 1 # listening does not change the state
        
        # observation array
        B = np.zeros((2, 2, 3))
        B[0, 0, 2] = 0.85 # correct info from listening
        B[0, 1, 2] = 0.15 # your ears deceive you
        B[1, 0, 2] = 0.15 # your ears deceive you
        B[1, 1, 2] = 0.85 # correct info from listening
        B[:, :, [0, 1]] = 0.5 # no useful info after choosing a door
        
        # reward array
        R = np.zeros((2, 2, 3))
        R[0, :, 0] = -100 # open left door, tiger
        R[1, :, 0] = 10 # open left door, no tiger
        R[0, :, 1] = 10 # open right door, no tiger
        R[1, :, 1] = -100 # open right door, tiger
        R[:, :, 2] = -1 # cost of listening
        R = R.astype('float64') 
        
        self.duration = duration # length of each episode
        self.pomdp = pomdp(A = A, B = B, R = R) # the POMDP
        self.action_space = spaces.Discrete(3) # action space
        self.observation_space = spaces.Discrete(2) # observation space

    def reset(self):
        self.state = randint(low = 0, high = self.n_state, size = 1) # draw initial state from a uniform distribution
        self.t = 0 # set time step (t) to 0
        pseudo_action = choice(range(self.n_act)) # it seems like we need a pseudo-action to sample an initial observation
        obs = choice(range(self.n_obs), p = self.pomdp.B[self.state, :, pseudo_action].squeeze()) # sample observation
        return obs
        
    def step(self, action):
        obs, reward, self.state = self.pomdp.sample(action = action, old_state = self.state)
        done = self.t == self.duration # finish if have completed a fixed number of time steps (duration)
        info = {'state' : self.state} # the agent isn't allowed to see this
        self.t += 1 # advance time step (t)
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        '''Print environment information on screen.'''
        print(f'Time step: {self.t}')
        print(f'State: {self.state}')
        
    def close(self):
        return # FIX THIS.

class maze(gym.Env):
    '''Local perception gridworld maze based on Figure 4 from McCallum (1993).
    This is a POMDP.
    I am interpreting this as an episodic task.
    
    Maze map with state numbers:
    2 3 4 7 8
    1   5   9
    0   6   10

    Maze map with observations:
    2 3 4 3 5
    1   1   1
    0   0   0

    Action labels:
    0 = go north, 1 = go east, 2 = go south, 3 = go west
    '''
    metadata = {'render.modes': ['human']}
    # FINISH UPDATING AND THEN TEST.
    def __init__(self):
        super(maze, self).__init__()
        self.n_state = 11 # number of states
        self.n_obs = 6 # number of observables
        self.n_act = 4 # number of actions
        
        # transition array
        A = np.zeros((11, 11, 4))
        # tracing path starting from state 0
        A[0, 1, 0] = 1 # go north from 0 to 1
        A[1, 2, 0] = 1 # go north from 1 to 2
        A[2, 3, 1] = 1 # go east from 2 to 3
        A[3, 4, 1] = 1 # go east from 3 to 4
        A[4, 5, 2] = 1 # go south from 4 to 5
        A[5, 6, 2] = 1 # go south from 5 to 6
        A[4, 7, 1] = 1 # go east from 4 to 7
        A[7, 8, 1] = 1 # go east from 7 to 8
        A[8, 9, 2] = 1 # go south from 8 to 9
        A[9, 10, 2] = 1 # go south from 9 to 10
        # tracing path starting from state 10 backwards
        A[10, 9, 0] = 1 # go north from 10 to 9
        A[9, 8, 0] = 1 # go north from 9 to 8
        A[8, 7, 3] = 1 # go west from 8 to 7
        A[7, 4, 3] = 1 # go west from 7 to 4
        A[6, 5, 0] = 1 # go north from 6 to 5
        A[5, 4, 0] = 1 # go north from 5 to 4
        A[4, 3, 3] = 1 # go west from 4 to 3
        A[3, 2, 3] = 1 # go west from 3 to 2
        A[2, 1, 2] = 1 # go south from 2 to 1
        A[1, 0, 2] = 1 # go south from 1 to 0
        # bumping into walls leaves the agent where it is
        A[0, 0, [1, 2, 3]] = 1
        A[6, 6, [1, 2, 3]] = 1
        A[10, 10, [1, 2, 3]] = 1
        A[1, 1, [1, 3]] = 1
        A[5, 5, [1, 3]] = 1
        A[9, 9, [1, 3]] = 1
        A[2, 2, [0, 3]] = 1
        A[3, 3, [0, 2]] = 1
        A[7, 7, [0, 2]] = 1
        A[4, 4, 0] = 1
        A[8, 8, [0, 1]] = 1
        
        # observation array
        B = np.zeros((11, 6, 4))
        B[[0, 6, 10], 0, :] = 1
        B[[1, 5, 9], 1, :] = 1
        B[2, 2, :] = 1
        B[[3, 7], 3, :] = 1
        B[4, 4, :] = 1
        B[8, 5, :] = 1
        
        # reward array
        R = np.zeros((11, 11, 4), dtype = 'float64')
        R[:, :, :] = -0.1 # default reward is -0.1
        R[:, 6, :] = 1 # goal location
        # punishment for bumping into walls
        R[0, 0, [1, 2, 3]] = -1
        R[6, 6, [1, 2, 3]] = -1
        R[10, 10, [1, 2, 3]] = -1
        R[1, 1, [1, 3]] = -1
        R[5, 5, [1, 3]] = -1
        R[9, 9, [1, 3]] = -1
        R[2, 2, [0, 3]] = -1
        R[3, 3, [0, 2]] = -1
        R[7, 7, [0, 2]] = -1
        R[4, 4, 0] = -1
        R[8, 8, [0, 1]] = -1
        
        self.pomdp = pomdp(A = A, B = B, R = R) # the POMDP
        self.action_space = spaces.Discrete(4) # action space
        self.observation_space = spaces.Discrete(6) # observation space
        
    def reset(self):
        self.state = randint(low = 0, high = self.n_state, size = 1) # draw initial state from a uniform distribution
        self.t = 0 # set time step (t) to 0
        pseudo_action = choice(range(self.n_act)) # it seems like a need a pseudo-action to sample an initial observation
        obs = choice(range(self.n_obs), p = self.pomdp.B[self.state, :, pseudo_action].squeeze()) # sample observation
        return obs
        
    def step(self, action):
        obs, reward, self.state = self.pomdp.sample(action = action, old_state = self.state)
        done = self.state == 6 # finish if have reached the goal state
        info = {'state' : self.state} # the agent isn't allowed to see this
        self.t += 1 # advance time step (t)
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        '''Print environment information on screen.'''
        print(f'Time step: {self.t}')
        print(f'State: {self.state}')
        
    def close(self):
        return # FIX THIS.
        
class random_pomdp(gym.Env):
    '''This creates a random partially observable Markov process (POMDP).'''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_state = 5, n_obs = 3, n_act = 2, duration = 50):
        super(random_pomdp, self).__init__()
        self.n_state = n_state # number of states
        self.n_obs = n_obs # number of observables
        self.n_act = n_act # number of actions
        self.duration = duration # length of each episode
        self.pomdp = make_random_pomdp(n_state = n_state, n_obs = n_obs, n_act = n_act) # the POMDP
        self.action_space = spaces.Discrete(n_act) # action space
        self.observation_space = spaces.Discrete(n_obs) # observation space
        
    def reset(self):
        self.state = randint(low = 0, high = self.n_state, size = 1) # draw initial state from a uniform distribution
        self.t = 0 # set time step (t) to 0
        pseudo_action = choice(range(self.n_act)) # it seems like we need a pseudo-action to sample an initial observation
        obs = choice(range(self.n_obs), p = self.pomdp.B[self.state, :, pseudo_action].squeeze()) # sample observation
        return obs
        
    def step(self, action):
        obs, reward, self.state = self.pomdp.sample(action = action, old_state = self.state)
        done = self.t == self.duration # finish if have completed a fixed number of time steps (duration)
        info = {'state' : self.state} # the agent isn't allowed to see this
        self.t += 1 # advance time step (t)
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        '''Print environment information on screen.'''
        print(f'Time step: {self.t}')
        print(f'State: {self.state}')
        
    def close(self):
        return # FIX THIS.

class random_mdp(gym.Env):
    '''This creates a random Markov process (MDP).  In this environment, states are fully observable.'''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_state = 5, n_act = 2, duration = 50):
        super(random_mdp, self).__init__()
        self.n_state = n_state # number of states
        self.n_act = n_act # number of actions
        self.duration = duration # length of each episode
        self.mdp = make_random_mdp(n_state = n_state, n_act = n_act) # the MDP
        self.action_space = spaces.Discrete(n_act) # action space
        self.observation_space = spaces.Discrete(n_state) # observation space
        
    def reset(self):
        self.state = randint(low = 0, high = self.n_state, size = 1) # draw initial state from a uniform distribution
        self.t = 0 # set time step (t) to 0
        obs = self.state # fully observable -> state = observation
        return obs
        
    def step(self, action):
        self.state, reward = self.mdp.sample(action = action, old_state = self.state)
        obs = self.state # tabular and fully observable, so observation = state
        done = self.t == self.duration # finish if have completed a fixed number of time steps (duration)
        info = {'state' : self.state} # the agent isn't allowed to see this
        self.t += 1 # advance time step (t)
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        '''Print environment information on screen.'''
        print(f'Time step: {self.t}')
        print(f'State: {self.state}')
        
    def close(self):
        return # FIX THIS.
    
class gng_reversal(gym.Env):
    '''
    This is a go/no-go reversal learning problem based loosely on the rat experiments of Schoenbaum and colleagues.

    States:
    0, 5, 6, 7 = inter-trial interval (ITI), 1 = odor 1, 2 = odor 2, 3 = sucrose, 4 = quinine
    
    Actions:
    0 = no-go/sit there, 1 = go/try to drink from well
    
    Observations:
    0 = inter-trial interval (ITI), 1 = odor 1, 2 = odor 2, 3 = sucrose, 4 = quinine
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(gng_reversal, self).__init__()        
        # INITIAL TRAINING
        
        # define transition array
        A = np.zeros((8, 8, 2)) # transition array
        A[5, 6, :] = 1 # ITI -> ITI
        A[6, 7, :] = 1 # ITI -> ITI
        A[7, 0, :] = 1 # ITI -> ITI
        A[0, [1, 2], :] = 0.5 # ITI -> odor 1 or odor 2
        A[1, 3, 1] = 1 # odor 1, drink -> sucrose
        A[2, 4, 1] = 1 # odor 2, drink -> quinine
        A[[1, 2], 5, 0] = 1 # trial ends after declining to drink
        A[[3, 4], 5, :] = 1 # trial ends after drinking
        # define observation array
        B = np.zeros((8, 5, 2)) # observation matrix
        B[[0, 5, 6, 7], 0, :] = 1 # ITI
        B[1, 1, :] = 1 # odor 1
        B[2, 2, :] = 1 # odor 2
        B[3, 3, :] = 1 # sucrose
        B[4, 4, :] = 1 # quinine      
        # define reward array
        R = np.zeros((8, 8, 2))
        R[1, 3, 1] = 1 # odor 1, drink -> sucrose
        R[2, 4, 1] = -1 # odor 2, drink -> quinine
        # create POMDP using A and R defined above
        self.pomdp0 = pomdp(A = A, B = B, R = R)
        
        # REVERSAL
        
        # define transition array
        A = np.zeros((8, 8, 2)) # transition array
        A[5, 6, :] = 1 # ITI -> ITI
        A[6, 7, :] = 1 # ITI -> ITI
        A[7, 0, :] = 1 # ITI -> ITI
        A[0, [1, 2], :] = 0.5 # ITI -> odor 1 or odor 2
        A[1, 4, 1] = 1 # odor 1, drink -> quinine
        A[2, 3, 1] = 1 # odor 2, drink -> sucrose
        A[[1, 2], 5, 0] = 1 # trial ends after declining to drink
        A[[3, 4], 5, :] = 1 # trial ends after drinking
        # define observation array
        B = np.zeros((8, 5, 2)) # observation matrix
        B[[0, 5, 6, 7], 0, :] = 1 # ITI
        B[1, 1, :] = 1 # odor 1
        B[2, 2, :] = 1 # odor 2
        B[3, 3, :] = 1 # sucrose
        B[4, 4, :] = 1 # quinine
        # define reward array
        R = np.zeros((8, 8, 2))
        R[1, 4, 1] = -1 # odor 1, drink -> quinine
        R[2, 3, 1] = 1 # odor 2, drink -> sucrose
        # create POMDP using A and R defined above
        self.pomdp1 = pomdp(A = A, B = B, R = R)
        
        # define state and action spaces
        self.action_space = spaces.Discrete(2) # action space
        self.observation_space = spaces.Discrete(5) # observation space
        
    def reset(self):
        self.state = 0
        self.t = 0 # set time step (t) to 0
        self.trial = 0
        obs = 0
        return obs
        
    def step(self, action):
        self.t += 1 # advance time step (t)
        if self.trial < 250:
        #if (self.trial in range(0, 250)) or (self.trial in range(500, 750)):
            obs, reward, self.state = self.pomdp0.sample(action = action, old_state = self.state)
        else:
            obs, reward, self.state = self.pomdp1.sample(action = action, old_state = self.state)
        #done = self.trial == 1000 # finish if have completed a fixed number of trials
        done = self.trial == 500
        info = {'state' : self.state} # the agent isn't allowed to see this
        if self.state == 0:
            self.trial += 1 # note that a trial has been completed
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        '''Print environment information on screen.'''
        print(f'Time step: {self.t}')
        print(f'State: {self.state}')
        
    def close(self):
        return # FIX THIS.

class reinf_deval(gym.Env):
    '''
    This is a two cue Pavlovian reinforcer devaluation, based loosely on Panayi and Kilcross (2018).
    
    States (initial training):
    0, 5, 6, 7 = inter-trial interval (ITI), 1 = sound 1, 2 = sound 2, 3 = food 1, 4 = food 2
    State (devaluation stage):
    0, 3, 4, 5 = inter-trial interval (ITI), 1 = food 1, 2 = nausea
    
    Actions:
    0 = sit there, 1 = food well/eat
    
    Observations:
    0 = inter-trial interval (ITI), 1 = sound 1, 2 = sound 2, 3 = food 1, 4 = food 2, 5 = nausea
    '''
    # FINISH UPDATING
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(reinf_deval, self).__init__()        
        # INITIAL TRAINING        
        
        A = np.zeros((8, 8, 2)) # transition array
        A[5, 6, :] = 1 # ITI -> ITI
        A[6, 7, :] = 1 # ITI -> ITI
        A[7, 0, :] = 1 # ITI -> ITI
        A[0, [1, 2], :] = 0.5 # ITI -> sound 1 or sound 2
        A[1, 3, 1] = 1 # sound 1, go to well -> food 1
        A[2, 4, 1] = 1 # sound 2, go to well -> food 2
        A[[1, 2, 3, 4], 5, 0] = 1 # trial ends after sitting there
        A[[3, 4], 5, 1] = 1 # trial ends after eating
        # define observation array
        B = np.zeros((8, 6, 2)) # observation matrix
        B[[0, 5, 6, 7], 0, :] = 1 # ITI
        B[1, 1, :] = 1 # sound 1
        B[2, 2, :] = 1 # sound 2
        B[3, 3, :] = 1 # food 1
        B[4, 4, :] = 1 # food 2
        # define reward array
        R = np.zeros((8, 8, 2))
        R[3, 5, 1] = 1 # food 1, eat -> ITI
        R[4, 5, 1] = 1 # food 2, eat -> ITI
        # create POMDP using A and R defined above
        self.pomdp0 = pomdp(A = A, B = B, R = R)
        
        # DEVALUATION STAGE
        
        # define transition array
        A = np.zeros((6, 6, 2)) # transition array
        A[3, 4, :] = 1 # ITI -> ITI
        A[4, 5, :] = 1 # ITI -> ITI
        A[5, 0, :] = 1 # ITI -> ITI
        A[0, 1, :] = 1 # ITI -> food 1
        A[1, 2, 1] = 1 # food 1, eat -> nausea
        A[1, 1, 0] = 1 # food 1, don't eat -> food 1
        A[2, 3, :] = 1 # trial ends after nausea  
        # define observation array
        B = np.zeros((6, 6, 2)) # observation matrix
        B[[0, 3, 4, 5], 0, :] = 1 # ITI
        B[1, 3, :] = 1 # food 1
        B[2, 5, :] = 1 # nausea
        # define reward array
        R = np.zeros((6, 6, 2))
        R[1, 2, 1] = -5 # food 1, eat -> nausea
        # create POMDP using A and R defined above
        self.pomdp1 = pomdp(A = A, B = B, R = R)
        
        # define state and action spaces
        self.action_space = spaces.Discrete(2) # action space
        self.observation_space = spaces.Discrete(6) # observation space
        
    def reset(self):
        self.state = 0
        self.t = 0 # set time step (t) to 0
        self.trial = 0
        obs = 0
        return obs
        
    def step(self, action):
        self.t += 1 # advance time step (t)
        if self.trial < 500:
            obs, reward, self.state = self.pomdp0.sample(action = action, old_state = self.state)
        else:
            obs, reward, self.state = self.pomdp1.sample(action = action, old_state = self.state)
        done = self.trial == 510 # finish if have completed a fixed number of trials
        info = {'state' : self.state} # the agent isn't allowed to see this
        if self.state == 0:
            self.trial += 1 # advance trial
        return obs, reward, done, info
    
    def render(self, mode='human', close=False):
        '''Print environment information on screen.'''
        print(f'Time step: {self.t}')
        print(f'State: {self.state}')
        
    def close(self):
        return # FIX THIS.

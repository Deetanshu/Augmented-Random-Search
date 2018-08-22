# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 21:46:55 2018

@author: deept
"""

#imports
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

#Hyperparameters (Page 6)

class Hp():
    
    def __init__(self):
        self.nb_steps = 25000
        self.episode_length = 1000
        self.learning_rate = 0.015
        self.nb_directions = 60
        self.nb_best_directions = 20
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.025
        self.seed = 7
        self.env_name = 'AntBulletEnv-v0'
        #0.015 0.025 60 20
#Normalize the states

class Normalizer():
    
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs) #Counter :D number of inputs vector of zeros
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs) #Numerator for calculating variance
        self.var = np.zeros(nb_inputs) #Variance
    
    def observe(self, x): #call this when new state 
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n # O N L I N E computation of mean
        self.mean_diff += (x - last_mean) * (x - self.mean) # online updated numerator for var calculation
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean #store dat mean bruh
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) /obs_std

#The artificial intelligence which is basically a policy babymaker --> Perceptron
class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size)) #Output number of rows into input num of columns
        #Theta is weight matrix (READ THE DAMN PAPER DEEP WHY ARE YOU LIKE THIS)
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
    
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for i in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r): 
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d #page 6 step 7, step part.
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step #coefficient into step.

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0. # number of actions plays
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        #reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

#training part

def train(env, policy, normalizer, hp):
    
    for step in range(hp.nb_steps):
        
        #Initializing deltas and positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions #Positive direction's rewards
        negative_rewards = [0] * hp.nb_directions #Negative direction's rewards
        
        #+ve rewards 
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta=deltas[k])
        
        #-ve rewards
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta=deltas[k])
        
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        #Sorting the rollouts
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[0:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        #update policy and then print 
        policy.update(rollouts, sigma_r)
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, 'Reward: ', reward_evaluation)
        if reward_evaluation >1000:
            break
        
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'ars_ant')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)    
env = wrappers.Monitor(env, monitor_dir, force = True)
nb_inputs= env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)    
np.savetxt('perceptron.txt', policy.theta, fmt = '%d')
    
    
    
    
    
    
    
    
    
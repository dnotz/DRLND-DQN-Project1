[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree. Project 1: Navigation

## Introduction

This is the first project of the [Deep Reinforcement Learning Nanodegree (DRLND)](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). For this project, we train an agent to navigate (and collect bananas!) in a large, square world. We choose to implement Deep Q Learning. We also implement both [Double](https://arxiv.org/abs/1509.06461) and [Dueling](https://arxiv.org/abs/1511.06581) DQN as optional algorithm extensions and improvements.

![Trained Agent][image1]

The environment is a Udacity adaptation of the Banana Collector from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Environment setup. This repository contains the environment for Linux. If you use a different operating system, please download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. If you downloaded the environment for your operating system, please decompress the file and place it in the `environment` directory of this repository.

3. Installing required dependencies. Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install all dependencies and set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to run this project.

## Instructions

Open the notebook [`Navigation.ipynb`](Navigation.ipynb), choose the 'drlnd' kernel and run all cells to train DQN agents to solve the banana environment!
The notebook imports Python modules for agent and network definition, which can be found in the `agents` and `networks` directories.

## Report
For details on the implemented learning algorithm, including model architecture and hyperparameters, and on the achieved rewards please see the [Report](Report.md).
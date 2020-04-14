[//]: # (Image References)

[image1]: doc/reward_plots.png "Rewards"

# Report of Project 1: Navigation

In the following, we will describe our DQN learning algorithm and its implemented extensions. We also give an overview of the used hyperparameters.
Then, we show and discuss our achieved results.
Finally, we give an outlook on future work.

## Learning Algorithm

Q Learning is a model-free, temporal-difference (TD) reinforcement learning (RL) algorithm.
It tries to learn the optimal action-value function **`Q`**.
In Deep Q Learning, this function is represented as a deep neural network.
The **`Q`** value estimates, or the weights of the neural network, are iteratively updated.
In this update, the TD error, the difference of the current **`Q`** value and the sum of the step reward and the discounted **`Q`** value of the next state is reduced.
Since the DQN training is notoriously unstable, successful agent training usually requires fixed Q targets and a replay buffer.
For more details, please see [this research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

### Double DQN

It has been shown that Deep Q Learning tends to overestimates **`Q`**  values due to noisy estimates. Double DQN solves this problem by using different network weights for optimal action selection and corresponding **`Q`** value estimation. For more details, please read
[this research paper](https://arxiv.org/abs/1509.06461).

### Dueling DQN

In many RL problems the **`Q`** values are mainly determined by the state while the action itself is less important.
For that reason, Dueling DQN uses to network streams to estimate the state value **`V`** and the advantage **`A`** separately and combines them for the **`Q`**  value estimate. For more details, please read
[this research paper](https://arxiv.org/abs/1511.06581).

## Hyperparameters

We train our agents for up to `n_episodes` episodes with up to`max_t` timesteps each. For explroration, we use epsilon decay, where `eps_start` defines the epsilon start value, `eps_end` defines the minimum epsilon value to not let exploration go to zero and `eps_decay` defines the decay factor to reduce epsilon after each episode.
We use the following hyperparameters:

* **`n_episodes = 1000`**
* **`max_t = 5000`**
* **`eps_start = 0.6`**
* **`eps_end = 0.01`**
* **`eps_decay = 0.98`**

For the DQN agents, we have to specify the size of the replay buffer with `BUFFER_SIZE` and the size of the training batches with `BATCH_SIZE`. Furthermore, `GAMMA` is the discount factor for future rewards and `TAU` is the interpolation factor between old and new network weights for the soft update of the target parameters.
The learning rate is given as `LR` and `UPDATE_EVERY` specified how often the network is updated.
We use the following hyperparameters:

* **`BUFFER_SIZE = 200000`**
* **`BATCH_SIZE = 64`**
* **`GAMMA = 0.99`**
* **`TAU = 2e-3`**
* **`LR = 5e-4`**
* **`UPDATE_EVERY = 4`**

We choose a simple neural network architecture consisting of fully connected (linear) layers only. We can specify the number of hidden layers and their corresponding sizes as a Python list.
In our experiments we used:

* **`fc_layer_sizes = [64, 128]`**

Hence, our Standard and Double DQN networks consisted of three layers.
For the Dueling DQN we use the last two specified layers for the parallel network streams.

## Achieved Scores (Rewards)

We have trained Standard, Double and Dueling DQN agents with the described parameters until they achieved an average episodic reward of +13 over 100 consecutive episodes.
Below you can see a plot of the resulting scores (rewards) against the number of episodes trained.

It is difficult to compare the performances of Standard, Double and Dueling DQN since we did not optimize for the other training hyperparameters, which might have a significant performance influence. In particular, the exploration seems to have a high impact on the required amount of training episodes.
Furthermore, from single training runs, no statistically significant results can be deduced. Ideally, one would repeat the same experiment several times with different `seed` values and then compute both mean and standard deviation over the runs. We leave this as an exercise to our readers.

The important take-away here is that each of the agents is able to solve the RL environment and achieve a high average reward!

![Rewards][image1]

## Ideas for Future Work

As mentioned above, we did not optimize the hyperparameters of the training but chose a set that seemed to work well. This is a future work topic.
Furthermore, repeating the same experiment several times, computing the means and standard deviations of the different agents would be great to make statistically significant claims about the differences in the agents' performances.
Finally, another idea for future work is to implement and evaluate further DQN improvements as suggested by Rainbow DQN. For more details, on Rainbow DQN please read [this research paper](https://arxiv.org/abs/1710.02298).

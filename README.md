Deep-Reinforcement Learing Termproject, Class No.0048
==============================================
Algorithm : A3C
----------------



HyperParameters :    
>discount_factor = 0.99   
>no_op_steps = 30   
>threads = 24   
>actor_lr = 1e-3 / np.log10(0.01 * n_episode + 10)   
>critic_lr = 1e-3 / np.log10(0.01 * n_episode + 10)   
>t_max = 20   
   
   
Input demensions :    
>resize(84,84), stack 4 frames, greyscale
   

  


Training Code
----------------------------------
###Comment:
    
    random seed를 지정하니 기존에 학습이 잘되던 환경에서의 양상이 이상하게(특히,Breakout) 돌아감에 따라, 주석처리하였다.
    기존에 존재하던 코드들은 하나의 신경망, 또는 동일한 구조의 global신경망과 local 신경망을 정의하였고, Actor와 Critic의 구문 분할이
    잘 이루어지지 않아 구조를 파악하는데 어려움이 있었다.
    이를 보완하여 Actor와 Critic의 각각의 구문 분할을 진행하였고-프로그램적으로 의미가 있는지는 잘 모르겠다. Matlab에서는 함수화를 자주 사용하는데,
    Python에서도 함수화를 자주 사용한다면 후에 각 파일들을 따로 관리하기 쉬울 것으로 예상된다.- 각각에 대해 신경망을 구현하였다.
    학문적 호기심에 Actor와 Critic의 신경망을 서로 다르게 구성해보았는데, 처음 몇 에피소드에 대한 성능은 개선되는 듯 하다가도 글로벌 신경망이 업데이트 됨에 따라 기존보다 성능이 저하되는 문제점을 발견하였다.
    결국 Actor와 Critic의 신경망은 동일하게 유지하였다.
    또, get_action에서 softmax를 거친 policy tensor에 NaN이 포함됨으로 인해 학습이 도중에 중지되는 문제가 발생하였고, NaN값이 발생할 경우 해당 Policy의 다음 action을 정할 확률을 완전히 동등한 확률로 대체하였다. 사실 직전 policy를 받길 원했으나, multithread 환경에서 이 메모리 버퍼를 구현하는것도, 얼마나 가지고 있어야 하는지에 대해서도 무지하여 어쩔 수 없이 동등한 확률로 완전히 랜덤하게 정하도록 하였다. 이러한 점은 학습 결과에 나쁜쪽으로 가장 큰 영향을 끼쳤을 것으로 판단된다.

```python
# --------------import libs-----------------------
import tensorflow as tf
import numpy as np
import os
import time
import gym
import random
import threading
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
# --------------import libs-----------------------

# --------------global_params---------------------
global n_episode, score_avg, score_max
n_episode, score_avg, score_max = 0, 0, 0
Max_len_episode = 10000

#np.random.seed(5) # random seed, 몇몇 환경에서는 학습이 안되는 부작용이 있음. 


class HyperParam: #Define Hyperparams over Envs
      discount_factor = 0.99
      no_op_steps = 30
      threads = 24
      actor_lr = 1e-3 / np.log10(0.01 * n_episode + 10)
      critic_lr = 1e-3 / np.log10(0.01 * n_episode + 10)
      t_max = 20
      
      
class envlist: # Define learning Env
    env_name = ("BreakoutDeterministic-v4",
                "SpaceInvaders-v0",
                "Pong-v0",
                "Assault-v0",
                "BeamRider-v0")
                
    state_size = ((84, 84, 4),
                  (84, 84, 4),
                  (84, 84, 4),
                  (84, 84, 4),
                  (84, 84, 4))
                  
    action_size = [3,
                   5,
                   5,
                   6,
                   8]

    action_dict = [{0: 1, 1: 2, 2: 3, 3: 4},
                   {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
                   {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
                   {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
                   {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}]

  def environment(n):
      env_name = envlist.env_name[n]
      state_size = envlist.state_size[n]
      action_size = envlist.action_size[n]
      return env_name, state_size, action_size

  
class Actor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.actor_nn()
        self.opt = Adam(HyperParam.actor_lr)

    def actor_nn(self):
        input_state = Input(shape=self.state_size)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_state)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        #conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        flatten = Flatten()(conv)
        fc = Dense(512, activation='relu')(flatten)
        policy = Dense(self.action_size, activation='linear')(fc)
        actor = tf.keras.models.Model(input_state, policy)
        return actor

    def get_action(self, state):
        state = np.float32(state / 255.)
        policy = self.model(state)[0]
        policy = tf.nn.softmax(policy)

        def replacenan(t):
            return tf.where(tf.math.is_nan(t), np.ones(self.action_size)/self.action_size, t)
            # act random select when NaN occurs
        policy = replacenan(policy)
        action = (np.random.choice(self.action_size, 1, p=policy.numpy()))[0]

        return action, policy

    def append_sample(self, states, actions, rewards, sampled_frame, action, reward):
        states.append(sampled_frame)
        act = np.zeros(self.action_size)
        act[action] = 1
        actions.append(act)
        rewards.append(reward)

    def actor_loss(self, actions, policy, advantages):
        action = tf.convert_to_tensor(actions, dtype=tf.float32)
        policy_prob = tf.nn.softmax(policy)
        action_prob = tf.reduce_sum(action * policy_prob, 1, keepdims=True)
        cross_entropy = -tf.math.log(action_prob * 1e-10)
        actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))

        entropy = tf.reduce_sum(policy_prob * tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        actor_loss += 0.01 * entropy
        return actor_loss


class Critic:
    def __init__(self, state_size):
        self.state_size = state_size
        self.model = self.critic_nn()
        self.opt = Adam(HyperParam.critic_lr)

    def critic_nn(self):
        input_state = Input(shape=self.state_size)
        conv = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_state)
        conv = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv)
        #conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)
        flatten = Flatten()(conv)
        fc = Dense(512, activation='relu')(flatten)
        value = Dense(1, activation='linear')(fc)
        critic = tf.keras.models.Model(input_state, value)
        return critic

    def critic_loss(self, discounted_pred, value):
        advantages = discounted_pred - value
        critic_loss = 0.5*tf.reduce_sum(tf.square(advantages))
        return advantages, critic_loss


class Agent:
    def __init__(self, n_env):
        self.n_env = n_env
        self.env_name = environment(n_env)[0]
        self.state_size = environment(n_env)[1]
        self.action_size = environment(n_env)[2]
        self.discount_factor = HyperParam.discount_factor
        self.no_op_steps = HyperParam.no_op_steps
        self.threads = HyperParam.threads
        self.actor_opt = Adam(HyperParam.actor_lr)
        self.critic_opt = Adam(HyperParam.actor_lr)
        self.optimizer = [self.actor_opt, self.critic_opt]

        self.global_Actor = Actor(self.state_size, self.action_size)
        self.global_Critic = Critic(self.state_size)

        self.recoder = tf.summary.create_file_writer('summary/%s' % self.env_name)
        self.model_path_actor = os.path.join(os.getcwd(),
                                             '../../../save_model', 'actor')
        self.model_path_critic = os.path.join(os.getcwd(),
                                              '../../../save_model', 'critic')

    def train(self):

        processor = [Operation(self.action_size, self.state_size,
                               self.global_Actor, self.global_Critic, self.optimizer,
                               self.discount_factor, self.env_name,
                               self.recoder, self.n_env) for i in range(self.threads)]

        for i, Op in enumerate(processor):
            print("Thread_No #{:d}".format(i))
            Op.start()

        while True:
            self.global_Actor.model.save_weights(self.model_path_actor, save_format="tf")
            self.global_Critic.model.save_weights(self.model_path_critic, save_format="tf")
            time.sleep(60 * 10)


class Operation(threading.Thread):
    def __init__(self, action_size, state_size, global_actor, global_critic,
                 optimizer, discount_factor, env_name, recoder, n_env):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.global_Actor = global_actor
        self.global_Critic = global_critic

        self.optimizer = optimizer
        self.discount_factor = discount_factor

        self.states, self.actions, self.rewards = [], [], []
        self.local_actor = Actor(self.state_size, self.action_size)
        self.local_critic = Critic(self.state_size)
        self.env = gym.make(env_name)
        self.recoder = recoder
        self.n_env = n_env

        self.avg_p_max = 0
        self.avg_loss = 0
        self.t_max = HyperParam.t_max
        self.t = 0
        self.action_dict = envlist.action_dict[n_env]

    def draw_tensorboard(self, score, step, e):
        avg_p_max = self.avg_p_max / float(step)
        with self.recoder.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=e)
            tf.summary.scalar('Average Max Prob/Episode', avg_p_max, step=e)
            tf.summary.scalar('Duration/Episode', step, step=e)

    def discounted_pred(self, rewards, flag):
        discounted_pred = np.zeros_like(rewards)
        op_constant = 0

        if not flag:
            last_state = np.float32(self.states[-1] / 255.)
            op_constant = self.local_critic.model(last_state)[-1].numpy()

        for t in reversed(range(0, len(rewards))):
            op_constant = op_constant * self.discount_factor + rewards[t]
            discounted_pred[t] = op_constant
        return discounted_pred

    def loss(self, flag):
        discounted_pred = self.discounted_pred(self.rewards, flag)
        discounted_pred = tf.convert_to_tensor(discounted_pred[:, None],
                                               dtype=tf.float32)
        states = np.zeros((len(self.states), 84, 84, 4))

        for i in range(len(self.states)):
            states[i] = self.states[i]
        states = np.float32(states / 255.)

        policy = self.local_actor.model(states)
        value = self.local_critic.model(states)

        advantages, critic_loss = self.local_critic.critic_loss(discounted_pred, value)
        actor_loss = self.local_actor.actor_loss(self.actions, policy, advantages)

        total_loss = 0.5 * critic_loss + actor_loss

        return total_loss

    def train_model(self, flag):
        global_Actor_params = self.global_Actor.model.trainable_variables
        global_Critic_params = self.global_Critic.model.trainable_variables

        local_actor_params = self.local_actor.model.trainable_variables
        local_critic_params = self.local_critic.model.trainable_variables

        with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
            total_loss = self.loss(flag)

        actor_grads = tape_1.gradient(total_loss, local_actor_params)
        critic_grads = tape_2.gradient(total_loss, local_critic_params)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 40.0)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, 40.0)

        self.optimizer[0].apply_gradients(zip(actor_grads, global_Actor_params))
        self.optimizer[1].apply_gradients(zip(critic_grads, global_Critic_params))
        self.local_actor.model.set_weights(self.global_Actor.model.get_weights())
        self.local_critic.model.set_weights(self.global_Critic.model.get_weights())
        self.states, self.actions, self.rewards = [], [], []

    def run(self):
        global n_episode, score_avg, score_max

        step = 0

        while n_episode < Max_len_episode:
            flag = False
            dead = False

            score, start_life = 0, 5
            observe = self.env.reset()

            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = self.env.step(1)

            state = pre_processing(observe)
            sampled_frame = np.stack([state, state, state, state], axis=2)
            sampled_frame = np.reshape([sampled_frame], (1, 84, 84, 4))

            while not flag:
                step += 1
                self.t += 1

                action, policy = self.local_actor.get_action(sampled_frame)
                real_action = self.action_dict[action]

                if dead:
                    # action0 = retry
                    action, real_action, dead = 0, 1, False
                observe, reward, flag, info = self.env.step(real_action)

                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_sampled_frame = np.append(next_state, sampled_frame[:, :, :, :3], axis=3)

                self.avg_p_max += np.amax(policy.numpy())

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                self.local_actor.append_sample(self.states, self.actions, self.rewards, sampled_frame, action, reward)

                if dead:
                    sampled_frame = np.stack((next_state, next_state,
                                              next_state, next_state), axis=2)
                    sampled_frame = np.reshape([sampled_frame], (1, 84, 84, 4))
                else:
                    sampled_frame = next_sampled_frame

                if self.t >= self.t_max or flag:
                    self.train_model(flag)
                    self.t = 0

                if flag:
                    n_episode += 1
                    score_max = score if score > score_max else score_max
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score

                    log = "n_episode: {:5d} | score : {:4.1f} | ".format(n_episode, score)
                    log += "score max : {:4.1f} | ".format(score_max)
                    log += "score avg : {:.3f}".format(score_avg)
                    print(log)

                    self.draw_tensorboard(score, step, n_episode)

                    self.avg_p_max = 0
                    step = 0


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Forceing device CPU:0

if __name__ == "__main__":
        global_agent = Agent(n_env=0) #You can choose Env by number
        global_agent.train()

```      
        
    
#Result Graph
--------------
###Comments:

    Tensorboard 활용에 대한 지식이 부족하여 다른 코드에 있는 Tensorboard 구문을 그대로 따왔다. 
    
    문제는 여기에 있었는데, score counting 방식이 가장 자연스럽고 직관적이라 생각되지만, 해당 코드에는 이상하게 score/n_episode 으로 지정을 해두었다.
    
    python에 익숙하지도 않고+디버깅에 정신팔린 나머지 조용히 있는 이 변수를 확인하는데 실패하였고, 결국 의도한 바와 다르게 에피소드 당 평균 점수가 되어버리면서, Breakout의 결과가 가장 이상하게 나오는 모양새가 되어버렸다. 
    
    Breakout의 학습 양상은 초반에 저조한 학습률을 보여주다가 에피소드가 축적되어 글로벌 에이전트가 갱신됨에 따라 폭발적으로 증가하는 양상이 반복적으로 나타나는데, 문제는 다른 환경에 비해 이 episode 구간이 길다는데 있었다.   
    
    해서, 다른 결과도 이상하지만 Breakout의 결과가 가장 이상함을 미리 알려둔다.




##SpaceInvader
![A3C_SpaceInvader](https://user-images.githubusercontent.com/108215235/175811703-ea263cbe-f023-40a8-8a99-84977a624eb8.PNG)   
Episode가 50도 채 되기 전에 어느정도 학습이 포화상태에 도달한 양상을 나타낸다.


##Pong
![A3C_Pong](https://user-images.githubusercontent.com/108215235/175811705-7e11a9bf-eda2-40cd-8eec-9c8b9132ec8a.PNG)   
시간이 지나도 학습에 실패하는 모습을 보인다. HyperParams를 수정하여도 똑같은 것을 보아, 신경망이나 구현한 알고리즘 자체의 문제인듯 하다.

##BreakoutDeterministicV4
![A3C_BreakoutDeterV4](https://user-images.githubusercontent.com/108215235/175811707-876231ee-066f-466c-bd92-cb1df3df169c.PNG)   
앞서 말한대로, 가장 왜곡된 양상을 나타내고 있다.(4000 전까지 Score이 증가하다가, 이후는 학습이 포화상태에 도달하였다.)


##BeamRider
![A3C_BeamRider](https://user-images.githubusercontent.com/108215235/175811708-03c88cf5-df06-4f84-be99-7ec4e1dc5c18.PNG)   
HyperParams를 수정하여도 결과가 크게 달라지지 않는 것을 보아, 학습에 실패한 것으로 추정된다.


##Assult
![A3C_Assult](https://user-images.githubusercontent.com/108215235/175811709-d49f6f14-087f-4dbe-83d4-82bcd13fb420.PNG)
episode 초반에 빠르게 포화되어 에피소드에 걸친 평균이 상대적으로 낮아진 상태로 포화하고있다.

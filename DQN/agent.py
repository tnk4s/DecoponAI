import random
from collections import deque

import torch
import numpy as np

from DQN.model import MatchaNet

CAPACITY = 300000
BATCH_SIZE = 512
GAMMA = 0.9
SAVE_FREQUENCY = 5e5


class Matcha:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.memory = deque(maxlen=CAPACITY)
        self.batch_size = BATCH_SIZE

        self.gamma = GAMMA

        # モデルのインスタンス
        #self.net = MatchaNet(self.state_dim, self.action_dim).float()#ORG
        self.net = MatchaNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1.0#最初は1だった
        #self.exploration_rate_decay = 0.999999934 #ep1M
        self.exploration_rate_decay = 0.999998 # ep10k
        self.exploration_rate_min = 0.01
        self.curr_step = 0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.save_every = SAVE_FREQUENCY  # モデルを保存するまでの実験ステップの数

        self.burnin = 1e4  # 経験を訓練させるために最低限必要なステップ数
        self.learn_every = 3  # Q_onlineを更新するタイミングを示すステップ数
        self.sync_every = 1e4  # Q_target & Q_onlineを同期させるタイミングを示すステップ数

    def act(self, state):
        """
        状態が与えられたとき、ε-greedy法に従って行動を選択

        Inputs:
            state(LazyFrame):現在の状態における一つの観測オブジェクトで、(state_dim)次元となる
        Outputs:
            action_idx (int): AIが取る行動を示す整数値
        """
        # 探索（EXPLORE）
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 活用（EXPLOIT）
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state.copy())
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()#目的座標を返す

        # exploration_rateを減衰させます
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # ステップを+1します
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        経験をself.memory (replay buffer)に保存

        Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        メモリから経験のバッチを取得
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
                self.save_dir / f"matcha_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )

        save_path = (
                self.save_dir / f"online_matcha_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.online.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MatchaNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        """経験のデータのバッチで、オンラインに行動価値関数(Q)を更新"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

            # メモリからサンプリング
        state, next_state, action, reward, done = self.recall()

        # TD Estimateの取得
        td_est = self.td_estimate(state, action)

        # TD Targetの取得
        td_tgt = self.td_target(reward, next_state, done)

        # 損失をQ_onlineに逆伝播させる
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

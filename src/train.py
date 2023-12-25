import datetime
from pathlib import Path

import torch.cuda
import gym
from gym.wrappers import FrameStack
from tqdm import tqdm

import src  # 消さないで
from DQN.agent import Matcha
from DQN.utils import SkipFrame, MetricLogger, ResizeObservation, GrayScaleObservation

# 環境の作成
env = gym.make("decoponEnv-v0")

#env = SkipFrame(env, skip=4)
#env = GrayScaleObservation(env)
#env = ResizeObservation(env, shape=84)
#env = FrameStack(env, num_stack=4)

#環境の初期化
env.reset()

#next_state, reward, done, _, info = env.step(action=0)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("trained_models") #/train_ep1M_model")
log_dir = Path("log") #/train_ep1M")
# save_dir.mkdir(parents=True)

#matcha = Matcha(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)#ORG
matcha = Matcha(state_dim=62, action_dim=14, save_dir=save_dir)

logger = MetricLogger(log_dir)

# episodes = 10000000
episodes = 20000

for e in tqdm(range(episodes)):

    state, _ = env.reset()

    # ゲーム開始！
    while True:

        # 現在の状態に対するエージェントの行動を決める
        action = matcha.act(state)

        # エージェントが行動を実行
        next_state, reward, done, _, info = env.step(action)

        # 記憶
        matcha.cache(state, next_state, action, reward, done)

        # 訓練
        q, loss = matcha.learn()

        # ログ保存
        logger.log_step(reward, loss, q)

        # 状態の更新
        state = next_state

        # ゲーム画面の描画
        #env.render()

        # ゲームが終了したかどうかを確認
        # if done or info["flag_get"]:  # ゲームの終了条件によってはこれ
        if done:
            matcha.save()
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=matcha.exploration_rate, step=matcha.curr_step)

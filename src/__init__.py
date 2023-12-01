import gym
from gym.envs.registration import register
from game_env import RemotePlayer

register(
    id='decoponEnv-v0',
    entry_point='src.game_env:DecoponGameEnv'
)

env = gym.make("decoponEnv")
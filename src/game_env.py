import random
from collections import namedtuple
from typing import Tuple

import pygame
import pymunk

from decopon.controller import Controller, Human, RemotePlayer

#==以下，追加分==

import gym
from gym import spaces
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from main import Game
from main import Polygon, Polygons, HEIGHT, WIDTH, TIMELIMIT

class DecoponGameEnv(gym.Env, Game):
    def __init__(self):#, controller: Controller):
        self.controller = RemotePlayer()
        super().__init__(self.controller)

        # アクション空間を定義
        #self.action_space = spaces.Discrete(3)  # 左，右，投下
        self.action_space = spaces.Discrete(1)   #投下座標のみ
        #self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)# 観測空間を定義 84×84の画像
        self.observation_space = spaces.Box(low=-1, high=HEIGHT)# 観測空間を定義 仮
        
        self.limit_y = 200#ゲームオーバーの高さ
        self.observation_data = None # 画像になるか，ただの数列
        #self.reward_range = [- pow((HEIGHT - self.limit_y), 2), 500.]
        self.reward_range = [-1. , 500.]

        self.exit_flag = False
        self.last_score = 0
        self.episode_start_time = 0  # 残り時間

        self.max_y = HEIGHT#球の最も高い座標

    def reset(self):
        self.__init__()
        self.countOverflow = 0
        self.episode_start_time = pygame.time.get_ticks()  # エピソード開始時間を記録
        return self.observe(), {}
    
    def run(self):#オーバーライド
        self.step()

    def step(self, destination):
        self.controller.set_destination(destination)
        self.update_game_status()

        observation = self.observe()
        reward = self.get_reward()
        done = self.is_done()
        info = {}

        return observation, reward, done, {}, info

    def update_game_status(self):# 描画関係を取り除き，ゲーム状態の更新
        act_not_flag = True

        while act_not_flag or self.controller.get_wait_counter() > 0:
            seconds = (pygame.time.get_ticks() - self.start_time) // 1000
            #print(pygame.time.get_ticks(), self.start_time)

            if self.isGameOver or seconds > TIMELIMIT:
                print("==GAME OVER==")
                print("self.isGameOver:", self.isGameOver)
                print("seconds > TIMELIMIT:", seconds > TIMELIMIT)
                print("score:", self.score)
                self.exit_flag = True#exit(0)
                break

            if self.check_event(pygame.QUIT):
                return
            if self.check_overflow():
                self.countOverflow += 1
            
            isLeft, isRight, isDrop = self.controller.update(self.indicator.centerx)#コントローラのアクションを取得
            #print("now_act:", isLeft, isRight, isDrop)

            if isLeft:
                for _ in range(self.controller.get_move_step(self.indicator.centerx)):
                    self.indicator.centerx -= 3
                #self.indicator.centerx -= int(self.controller.get_move_step(self.indicator.centerx) * 3)
            elif isRight:
                for _ in range(self.controller.get_move_step(self.indicator.centerx)):
                    self.indicator.centerx += 3
                #self.indicator.centerx += int(self.controller.get_move_step(self.indicator.centerx) * 3)
            elif isDrop and pygame.time.get_ticks() - self.drop_ticks > 500 and not self.check_overflow():
                self.create_poly(self.indicator.centerx, self.indicator.topleft[1], self.current)
                self.drop_ticks = pygame.time.get_ticks()
                self.current = self.next
                self.next = random.randint(0, 4)
                self.countOverflow = 0
                act_not_flag = False
            
            if self.indicator.centerx < 65:
                self.indicator.centerx = WIDTH - 65
            if self.indicator.centerx > WIDTH - 65:
                self.indicator.centerx = 65

                    
            if self.countOverflow > 200:
                self.isGameOver = True
            
            self.render()
            for _ in range(60):#120倍速？
                self.space.step(1/30)
            self.fps(600)



    
    def render(self, mode='human'): # run()のうち，描画関係のみこちらに移植
        self.window.fill((89, 178, 36))
        pygame.draw.rect(self.window, (255, 255, 255), self.indicator)

        poly = Polygons[self.current]
        pygame.draw.circle(
            self.window, poly.color, (self.indicator.centerx, self.indicator.topleft[1]), poly.radius
        )
        poly = Polygons[self.next]
        pygame.draw.circle(self.window, poly.color, (WIDTH - 60, 60), poly.radius)

        for poly in self.poly:
            pygame.draw.circle(
                self.window, poly.color, (int(poly.body.position.x), int(poly.body.position.y)), poly.radius
            )

        self.draw_walls()

        score_text = self.font.render(f"スコア: {self.score}", True, (255, 255, 255))
        score_position = (10, 10)
        self.window.blit(score_text, score_position)

        # スコアと残り時間を描画
        score_text = self.font.render(f"スコア: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        elapsed_time = (pygame.time.get_ticks() - self.episode_start_time) // 1000  # 経過時間を計算
        time_text = self.font.render(f"残り時間: {TIMELIMIT - elapsed_time}", True, (255, 255, 255))
        self.window.blit(time_text, (10, 30))
        """
        text = self.font.render(f"残り時間: {TIMELIMIT - seconds}", True, (255, 255, 255))
        position = (10, 30)
        self.window.blit(text, position)
        """

        text = self.font.render("シンカ", True, (255, 255, 255))
        position = (10, 50)
        self.window.blit(text, position)

        for i, poly in enumerate(self.progress):
            pygame.draw.rect(self.window, Polygons[i].color, poly)

        pygame.display.update()
        


    def close(self):
        pygame.quit()
    
    def get_reward(self):#報酬
        reward = 0
        diff_score = self.score - self.last_score
        if diff_score > 0:
            reward = diff_score
            #reward = reward * ((self.max_y - self.limit_y) / (HEIGHT - self.limit_y))
        else:
            reward = -0.1
            # if (self.max_y - self.limit_y) != 0:
            #     reward = - pow((HEIGHT - self.limit_y) / (self.max_y - self.limit_y), 2)
            # else:
            #     reward = - pow((HEIGHT - self.limit_y), 2)
        self.last_score = self.score

        return reward
    
    def observe(self):
        # 今の玉と次の玉と上から10個の玉の位置を返す．
        observation = []
        poly = Polygons[self.current]#今の
        observation.append(poly.index)
        poly = Polygons[self.next]#次の
        observation.append(poly.index)

        all_poly_data = []
        for poly in self.poly:
            tmp = [int(poly.index), int(poly.body.position.x), int(poly.body.position.y)]
            all_poly_data.append(tmp)
        
        all_poly_sorted = sorted(all_poly_data, reverse=False, key=lambda x: x[2])#2番目の要素=y座標でソート
        counter = 0
        for s_poly in all_poly_sorted:
            observation.append(s_poly[0])#index
            observation.append(s_poly[1])#x
            observation.append(s_poly[2])#y
            counter += 1
            if counter == 20:
                break
        if counter < 20:
            for _ in range(20 - counter):
                observation.append(int(-1))#index
                observation.append(int(-1))#x
                observation.append(int(-1))#y

        self.max_y = observation[4]
        observation = np.array(observation, dtype = np.float32)
        #print("observation:", observation)
        #print("observation.shape:" ,observation.shape)# 2 + 3*10 = 32?
        return observation
    
    def is_done(self):
        """ゲームが終了したかどうかを判定"""
        # タイムリミット
        if (pygame.time.get_ticks() - self.start_time) // 1000 > TIMELIMIT:
            return True
        
        return self.exit_flag



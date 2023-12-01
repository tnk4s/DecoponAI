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

import torch.nn as nn

class AiDrive(Game):
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.my_brain = MyBrain(62, 350-14).float()
        cpt = torch.load("./trained_models/online_matcha_net_0_20231130_0.chkpt")
        stdict_m = cpt["model"]
        self.my_brain.model.load_state_dict(stdict_m)
        self.controller = RemotePlayer()
        super().__init__(self.controller)
        self.exit_flag = False
        self.episode_start_time = 0  # 残り時間
    
    def run(self):
        drop_wait_max = 60 * 1 # 適当
        while True:
            if self.exit_flag:
                break
            seconds = (pygame.time.get_ticks() - self.start_time) // 1000

            #============================================
            # 現在の状態に対するエージェントの行動を決める
            observation = self.observe()
            destination = self.decide_auto_action(observation)
            self.controller.set_destination(destination)
            #============================================
            act_not_flag = True
            drop_wait_max = 60 * 1 # 適当

            while act_not_flag:
                #============================================
                # 現在の状態に対するエージェントの行動を決める
                observation = self.observe()
                destination = self.decide_auto_action(observation)
                self.controller.set_destination(destination)
                #============================================


                seconds = (pygame.time.get_ticks() - self.start_time) // 1000
                #seconds = (self.accel_time - self.start_time) // 1000
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
                    self.indicator.centerx -= 3
                elif isRight:
                    self.indicator.centerx += 3
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
                self.space.step(1 / 60)
                #self.accel_time += 1000/60.0
                self.fps(60)
            
            if not self.exit_flag:
                for _ in range(drop_wait_max):# 落ちるまで待つ?
                    seconds = (pygame.time.get_ticks() - self.start_time) // 1000
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
                    
                    if self.countOverflow > 200:
                        self.isGameOver = True
                    
                    self.render()
                    self.space.step(1 / 60)
                    #self.accel_time += 1000/60.0
                    self.fps(60)
    
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

        self.space.step(1 / 60)
        pygame.display.update()
        self.fps(60)
    
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
            if counter == 10:
                break
        if counter < 10:
            for _ in range(10 - counter):
                observation.append(int(-1))#index
                observation.append(int(-1))#x
                observation.append(int(-1))#y

        observation = np.array(observation, dtype = np.float32)
        #print("observation:", observation)
        #print("observation.shape:" ,observation.shape)# 2 + 3*10 = 32?
        return observation
    
    def decide_auto_action(self, observation):
        self.my_brain.eval()  # ネットワークを推論モードに切り替える
        with torch.no_grad():
            observation = observation.__array__()
            if self.use_cuda:
                observation = torch.tensor(observation).cuda()
            else:
                observation = torch.tensor(observation.copy())
            observation = observation.unsqueeze(0)

            res = self.my_brain(observation)#.max(1)[1].view(1, 1)
            res = np.argmax(res, axis=1).item()#目的座標を返す
            #print("decide_auto_action:", res)
        return res

class MyBrain(nn.Module):
    def __init__(self, input_dim = 32, output_dim = 350):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 160),
            nn.ReLU(),
            nn.Linear(160, output_dim)
        )

    def forward(self, input):
        input = input.to(torch.float)
        return self.model(input)


if __name__ == "__main__":
    # AIでやる
    AiDrive().run()
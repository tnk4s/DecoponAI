from abc import ABC, abstractmethod
from typing import Tuple

import pygame
import numpy as np


class Controller(ABC):
    @abstractmethod
    def update(self) -> Tuple[bool, bool, bool]:
        return True, True, True


class Human(Controller):
    def __init__(self) -> None:
        super().__init__()

    def update(self) -> Tuple[bool, bool, bool]:
        pressedKeys = pygame.key.get_pressed()
        return pressedKeys[pygame.K_LEFT], pressedKeys[pygame.K_RIGHT], pressedKeys[pygame.K_SPACE]

class RemotePlayer(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.destination = 0
        self.wait_counter = 0

    def update(self, indicator_centerx) -> Tuple[bool, bool, bool]:
        if self.wait_counter == 0:
            if abs(indicator_centerx - self.destination) <=2:# 投下
                self.wait_counter = int((60/120) * 2.0)
                return (False, False, True)

            if (self.destination - indicator_centerx) < 0:#左
                return (True, False, False)

            return (False, True, False)# 右

        else:# 停止
            self.wait_counter = self.wait_counter - 1
            return (False, False, False)
        

    def set_destination(self, destination):
        #print("set_destination", destination)
        self.destination = (destination * 25 + 13) + 65
    
    def get_wait_counter(self):
        return self.wait_counter

    def get_move_step(self, indicator_centerx):
        return int(abs(indicator_centerx - self.destination) / 3)


import torch
from DQN.model import MatchaNet

class AiDrive(Controller):
    def __init__(self) -> None:
        super().__init__()

        self.state_dim = 62
        self.action_dim = 14
        self.net = MatchaNet(self.state_dim, self.action_dim).float()
        cpt = torch.load("./trained_models/matcha_net_1_out14_5.chkpt")
        stdict_m = cpt["model"]
        self.net.load_state_dict(stdict_m)

        self.destination = 0
        self.setted_flag = False
        self.wait_counter = 0

    def update(self, data_pack) -> Tuple[bool, bool, bool]:
        indicator_centerx, Polygons, current_id, next_id, poly = data_pack
        observation = self.make_observation(Polygons, current_id, next_id, poly)
        self.net.eval()  # ネットワークを推論モードに切り替える
        with torch.no_grad():
            observation = observation.__array__()
            observation = torch.tensor(observation.copy())
            observation = observation.unsqueeze(0)

            res = self.net(observation, "online")#.max(1)[1].view(1, 1)
            res = np.argmax(res, axis=1).item()#目的座標を返す

        if self.setted_flag == False:
            self.set_destination(res)
        
        if self.wait_counter == 0:
            
            if abs(indicator_centerx - self.destination) <=2:# 投下
                self.wait_counter = int(60 * 2.0)
                self.setted_flag = False
                return (False, False, True)

            if (self.destination - indicator_centerx) < 0:#左
                return (True, False, False)

            return (False, True, False)# 右

        else:# 停止
            self.wait_counter = self.wait_counter - 1
            return (False, False, False)
        

    def set_destination(self, destination):
        #print("set_destination", destination)
        self.setted_flag = True
        self.destination = (destination * 25  + 13 ) + 65
    
    def make_observation(self, Polygons, current_id, next_id, poly):
        # 今の玉と次の玉と上から10個の玉の位置を返す．
        observation = []
        tmp_poly = Polygons[current_id]#今の
        observation.append(tmp_poly.index)
        tmp_poly = Polygons[next_id]#次の
        observation.append(tmp_poly.index)

        all_poly_data = []
        for tmp_poly in poly:
            tmp = [int(tmp_poly.index), int(tmp_poly.body.position.x), int(tmp_poly.body.position.y)]
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

        observation = np.array(observation, dtype = np.float32)
        return observation


class HybridDrive(Controller):
    def __init__(self) -> None:
        super().__init__()

        self.state_dim = 62
        self.action_dim = 14
        self.net = MatchaNet(self.state_dim, self.action_dim).float()
        cpt = torch.load("./trained_models/matcha_net_1_out14_5.chkpt")
        stdict_m = cpt["model"]
        self.net.load_state_dict(stdict_m)

        self.destination = 0
        self.setted_flag = False
        self.wait_counter = 0

        self.must_flag = False
        self.good_drop_pos = -1
        self.dropping_id = -1
        self.must_set_count = 0

    def update(self, data_pack) -> Tuple[bool, bool, bool]:
        indicator_centerx, Polygons, current_id, next_id, poly = data_pack
        observation = self.make_observation(Polygons, current_id, next_id, poly)
        self.net.eval()  # ネットワークを推論モードに切り替える
        with torch.no_grad():
            observation = observation.__array__()
            observation = torch.tensor(observation.copy())
            observation = observation.unsqueeze(0)

            res = self.net(observation, "online")#.max(1)[1].view(1, 1)
            res = np.argmax(res, axis=1).item()#目的座標を返す

        if self.setted_flag == False:
            self.set_destination(res)
        
        if self.wait_counter == 0:
            if abs(indicator_centerx - self.destination) <=2:# 投下
                self.wait_counter = int(60 * 2.0)
                self.setted_flag = False
                return (False, False, True)

            if (self.destination - indicator_centerx) < 0:#左
                return (True, False, False)

            return (False, True, False)# 右
        else:# 停止
            self.wait_counter = self.wait_counter - 1
            return (False, False, False)
        

    def set_destination(self, destination):
        #print("set_destination", destination)
        self.setted_flag = True
        self.destination = (destination * 25  + 13 ) + 65
        if self.must_flag:
            self.destination = self.good_drop_pos
            #print("SET_GOOD_DROP_POS:", self.destination)
    
    def make_observation(self, Polygons, current_id, next_id, poly):
        self.must_flag = False
        self.good_drop_pos = -1
        highest_x = 0
        highest_y = 0
        # 今の玉と次の玉と上から10個の玉の位置を返す．
        observation = []
        tmp_poly = Polygons[current_id]#今の
        observation.append(tmp_poly.index)
        self.dropping_id = tmp_poly.index

        tmp_poly = Polygons[next_id]#次の
        observation.append(tmp_poly.index)

        all_poly_data = []
        for tmp_poly in poly:
            tmp = [int(tmp_poly.index), int(tmp_poly.body.position.x), int(tmp_poly.body.position.y)]
            all_poly_data.append(tmp)
        
        all_poly_sorted = sorted(all_poly_data, reverse=False, key=lambda x: x[2])#2番目の要素=y座標でソート
        counter = 0
        for s_poly in all_poly_sorted:
            observation.append(s_poly[0])#index
            observation.append(s_poly[1])#x
            observation.append(s_poly[2])#y

            if counter == 0:
                highest_x = s_poly[1]
                highest_y = s_poly[2]
                #print("highest_y:", highest_y)
            counter += 1
            if counter == 20:
                break
            
            if s_poly[0] == self.dropping_id and (self.dropping_id <= 4):# 強制誘導制御用
                if counter < 8 and self.must_flag == False:
                    self.must_flag = True
                    self.good_drop_pos = s_poly[1]#落とすべき座標を記録
                    if highest_y > 280:
                        self.must_flag = False
                        #print("!!CANCEL self.good_drop_pos!!")

        if counter < 20:
            for _ in range(20 - counter):
                observation.append(int(-1))#index
                observation.append(int(-1))#x
                observation.append(int(-1))#y

        observation = np.array(observation, dtype = np.float32)
        return observation

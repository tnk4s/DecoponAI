from abc import ABC, abstractmethod
from typing import Tuple

import pygame


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
                self.wait_counter = int(60 * 1.5)
                return (False, False, True)

            if (self.destination - indicator_centerx) < 0:#左
                return (True, False, False)

            return (False, True, False)# 右

        else:# 停止
            self.wait_counter = self.wait_counter - 1
            return (False, False, False)
        

    def set_destination(self, destination):
        #print("set_destination", destination)
        self.destination = destination + 65 + 7
    
    def get_wait_counter(self):
        return self.wait_counter
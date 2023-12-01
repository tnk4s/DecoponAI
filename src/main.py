import random
from collections import namedtuple
from typing import Tuple

import pygame
import pymunk

from decopon.controller import Controller, Human, AiDrive

Polygon = namedtuple("Polygon", ["mass", "radius", "color", "score", "index"])
Polygons = [
    Polygon(1, 13, (255, 0, 127), 0, 0),
    Polygon(2, 17, (255, 0, 255), 1, 1),
    Polygon(3, 24, (127, 0, 255), 3, 2),
    Polygon(4, 28, (0, 0, 255), 6, 3),
    Polygon(5, 35, (0, 127, 255), 10, 4),
    Polygon(6, 46, (0, 255, 255), 15, 5),
    Polygon(7, 53, (0, 255, 127), 21, 6),
    Polygon(8, 65, (0, 255, 0), 28, 7),
    Polygon(9, 73, (127, 255, 0), 36, 8),
    Polygon(10, 90, (255, 255, 0), 45, 9),
    Polygon(11, 100, (255, 127, 0), 55, 10),
]

pygame.init()
pygame.display.set_caption("Decopon")

HEIGHT, WIDTH = 640, 480
TIMELIMIT = 1000


class Game:
    def __init__(self, controller: Controller):
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.fps = pygame.time.Clock().tick
        self.controller = controller

        self.space = pymunk.Space()
        self.space.gravity = (0, 1000)
        collision_handler = self.space.add_collision_handler(1, 1)
        collision_handler.begin = self.merge

        self.walls = []
        self.create_walls()

        self.indicator = pygame.Rect(WIDTH / 2, 100, 3, HEIGHT - 100)

        self.poly = []

        self.drop_ticks = pygame.time.get_ticks()

        self.current = random.randint(0, 4)
        self.next = random.randint(0, 4)

        self.font = pygame.font.Font("resources/BestTen-DOT.otf", 16)
        self.score = 0

        self.isGameOver = False
        self.countOverflow = 0

        self.progress = [pygame.Rect(10 + i * 20, 70, 20, 20) for i in range(11)]

    def merge(self, polys, space, _):
        p0, p1 = polys.shapes
        if p0 not in self.poly or p1 not in self.poly:# エラー回避用に増設
            return True

        if p0.index == 10 and p1.index == 10:
            self.poly.remove(p0)
            self.poly.remove(p1)
            space.remove(p0, p0.body, p1, p1.body)
            self.score += 100 #double
            return False

        if p0.index == p1.index:
            self.score += Polygons[p0.index].score
            x = (p0.body.position.x + p1.body.position.x) / 2
            y = (p0.body.position.y + p1.body.position.y) / 2
            self.create_poly(x, y, p1.index + 1)
            self.poly.remove(p0)
            self.poly.remove(p1)
            space.remove(p0, p0.body, p1, p1.body)

        return True

    def create_walls(self):
        floor = pymunk.Segment(self.space.static_body, (50, HEIGHT - 10), (WIDTH - 50, HEIGHT - 10), 10)
        floor.friction = 0.8
        floor.elasticity = 0.8

        self.walls.append(floor)
        self.space.add(floor)

        left = pymunk.Segment(self.space.static_body, (50, 200), (50, HEIGHT), 10)
        left.friction = 1.0
        left.elasticity = 0.95
        self.walls.append(left)
        self.space.add(left)

        right = pymunk.Segment(self.space.static_body, (WIDTH - 50, 200), (WIDTH - 50, HEIGHT), 10)
        right.friction = 1.0
        right.elasticity = 0.95
        self.walls.append(right)
        self.space.add(right)

        self.start_time = pygame.time.get_ticks()

    def check_event(self, event):
        for e in pygame.event.get():
            if e.type == event:
                return True
        return False

    def create_poly(self, x: int, y: int, idx: int):
        poly = Polygons[idx]

        mass = poly.mass
        radius = poly.radius
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        body.radius = radius
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        shape.elasticity = 0.4
        shape.collision_type = 1
        shape.index = poly.index
        shape.color = poly.color
        self.space.add(body, shape)
        self.poly.append(shape)

    def draw_walls(self):
        pygame.draw.line(self.window, (240, 190, 153), (40, 200), (WIDTH - 40, 200), 3)
        for wall in self.walls:
            p1 = int(wall.a[0]), int(wall.a[1])
            p2 = int(wall.b[0]), int(wall.b[1])
            pygame.draw.line(self.window, (240, 190, 153), p1, p2, 20)

    def check_overflow(self):
        for poly in self.poly:
            if poly.body.position.y - (poly.radius) < 200:
                return True
        return False

    def run(self):
        while True:
            seconds = (pygame.time.get_ticks() - self.start_time) // 1000

            if self.isGameOver or seconds > TIMELIMIT:
                print(self.score)
                exit(0)

            if self.check_event(pygame.QUIT):
                break
            if self.check_overflow():
                self.countOverflow += 1

            #isLeft, isRight, isDrop = self.controller.update()
            isLeft, isRight, isDrop = self.controller.update(self.indicator.centerx, Polygons, self.current, self.next, self.poly)

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

            if self.indicator.centerx < 65:
                self.indicator.centerx = WIDTH - 65
            if self.indicator.centerx > WIDTH - 65:
                self.indicator.centerx = 65

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

            text = self.font.render(f"残り時間: {TIMELIMIT - seconds}", True, (255, 255, 255))
            position = (10, 30)
            self.window.blit(text, position)

            text = self.font.render("シンカ", True, (255, 255, 255))
            position = (10, 50)
            self.window.blit(text, position)

            for i, poly in enumerate(self.progress):
                pygame.draw.rect(self.window, Polygons[i].color, poly)

            self.space.step(1 / 60)
            pygame.display.update()
            self.fps(60)

            if self.countOverflow > 200:
                self.isGameOver = True


class RandomPlayer(Controller):
    def __init__(self) -> None:
        super().__init__()

    def update(self) -> Tuple[bool, bool, bool]:
        return tuple(random.choice([True, False]) for _ in range(3))

if __name__ == "__main__":
    # 人間でプレイしたいとき
    #Game(Human()).run()
    # AIでやる場合
    Game(AiDrive()).run()
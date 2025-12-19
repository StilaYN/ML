# env.py
import random

STATE_DIM = 13
ACTION_SCALE = 8.0

class RacingEnv:
    def __init__(self, seed=None, max_steps=2000):
        if seed is not None:
            random.seed(seed)
        self.W, self.H = 400, 600
        self.ROAD_LEFT = self.W // 2 - 150
        self.ROAD_RIGHT = self.W // 2 + 150
        self.max_steps = max_steps
        self.STATE_DIM = STATE_DIM
        self.reset()

    def reset(self):
        self.player_x = self.W // 2
        self.enemies = []
        self.enemy_speed = 7.0
        self.spawn_timer = 0
        self.steps = 0
        self.score = 0
        return self._state()

    def _state(self):
        px = (self.player_x - self.W / 2) / (self.W / 2)
        player_y = self.H - 80  # позиция игрока по Y

        dist_left = (self.player_x - self.ROAD_LEFT) / 150
        dist_right = (self.ROAD_RIGHT - self.player_x) / 150

        # Рассчитываем расстояние до каждого врага и сортируем по близости
        enemies_with_dist = []
        for ex, ey, passed in self.enemies:
            if ey < self.H - 50:  # только враги перед игроком (не прошедшие)
                dx = self.player_x - ex
                dy = player_y - ey
                dist = (dx**2 + dy**2) ** 0.5
                enemies_with_dist.append((ex, ey, dist))

        # Сортируем по расстоянию (ближайшие первыми)
        enemies_with_dist.sort(key=lambda x: x[2])

        e1x = e1y = e2x = e2y = 0.0
        if len(enemies_with_dist) > 0:
            ex, ey, _ = enemies_with_dist[0]
            e1x = (ex - self.W / 2) / (self.W / 2)
            e1y = (player_y - ey) / self.H  # нормализуем по высоте
        if len(enemies_with_dist) > 1:
            ex, ey, _ = enemies_with_dist[1]
            e2x = (ex - self.W / 2) / (self.W / 2)
            e2y = (player_y - ey) / self.H

        danger_density = min(len(self.enemies) / 5.0, 1.0)

        return [
            px,
            dist_left - dist_right,
            e1x, e1y, e2x, e2y,
            danger_density,
            self.steps / 1000.0,
            self.enemy_speed / 10.0,
            abs(px),
            float(dist_left < 0.25),
            float(dist_right < 0.25),
            min(len(enemies_with_dist) / 3.0, 1.0)
        ]

    def step(self, action):
        move = action * ACTION_SCALE
        self.player_x = max(self.ROAD_LEFT + 20, min(self.ROAD_RIGHT - 20, self.player_x + move))

        self.spawn_timer += 1
        spawn_freq = max(20, 50 - self.score * 2)
        if self.spawn_timer >= spawn_freq:
            for _ in range(2):
                if random.random() < 0.6:
                    x = random.uniform(self.ROAD_LEFT + 25, self.ROAD_RIGHT - 25)
                    self.enemies.append([x, -60, False])
            self.spawn_timer = 0

        reward = 0.1  # small survival reward
        center = (self.ROAD_LEFT + self.ROAD_RIGHT) / 2

        for e in self.enemies[:]:
            e[1] += self.enemy_speed
            if not e[2] and e[1] > self.H - 50:
                e[2] = True
                self.score += 1
                reward += 0.2  # was 15 → now 0.2
            if e[1] > self.H:
                self.enemies.remove(e)

        if self.score > 10:
            self.enemy_speed = 7.0 + (self.score - 10) * 0.1

        self.steps += 1

        # Collision check
        collision = any(
            abs(self.player_x - ex) < 35 and abs((self.H - 80) - ey) < 50
            for ex, ey, _ in self.enemies
        )

        if collision:
            reward = -1.0  # normalized penalty
        else:
            # Mild penalty for hugging edges
            if abs(self.player_x - center) > 130:
                reward -= 0.05

        # Penalize inactivity
        if abs(move) < 0.5:
            reward -= 0.02

        done = collision or self.steps >= self.max_steps
        if done and not collision:
            reward += 0.5  # bonus for surviving

        return self._state(), reward, done, {"score": self.score}
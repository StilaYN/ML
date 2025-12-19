import argparse
import sys
import torch
import pygame
import numpy as np

from env import RacingEnv
from model import GaussianPolicy

# ==============================
# Args
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to trained policy (.pth)"
)
parser.add_argument(
    "--speed",
    type=int,
    default=60,
    help="Game speed (FPS)"
)
args = parser.parse_args()

# ==============================
# Pygame init
# ==============================
pygame.init()

WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Racing - Fast Reaction")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 32)
small_font = pygame.font.SysFont(None, 24)

# ==============================
# Env & Policy
# ==============================
env = RacingEnv(seed=42)
policy = GaussianPolicy()

try:
    checkpoint = torch.load(args.model, map_location="cpu")
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
except:
    print("Error loading model, using untrained policy")

policy.eval()

state = env.reset()
action_history = []

# ==============================
# Render helpers
# ==============================
def render_info(screen, env, action):
    # Score
    score_text = font.render(f"Score: {env.score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))
    
    # Steps
    steps_text = font.render(f"Steps: {env.steps}", True, (255, 255, 255))
    screen.blit(steps_text, (10, 45))
    
    # Action
    action_text = small_font.render(f"Action: {action:.3f}", True, (255, 255, 255))
    screen.blit(action_text, (10, 80))
    
    # Position
    pos_text = small_font.render(f"Position: {env.player_x:.1f}", True, (255, 255, 255))
    screen.blit(pos_text, (10, 105))
    
    # Enemies count
    enemies_text = small_font.render(f"Enemies: {len(env.enemies)}", True, (255, 255, 255))
    screen.blit(enemies_text, (10, 130))

def render_road_markers(screen, env):
    # Разделительные линии
    for y in range(0, HEIGHT, 40):
        pygame.draw.rect(
            screen,
            (255, 255, 0),
            (env.ROAD_LEFT + (env.ROAD_RIGHT - env.ROAD_LEFT) // 2 - 5, y, 10, 20)
        )
    
    # Края дороги
    pygame.draw.line(
        screen,
        (255, 255, 255),
        (env.ROAD_LEFT, 0),
        (env.ROAD_LEFT, HEIGHT),
        3
    )
    pygame.draw.line(
        screen,
        (255, 255, 255),
        (env.ROAD_RIGHT, 0),
        (env.ROAD_RIGHT, HEIGHT),
        3
    )

# ==============================
# Main loop
# ==============================
running = True
paused = False

while running:
    dt = clock.tick(args.speed)
    
    # ---- events ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                state = env.reset()
                action_history = []
    
    if paused:
        # Показ паузы
        pause_text = font.render("PAUSED", True, (255, 0, 0))
        screen.blit(pause_text, (WIDTH // 2 - 50, HEIGHT // 2))
        pygame.display.flip()
        continue
    
    # ---- agent action ----
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, std, _ = policy(state_t)  # ← добавили _
    
    # Добавим немного разведки для игры
        exploration = 0.05 * torch.randn_like(mean)
        action = torch.tanh(mean + exploration).item()
        
        # Сохраняем историю действий
        action_history.append(action)
        if len(action_history) > 10:
            action_history.pop(0)
    
    # ---- environment step ----
    state, reward, done, _ = env.step(action)
    
    # ==============================
    # Render
    # ==============================
    screen.fill((30, 60, 30))  # Темный зеленый фон
    
    # Дорога
    pygame.draw.rect(
        screen,
        (80, 80, 80),
        (env.ROAD_LEFT, 0, env.ROAD_RIGHT - env.ROAD_LEFT, HEIGHT)
    )
    
    # Дорожная разметка
    render_road_markers(screen, env)
    
    # Враги
    for x, y, passed in env.enemies:
        color = (200, 50, 50) if not passed else (100, 100, 100)
        pygame.draw.rect(
            screen,
            color,
            (x - 20, y, 40, 60)
        )
        # Индикатор дистанции
        if y < HEIGHT - 100:
            dist = HEIGHT - 80 - y
            if dist < 200:
                pygame.draw.line(
                    screen,
                    (255, 100, 100),
                    (env.player_x, HEIGHT - 80),
                    (x, y + 30),
                    1
                )
    
    # Игрок
    player_color = (0, 120, 255)
    pygame.draw.rect(
        screen,
        player_color,
        (env.player_x - 20, HEIGHT - 80, 40, 60)
    )
    
    # Направление движения
    if action_history:
        avg_action = np.mean(action_history[-3:]) if len(action_history) >= 3 else action
        arrow_len = 30
        arrow_x = env.player_x + avg_action * arrow_len
        pygame.draw.line(
            screen,
            (255, 255, 0),
            (env.player_x, HEIGHT - 50),
            (arrow_x, HEIGHT - 50),
            3
        )
        pygame.draw.circle(
            screen,
            (255, 200, 0),
            (int(arrow_x), HEIGHT - 50),
            5
        )
    
    # Информация
    render_info(screen, env, action)
    
    # Предупреждение о краях
    if env.player_x < env.ROAD_LEFT + 50:
        edge_text = small_font.render("LEFT EDGE!", True, (255, 100, 100))
        screen.blit(edge_text, (env.ROAD_LEFT + 10, HEIGHT - 150))
    elif env.player_x > env.ROAD_RIGHT - 50:
        edge_text = small_font.render("RIGHT EDGE!", True, (255, 100, 100))
        screen.blit(edge_text, (env.ROAD_RIGHT - 100, HEIGHT - 150))
    
    pygame.display.flip()
    
    # ---- reset on done ----
    if done:
        # Эффект столкновения
        if reward < -50:
            for _ in range(10):
                screen.fill((255, 50, 50))
                pygame.display.flip()
                pygame.time.delay(50)
                screen.fill((30, 60, 30))
                pygame.display.flip()
                pygame.time.delay(50)
        
        state = env.reset()
        action_history = []
        pygame.time.delay(500)  # Пауза перед перезапуском

pygame.quit()
sys.exit()
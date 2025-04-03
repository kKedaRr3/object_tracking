import numpy as np
import cv2

# Ustawienie rozdzielczości 50x50
width, height = 50, 50

# Tworzymy obiekt video writer z kodekiem 'mp4v' (kodek dla MP4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Można też użyć *'H264' jeśli masz odpowiedni kodek
out = cv2.VideoWriter('../data/moving_ball.mp4', fourcc, 20.0, (width, height))  # Zmiana na .mp4

# Parametry ruchu kulki
ball_radius = 5
ball_center = [width // 4, height // 2]
ball_speed = [2, 1]

# Tworzymy animację
for _ in range(100):  # 100 klatek
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # Tło białe (wszystkie piksele mają wartość 255)

    # Rysujemy kulkę (zielona)
    cv2.circle(frame, tuple(ball_center), ball_radius, (0, 255, 0), -1)

    # Ruch kulki
    ball_center[0] += ball_speed[0]
    ball_center[1] += ball_speed[1]

    # Odbicia kulki od krawędzi
    if ball_center[0] - ball_radius <= 0 or ball_center[0] + ball_radius >= width:
        ball_speed[0] = -ball_speed[0]
    if ball_center[1] - ball_radius <= 0 or ball_center[1] + ball_radius >= height:
        ball_speed[1] = -ball_speed[1]

    # Zapisujemy klatkę do pliku
    out.write(frame)

# Zamykanie pliku
out.release()

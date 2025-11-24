# ==============================
# Импорты
# ==============================
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import math
import time
from PIL import ImageFont, ImageDraw, Image

font_path = "seguiemj.ttf"  # пример: шрифт Windows с эмодзи
font_size = 32
font = ImageFont.truetype(font_path, font_size)

status_game_text = ""
status_color = (0, 255, 0)  # по умолчанию зелёный
        
# ==============================
# Настройки
# ==============================
VIDEO_PATH = "test_videos/fragment_1080p50_av1.mp4"   # входное видео
OUTPUT_PATH = "results/football_analytics_out.mp4"  # результат
MODEL_PATH = "weights/best.pt"  # путь к YOLO модели
INFER_IMG_SIZE = 640      # размер входа модели 1280
USE_HSV_BALL_FILTER = False #True
ASSIGNMENT_DIST_THR = 120  # px, для Kalman трекера
SHOT_SPEED_THRESHOLD = 15  # px/frame, порог скорости для удара по воротам
STRONG_PASS_THRESHOLD = 20 # px/frame, порог скорости для сильного удара/паса
GOAL_AREA_RATIO = 0.2      # зона ворот: 20% слева/справа
GOAL_ZONE_MARGIN = 50       # px — зона ворот для опасного момента
MAX_TRAJECTORY_LEN = 20     # длина линии траектории мяча

# ==============================
# Классы, цвета, подписи
# ==============================
labels = ["Player-L", "Player-R", "GK-L", "GK-R", "myach", "Main Ref", "Side Ref", "Staff"]
box_colors = {
    "0": (150, 50, 50),
    "1": (37, 47, 150),
    "2": (41, 248, 165),
    "3": (166, 196, 10),
    "4": (155, 62, 157),
    "5": (123, 174, 213),
    "6": (217, 89, 204),
    "7": (22, 11, 15)
}

# ==============================
# Вспомогательные функции
# ==============================
def get_grass_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(frame, mask=mask)[:3]
    return grass_color

def get_players_boxes(result):
    players_imgs, players_boxes = [], []
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 0:  # Player
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w = result.orig_img.shape[:2]
            x1c, x2c = max(0, x1), min(w, x2)
            y1c, y2c = max(0, y1), min(h, y2)
            players_imgs.append(result.orig_img[y1c:y2c, x1c:x2c])
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players_imgs, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None and frame is not None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
    for img in players_imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([max(0, grass_hsv[0,0,0]-10), 40, 40])
        upper_green = np.array([min(179, grass_hsv[0,0,0]+10), 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(img.shape[:2], np.uint8)
        upper_mask[0:img.shape[0]//2, :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def get_kits_classifier(kits_colors):
    if len(kits_colors) < 2:
        return None
    clf = KMeans(n_clusters=2)
    clf.fit(kits_colors)
    return clf

def classify_kits(clf, kits_colors):
    if clf is None:
        return np.zeros(len(kits_colors), dtype=int)
    return clf.predict(kits_colors)

def get_left_team_label(players_boxes, kits_colors, clf):
    if clf is None or len(players_boxes) == 0:
        return 0
    teams = classify_kits(clf, kits_colors)
    team_0_x, team_1_x = [], []
    for i, box in enumerate(players_boxes):
        coords = box.xyxy[0].cpu().numpy()
        x1 = int(coords[0])
        if teams[i]==0:
            team_0_x.append(x1)
        else:
            team_1_x.append(x1)
    if len(team_0_x)==0: return 1
    if len(team_1_x)==0: return 0
    return 0 if np.mean(team_0_x) < np.mean(team_1_x) else 1

def get_ball_detections_from_result(result):
    dets = []
    for box in result.boxes:
        cls = int(box.cls.numpy()[0])
        if cls == 2:  # Ball
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            dets.append((x1,y1,x2,y2,cx,cy))
    return dets

def is_ball_color_ok(bbox, frame):
    x1,y1,x2,y2,_,_ = bbox
    h, w = frame.shape[:2]
    x1c, x2c = max(0,x1), min(w,x2)
    y1c, y2c = max(0,y1), min(h,y2)
    if x2c - x1c <= 2 or y2c - y1c <= 2:
        return False
    crop = frame[y1c:y2c, x1c:x2c]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 150])
    upper = np.array([179, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)
    ratio = mask.sum() / (255.0 * mask.size)
    return ratio > 0.02

# ==============================
# Kalman трекер мяча
# ==============================
class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        dt = 1.
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_update_time = None
        self.age = 0
        self.missed = 0

    def initialize(self, x, y):
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0.],[0.]], dtype=np.float32)
        self.initialized = True
        self.missed = 0
        self.age = 1
        self.last_update_time = time.time()

    def predict(self):
        if not self.initialized: return None
        state = self.kf.predict()
        x, y = state[0,0], state[1,0]
        return int(x), int(y)

    def update(self, detections):
        if not self.initialized:
            if len(detections) > 0:
                x1,y1,x2,y2,cx,cy = detections[0]
                self.initialize(cx, cy)
                return (cx, cy)
            else:
                return None

        pred = self.predict()
        if len(detections) == 0:
            self.missed += 1
            self.age += 1
            return pred

        best = None
        best_dist = float('inf')
        for det in detections:
            cx = det[4]; cy = det[5]
            d = math.hypot(cx - pred[0], cy - pred[1])
            if d < best_dist:
                best_dist = d
                best = det

        if best is not None and best_dist <= ASSIGNMENT_DIST_THR:
            meas = np.array([[np.float32(best[4])], [np.float32(best[5])]], dtype=np.float32)
            self.kf.correct(meas)
            self.missed = 0
            self.age += 1
            self.last_update_time = time.time()
            return (int(best[4]), int(best[5]))
        else:
            self.missed += 1
            self.age += 1
            return pred

    def reset_if_old(self, max_missed=30):
        if self.missed > max_missed:
            self.initialized = False
            self.missed = 0
            self.age = 0

# ==============================
# Загрузка видео и модели
# ==============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

kits_clf = None
left_team_label = 0
grass_hsv = None

# ==============================
# Статистика
# ==============================
possession_counts = [0,0]
last_possession = None
possession_changes = 0
ball_distance = 0.0
prev_ball_pos = None
shots_on_goal = [0,0]
strong_passes = [0,0]
ball_positions_history = []
poss_l_pct = 0
poss_r_pct = 0
MAX_JUMP_THRESHOLD = 50

# =========================
# Инициализация флагов эпизодов
# =========================
in_goal_zone_L = False
in_goal_zone_R = False

# ==============================
# Инициализация трекера
# ==============================
ball_tracker = BallTracker()
# =========================
# Инициализация переменных для дистанции игроков
prev_player_positions = []
team_distance = {0:0.0, 1:0.0}
MAX_JUMP_THRESHOLD = 30  # px, фильтр резких скачков

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # =========================
    # YOLO inference
    # =========================
    result = model(frame, conf=0.25, verbose=False, imgsz=INFER_IMG_SIZE)[0] if INFER_IMG_SIZE else model(frame, conf=0.4, verbose=False)[0]

    # =========================
    # Игроки и цвета форм
    # =========================
    players_imgs, players_boxes = get_players_boxes(result)
    kits_colors = get_kits_colors(players_imgs, grass_hsv, frame)
    if kits_clf is None and len(players_imgs) > 1:
        kits_clf = get_kits_classifier(kits_colors)
        left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    # =========================
    # Мяч
    # =========================
    ball_dets = get_ball_detections_from_result(result)
    if USE_HSV_BALL_FILTER and len(ball_dets) > 0:
        ball_dets = [d for d in ball_dets if is_ball_color_ok(d, frame)]

    tracked_ball = ball_tracker.update(ball_dets)
    ball_tracker.reset_if_old(max_missed=45)

    bx = by = None
    ball_speed = 0
    if tracked_ball:
        bx, by = tracked_ball
        current_possession = 0 if bx < width/2 else 1
        possession_counts[current_possession] += 1
        last_possession = current_possession

        dx = dy = 0
        if prev_ball_pos is not None:
            dx = bx - prev_ball_pos[0]
            dy = by - prev_ball_pos[1]
            ball_speed = math.hypot(dx, dy)
            ball_distance += ball_speed
        prev_ball_pos = (bx, by)

        total_possession = sum(possession_counts) or 1
        poss_l_pct = possession_counts[0] / total_possession
        poss_r_pct = possession_counts[1] / total_possession

        # --- Проверка наличия вратаря и определение зон ворот ---
        gk_left_in_frame = any(int(box.cls.numpy()[0]) == 2 for box in result.boxes)
        gk_right_in_frame = any(int(box.cls.numpy()[0]) == 3 for box in result.boxes)

        left_goal_zone = (0, int(width * GOAL_AREA_RATIO + GOAL_ZONE_MARGIN)) if gk_left_in_frame else None
        right_goal_zone = (int(width*(1-GOAL_AREA_RATIO) - GOAL_ZONE_MARGIN), width) if gk_right_in_frame else None

        # --- Удары по воротам с учётом эпизодов и направления движения ---
        if left_goal_zone and bx >= left_goal_zone[0] and bx <= left_goal_zone[1]:
            if gk_left_in_frame:
                moving_towards_goal = dx < 0
                if moving_towards_goal and not in_goal_zone_L:
                    shots_on_goal[1] += 1
                    in_goal_zone_L = True
        else:
            in_goal_zone_L = False

        if right_goal_zone and bx >= right_goal_zone[0] and bx <= right_goal_zone[1]:
            if gk_right_in_frame:
                moving_towards_goal = dx > 0
                if moving_towards_goal and not in_goal_zone_R:
                    shots_on_goal[0] += 1
                    in_goal_zone_R = True
        else:
            in_goal_zone_R = False

        if ball_speed > STRONG_PASS_THRESHOLD:
            strong_passes[last_possession] += 1


    # Определяем статус игры с проверкой ворот
    left_goal_zone = (0, int(width * GOAL_AREA_RATIO + GOAL_ZONE_MARGIN))
    right_goal_zone = (int(width*(1-GOAL_AREA_RATIO) - GOAL_ZONE_MARGIN), width)

    status_game_text = "🟢 Safe Play"
    status_color = (0, 255, 0)

    if bx is not None:
        # Проверяем левую зону ворот
        if left_goal_zone[0] <= bx <= left_goal_zone[1] and gk_left_in_frame:
            status_game_text = "🟡 Dangerous Moment"
            status_color = (0, 255, 255)
            # если мяч очень близко к воротам
            if bx <= left_goal_zone[1] and bx >= left_goal_zone[1] - 30:
                status_game_text = "🔴 Goal Threat"
                status_color = (0, 0, 255)

        # Проверяем правую зону ворот
        elif right_goal_zone[0] <= bx <= right_goal_zone[1] and gk_right_in_frame:
            status_game_text = "🟡 Dangerous Moment"
            status_color = (0, 255, 255)
            if bx >= right_goal_zone[0] and bx <= right_goal_zone[0] + 30:
                status_game_text = "🔴 Goal Threat"
                status_color = (0, 0, 255)


    # =========================
    # Рисуем статус игры
    # =========================
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10, 10), status_game_text, font=font, fill=status_color)
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # =========================
    # Отрисовка игроков и подсчёт дистанции
    # =========================
    player_idx = 0
    team_counts = {0:0, 1:0}

    for box in result.boxes:
        label_orig = int(box.cls.numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if label_orig == 0:  # игрок
            team = classify_kits(kits_clf, [kits_colors[player_idx]])[0]
            label = 0 if team == left_team_label else 1
            team_counts[label] += 1

            # >>> ADDED: подсчёт дистанции и скорости игроков
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            if player_idx >= len(prev_player_positions):
                prev_player_positions.append((cx, cy))
            else:
                px, py = prev_player_positions[player_idx]
                d = math.hypot(cx - px, cy - py)
                if d < MAX_JUMP_THRESHOLD:
                    team_distance[label] += d
                prev_player_positions[player_idx] = (cx, cy)

            player_idx += 1
        elif label_orig == 1:
            label = 2 if x1 < width/2 else 3
        else:
            label = label_orig + 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
        cv2.putText(frame, labels[label], (x1-30, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    total_distance = team_distance[0] + team_distance[1] or 1  # чтобы не делить на 0
    pct_L = team_distance[0] / total_distance
    pct_R = team_distance[1] / total_distance   

    # Владение мячом (нижняя полоска)
    bar_w, bar_h = 200, 15
    bar_x, bar_y = 10, height - 30
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (180,180,180), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w*poss_l_pct), bar_y+bar_h), box_colors['0'], -1)
    cv2.rectangle(frame, (bar_x + int(bar_w*poss_l_pct), bar_y), (bar_x+bar_w, bar_y+bar_h), box_colors['1'], -1)
    cv2.putText(frame, f"Possession L:{int(poss_l_pct*100)}% R:{int(poss_r_pct*100)}%", 
                (bar_x, bar_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Дистанция (верхняя полоска)
    bar_y2 = bar_y - 50  # чуть выше
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x+bar_w, bar_y2+bar_h), (180,180,180), -1)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + int(bar_w*pct_L), bar_y2+bar_h), (0,255,0), -1)  # зеленый для левой команды
    cv2.rectangle(frame, (bar_x + int(bar_w*pct_L), bar_y2), (bar_x+bar_w, bar_y2+bar_h), (0,0,255), -1)  # красный для правой команды
    cv2.putText(frame, f"Distance L:{int(pct_L*100)}% R:{int(pct_R*100)}%", 
                (bar_x, bar_y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


    # =========================
    # Мини-карта поля
    # =========================
    map_w, map_h = 200, 120
    map_x, map_y = width - map_w - 10, 10
    cv2.rectangle(frame, (map_x, map_y), (map_x+map_w, map_y+map_h), (50,50,50), -1)
    player_idx_for_kits = 0
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        px = int((x1 + x2)/2 * map_w / width) + map_x
        py = int((y1 + y2)/2 * map_h / height) + map_y
        if label in [0,1]:
            if player_idx_for_kits < len(kits_colors):
                team = classify_kits(kits_clf, [kits_colors[player_idx_for_kits]])[0] if kits_clf else 0
            else:
                team = 0
            color = box_colors[str(0 if team==left_team_label else 1)]
            player_idx_for_kits += 1
            cv2.circle(frame, (px, py), 4, color, -1)
        elif label == 2:
            cv2.circle(frame, (px, py), 3, (0,255,255), -1)

    # =========================
    # Остальная статистика
    # =========================
    cv2.putText(frame, f"Players L:{team_counts.get(0,0)} R:{team_counts.get(1,0)}", (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    #cv2.putText(frame, f"Distance L:{team_distance[0]:.1f} R:{team_distance[1]:.1f}", (10,250),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(frame, f"Ball pos: {tracked_ball}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    cv2.putText(frame, f"Ball speed: {ball_speed:.2f}", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    cv2.putText(frame, f"Ball distance: {ball_distance:.1f}", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
    #cv2.putText(frame, f"Shots on goal L:{shots_on_goal[0]} R:{shots_on_goal[1]}", (10,220),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

    # =========================
    # Запись и отображение
    # =========================
    out.write(frame)
    cv2.imshow("Football Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Видео обработано и сохранено в:", OUTPUT_PATH)
        
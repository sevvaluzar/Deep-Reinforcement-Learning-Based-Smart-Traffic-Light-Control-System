from __future__ import annotations
from google.colab import files
from IPython.display import Image, display

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1) AYARLAR
# ============================================================

SEED = 42
OUTPUT_DIR = Path("outputs_drl_traffic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Daha hızlı deneme için 150-250, daha stabil sonuç için 500-800 kullanabilirsiniz.
EPISODES = 500
MAX_SIM_SECONDS = 360
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
LEARNING_RATE = 1e-3
GAMMA_PER_SECOND = 0.995
TARGET_UPDATE_EVERY = 400
TRAIN_AFTER = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 9_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

# GIF okunabilirlik ayarları.
# GIF her 3 simülasyon saniyesinde bir kare alır; her kare 0.9 saniye görünür.
# Böylece araç, yaya ve ambulans detayları ekranda yeterince uzun kalır.
GIF_FRAME_EVERY_SECONDS = 2
GIF_FRAME_DURATION_MS = 900
GIF_MAX_FRAMES = 90
GIF_EVALUATION_SECONDS = 220



# ============================================================
# 2) ORTAM: 4 KOLLU AKILLI KAVŞAK
# ============================================================

DIRECTIONS = ["N", "S", "E", "W"]
MOVEMENTS = ["Duz", "Sol", "Sag"]
GREEN_DURATIONS = [8, 12, 16]

# Fazlar: araç fazlarında tüm yaya ışıkları kırmızıdır.
PHASES = {
    0: {"name": "Kuzey-Guney Duz/Sag", "lanes": [0, 2, 3, 5]},
    1: {"name": "Dogu-Bati Duz/Sag", "lanes": [6, 8, 9, 11]},
    2: {"name": "Kuzey-Guney Sol", "lanes": [1, 4]},
    3: {"name": "Dogu-Bati Sol", "lanes": [7, 10]},
    4: {"name": "Yaya Fazı", "lanes": []},
}

YELLOW_SECONDS = 3
MAX_PEDESTRIAN_WAIT = 34
MAX_EMERGENCY_WAIT = 10


def lane_name(lane_idx: int) -> str:
    return f"{DIRECTIONS[lane_idx // 3]}-{MOVEMENTS[lane_idx % 3]}"


def phase_for_lane(lane_idx: int) -> int:
    """Acil aracın bulunduğu şeride yeşil sağlayan korumalı fazı döndürür."""
    direction = lane_idx // 3
    movement = lane_idx % 3
    is_ns = direction in (0, 1)
    if movement == 1:  # sol dönüş
        return 2 if is_ns else 3
    return 0 if is_ns else 1


class TrafficIntersectionEnv:
    """Harici SUMO gerektirmeyen, eğitim için tasarlanmış kavşak ortamı.

    Her adımda ajan, hedef fazı ve o fazın yeşil süresini seçer.
    Aksiyon sayısı: 5 faz x 3 süre = 15.
    """

    def __init__(self, seed: int = SEED, max_sim_seconds: int = MAX_SIM_SECONDS):
        self.rng = np.random.default_rng(seed)
        self.max_sim_seconds = max_sim_seconds
        self.num_lanes = 12
        self.num_ped_crossings = 4
        self.action_size = len(PHASES) * len(GREEN_DURATIONS)
        self.observation_size = 67
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.time = 0
        self.current_phase = 0
        self.phase_elapsed = 0
        self.signal_mode = "green"  # green veya yellow; GIF renderında kullanılır.
        self.vehicle_q = self.rng.integers(0, 5, size=self.num_lanes).astype(np.float32)
        self.vehicle_wait_sum = np.zeros(self.num_lanes, dtype=np.float32)
        self.ped_q = self.rng.integers(0, 3, size=self.num_ped_crossings).astype(np.float32)
        self.ped_wait_sum = np.zeros(self.num_ped_crossings, dtype=np.float32)

        # Her bölümde farklı bir ana yön daha yoğun olabilir.
        self.direction_demand = np.ones(4, dtype=np.float32)
        self.busy_direction = int(self.rng.integers(0, 4))
        self.direction_demand[self.busy_direction] = float(self.rng.uniform(1.25, 1.75))

        # Geçici şerit kapanması: bölümün ortasında başlar ve daha sonra açılır.
        self.closure_lane = int(self.rng.integers(0, self.num_lanes))
        self.closure_start = int(self.rng.integers(85, 135))
        self.closure_end = self.closure_start + int(self.rng.integers(35, 56))
        self.closed_lane = -1

        # Bir ambulans / itfaiye aracı bölüm içinde görünür.
        self.emergency_lane = -1
        self.emergency_wait = 0.0
        self.emergency_spawn_time = int(self.rng.integers(120, 190))
        self.emergency_spawned = False
        self.emergency_served = False

        self.total_passed_vehicles = 0
        self.total_crossed_pedestrians = 0
        self.total_phase_switches = 0
        self.last_override_reason = "Yok"
        return self._get_obs()

    def decode_action(self, action: int) -> Tuple[int, int]:
        phase = int(action // len(GREEN_DURATIONS))
        duration = int(GREEN_DURATIONS[action % len(GREEN_DURATIONS)])
        return phase, duration

    def step(
        self,
        action: int,
        frame_callback=None,
        frame_every_seconds: int = GIF_FRAME_EVERY_SECONDS,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Bir DRL aksiyonunu uygular.

        Eğitim sırasında frame_callback verilmez. GIF oluştururken callback kullanılır ve
        ortam her birkaç simülasyon saniyesinde bir kare render eder. Böylece GIF yalnızca
        8/12/16 saniyelik karar noktalarını değil, trafik değişimini de gösterir.
        """
        requested_phase, requested_duration = self.decode_action(int(action))
        target_phase, green_seconds = requested_phase, requested_duration
        override_reason = "Yok"

        # Güvenlik katmanı: ajan geç kalırsa acil araca ve aşırı bekleyen yayaya zorunlu öncelik.
        if self.emergency_lane >= 0 and self.emergency_wait >= MAX_EMERGENCY_WAIT:
            target_phase = phase_for_lane(self.emergency_lane)
            green_seconds = max(GREEN_DURATIONS)
            override_reason = "Acil araç güvenlik önceliği"
        elif self._max_ped_wait() >= MAX_PEDESTRIAN_WAIT:
            target_phase = 4
            green_seconds = max(GREEN_DURATIONS)
            override_reason = "Maksimum yaya bekleme sınırı"

        start_time = self.time
        reward = 0.0
        switched = target_phase != self.current_phase

        def capture_frame_if_needed(force: bool = False) -> None:
            if frame_callback is None:
                return
            if force or self.time % max(1, frame_every_seconds) == 0:
                frame_callback(self.render_rgb())

        # Faz değişiminde sarı ışık: tüm araç ve yayalar güvenlik için bekler.
        if switched:
            self.total_phase_switches += 1
            reward -= 1.5
            self.signal_mode = "yellow"
            for _ in range(YELLOW_SECONDS):
                if self.time >= self.max_sim_seconds:
                    break
                reward += self._simulate_one_second(active_phase=None)
                capture_frame_if_needed()
            self.current_phase = target_phase
            self.phase_elapsed = 0
            self.signal_mode = "green"
            capture_frame_if_needed(force=True)

        # Seçilen dinamik yeşil süre boyunca ortamı ilerlet.
        for _ in range(green_seconds):
            if self.time >= self.max_sim_seconds:
                break
            reward += self._simulate_one_second(active_phase=self.current_phase)
            self.phase_elapsed += 1
            capture_frame_if_needed()

        self.last_override_reason = override_reason
        terminated = self.time >= self.max_sim_seconds
        elapsed = self.time - start_time

        info = {
            "elapsed": elapsed,
            "requested_phase": requested_phase,
            "selected_phase": self.current_phase,
            "selected_duration": green_seconds,
            "override_reason": override_reason,
            "vehicle_queue": float(self.vehicle_q.sum()),
            "pedestrian_queue": float(self.ped_q.sum()),
            "emergency_wait": float(self.emergency_wait),
            "closed_lane": self.closed_lane,
        }
        return self._get_obs(), float(reward), terminated, info

    def _simulate_one_second(self, active_phase: int | None) -> float:
        self.time += 1
        self._update_dynamic_events()
        self._spawn_emergency_if_needed()

        # Trafik talebi gün içinde değişir: normal -> yoğun saat -> normal.
        demand_factor = self._time_demand_factor()
        base_lane_rates = np.array(
            [0.20, 0.09, 0.06, 0.18, 0.08, 0.05, 0.22, 0.10, 0.06, 0.17, 0.08, 0.05],
            dtype=np.float32,
        )
        lane_factors = np.repeat(self.direction_demand, 3)
        arrivals = self.rng.poisson(base_lane_rates * lane_factors * demand_factor).astype(np.float32)
        self.vehicle_q += arrivals

        ped_rates = np.array([0.08, 0.08, 0.06, 0.06], dtype=np.float32)
        if demand_factor > 1.25:
            ped_rates *= 1.35
        self.ped_q += self.rng.poisson(ped_rates).astype(np.float32)

        # Mevcut kuyrukların bir saniyelik bekleme maliyeti.
        self.vehicle_wait_sum += self.vehicle_q
        self.ped_wait_sum += self.ped_q
        if self.emergency_lane >= 0:
            self.emergency_wait += 1.0

        passed_vehicles = 0
        crossed_pedestrians = 0

        if active_phase is not None and active_phase != 4:
            for lane in PHASES[active_phase]["lanes"]:
                if lane == self.closed_lane:
                    continue
                service_capacity = 2 if lane % 3 == 0 else 1
                passed_vehicles += self._serve_vehicle_lane(lane, service_capacity)

        if active_phase == 4:
            for crossing in range(self.num_ped_crossings):
                crossed_pedestrians += self._serve_pedestrian_crossing(crossing, capacity=3)

        self.total_passed_vehicles += passed_vehicles
        self.total_crossed_pedestrians += crossed_pedestrians

        # Çok amaçlı ödül fonksiyonu.
        reward = (
            -0.55 * float(self.vehicle_q.sum())
            -1.10 * float(self.ped_q.sum())
            -0.012 * float(self.vehicle_wait_sum.sum())
            -0.045 * float(self.ped_wait_sum.sum())
            +0.75 * passed_vehicles
            +1.50 * crossed_pedestrians
        )

        # Acil araç beklerken yüksek ceza uygulanır; öncelik öğrenmesini güçlendirir.
        if self.emergency_lane >= 0:
            reward -= 6.0 + 1.5 * self.emergency_wait

        # Şerit kapanmasında o şeritte aşırı birikme ek maliyet oluşturur.
        if self.closed_lane >= 0:
            reward -= 0.35 * float(self.vehicle_q[self.closed_lane])

        # Ödül ölçekleme, Q-değerlerinin sayısal olarak daha stabil öğrenilmesini sağlar.
        return float(reward * 0.02)

    def _serve_vehicle_lane(self, lane: int, capacity: int) -> int:
        queue_before = float(self.vehicle_q[lane])
        if queue_before <= 0:
            return 0
        departed = int(min(queue_before, capacity))

        # Ortalama yaş yaklaşımıyla, ayrılan araçların bekleme toplamını çıkar.
        average_wait = self.vehicle_wait_sum[lane] / max(queue_before, 1.0)
        self.vehicle_wait_sum[lane] = max(0.0, self.vehicle_wait_sum[lane] - departed * average_wait)
        self.vehicle_q[lane] -= departed

        # Acil araç, öncelikli faz aktif olduğunda ilk geçen araç kabul edilir.
        if self.emergency_lane == lane and departed > 0:
            self.emergency_lane = -1
            self.emergency_wait = 0.0
            self.emergency_served = True
        return departed

    def _serve_pedestrian_crossing(self, crossing: int, capacity: int) -> int:
        queue_before = float(self.ped_q[crossing])
        if queue_before <= 0:
            return 0
        crossed = int(min(queue_before, capacity))
        average_wait = self.ped_wait_sum[crossing] / max(queue_before, 1.0)
        self.ped_wait_sum[crossing] = max(0.0, self.ped_wait_sum[crossing] - crossed * average_wait)
        self.ped_q[crossing] -= crossed
        return crossed

    def _update_dynamic_events(self) -> None:
        if self.closure_start <= self.time < self.closure_end:
            self.closed_lane = self.closure_lane
        else:
            self.closed_lane = -1

    def _spawn_emergency_if_needed(self) -> None:
        if self.emergency_spawned or self.time < self.emergency_spawn_time:
            return
        candidate_lanes = [i for i in range(self.num_lanes) if i != self.closed_lane]
        self.emergency_lane = int(self.rng.choice(candidate_lanes))
        self.vehicle_q[self.emergency_lane] += 1.0
        self.emergency_spawned = True

    def _time_demand_factor(self) -> float:
        # 0-119 normal, 120-239 yoğun saat, 240-360 normal.
        if 120 <= self.time < 240:
            return 1.45
        if self.time >= 300:
            return 0.85
        return 1.0

    def _current_ped_wait_vector(self) -> np.ndarray:
        return self.ped_wait_sum / np.maximum(self.ped_q, 1.0)

    def _max_ped_wait(self) -> float:
        return float(np.max(self._current_ped_wait_vector())) if self.ped_q.sum() > 0 else 0.0

    def _get_obs(self) -> np.ndarray:
        lane_avg_wait = self.vehicle_wait_sum / np.maximum(self.vehicle_q, 1.0)
        ped_avg_wait = self._current_ped_wait_vector()

        phase_one_hot = np.zeros(len(PHASES), dtype=np.float32)
        phase_one_hot[self.current_phase] = 1.0

        closure_one_hot = np.zeros(self.num_lanes, dtype=np.float32)
        if self.closed_lane >= 0:
            closure_one_hot[self.closed_lane] = 1.0
        closure_remaining = 0.0
        if self.closed_lane >= 0:
            closure_remaining = max(0.0, (self.closure_end - self.time) / 60.0)

        emergency_one_hot = np.zeros(self.num_lanes, dtype=np.float32)
        if self.emergency_lane >= 0:
            emergency_one_hot[self.emergency_lane] = 1.0

        # Talep profili: normal / yoğun saat / düşük trafik.
        profile = np.zeros(3, dtype=np.float32)
        factor = self._time_demand_factor()
        profile[1 if factor > 1.2 else (2 if factor < 0.95 else 0)] = 1.0

        obs = np.concatenate(
            [
                np.clip(self.vehicle_q / 25.0, 0, 1),                 # 12
                np.clip(lane_avg_wait / 60.0, 0, 1),                  # 12
                np.clip(self.ped_q / 15.0, 0, 1),                     # 4
                np.clip(ped_avg_wait / 60.0, 0, 1),                   # 4
                phase_one_hot,                                         # 5
                np.array([min(self.phase_elapsed / 20.0, 1.0)], dtype=np.float32),
                closure_one_hot,                                      # 12
                np.array([closure_remaining], dtype=np.float32),
                emergency_one_hot,                                    # 12
                np.array([min(self.emergency_wait / 30.0, 1.0)], dtype=np.float32),
                profile,                                               # 3
            ]
        ).astype(np.float32)
        assert obs.shape[0] == self.observation_size
        return obs

    # ------------------------- GIF render -------------------------
    @staticmethod
    def _car_angle(direction: int) -> float:
        """Arabanın burun yönü: N->güneye, S->kuzeye, E->batıya, W->doğuya."""
        return {0: -90, 1: 90, 2: 180, 3: 0}[direction]

    @staticmethod
    def _add_rotated_patch(ax, patch, x: float, y: float, angle: float) -> None:
        patch.set_transform(Affine2D().rotate_deg_around(x, y, angle) + ax.transData)
        ax.add_patch(patch)

    def _draw_car(self, ax, x: float, y: float, direction: int, emergency: bool = False) -> None:
        """Üstten görünüşlü, camlı ve tekerlekli araba silüeti çizer."""
        angle = self._car_angle(direction)
        body_color = "#e83737" if emergency else "#f5ad3c"
        outline = "#5d1515" if emergency else "#5a3b12"
        length, width = 0.56, 0.30

        # Ana gövde
        body = FancyBboxPatch(
            (x - length / 2, y - width / 2), length, width,
            boxstyle="round,pad=0.015,rounding_size=0.07",
            facecolor=body_color, edgecolor=outline, linewidth=1.2, zorder=14,
        )
        self._add_rotated_patch(ax, body, x, y, angle)

        # Ön cam, arka cam ve tavan. Baz şekil sağa bakar; daha sonra döndürülür.
        windshield = Polygon(
            [(x + 0.06, y - 0.105), (x + 0.20, y - 0.09),
             (x + 0.20, y + 0.09), (x + 0.06, y + 0.105)],
            closed=True, facecolor="#a9d8ef", edgecolor="#356075", linewidth=0.55, zorder=16,
        )
        rear_window = Polygon(
            [(x - 0.18, y - 0.08), (x - 0.05, y - 0.10),
             (x - 0.05, y + 0.10), (x - 0.18, y + 0.08)],
            closed=True, facecolor="#8ec5df", edgecolor="#356075", linewidth=0.5, zorder=16,
        )
        roof = FancyBboxPatch(
            (x - 0.07, y - 0.105), 0.18, 0.21,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            facecolor="#d7edf6" if not emergency else "#ffffff",
            edgecolor="#356075", linewidth=0.45, zorder=15,
        )
        for part in (windshield, rear_window, roof):
            self._add_rotated_patch(ax, part, x, y, angle)

        # Dört lastik
        for wx, wy in ((x - 0.16, y - 0.17), (x - 0.16, y + 0.17),
                       (x + 0.16, y - 0.17), (x + 0.16, y + 0.17)):
            wheel = FancyBboxPatch(
                (wx - 0.045, wy - 0.03), 0.09, 0.06,
                boxstyle="round,pad=0.0,rounding_size=0.015",
                facecolor="#202020", edgecolor="#080808", linewidth=0.5, zorder=13,
            )
            self._add_rotated_patch(ax, wheel, x, y, angle)

        # Farlar
        for hx, hy in ((x + 0.285, y - 0.085), (x + 0.285, y + 0.085)):
            lamp = Circle((hx, hy), 0.023, facecolor="#fff6b7", edgecolor="#ad8e33", linewidth=0.3, zorder=17)
            self._add_rotated_patch(ax, lamp, x, y, angle)

        if emergency:
            # Ambulans/itfaiye için beyaz şerit, + işareti ve ikaz lambaları.
            stripe = Rectangle((x - 0.02, y - 0.135), 0.055, 0.27,
                               facecolor="white", edgecolor="none", zorder=18)
            self._add_rotated_patch(ax, stripe, x, y, angle)
            ax.text(x, y, "+", color="#b40f0f", fontsize=9, fontweight="bold",
                    ha="center", va="center", zorder=20,
                    rotation=angle)
            for lx, color in ((x - 0.03, "#2279e8"), (x + 0.07, "#ea2637")):
                beacon = Circle((lx, y + 0.17), 0.032, facecolor=color, edgecolor="#262626", linewidth=0.35, zorder=20)
                self._add_rotated_patch(ax, beacon, x, y, angle)

    def _draw_pedestrian(self, ax, x: float, y: float, facing: int = 1) -> None:
        """Baş, gövde, kollar ve bacaklardan oluşan belirgin yürüyen insan ikonu çizer."""
        s = 0.22
        skin = "#f4c6a8"
        cloth = "#2378b9"
        # Baş
        ax.add_patch(Circle((x, y + 0.17 * s / 0.22), 0.073, facecolor=skin,
                            edgecolor="#6d4a3b", linewidth=0.55, zorder=22))
        # Gövde, kol ve bacaklar; çizgi kalınlığı GIF'te görünür olacak şekilde büyük tutulur.
        ax.plot([x, x], [y + 0.09, y - 0.10], color=cloth, lw=2.6, solid_capstyle="round", zorder=21)
        ax.plot([x, x - 0.11 * facing], [y + 0.03, y - 0.035], color=cloth, lw=2.0, solid_capstyle="round", zorder=21)
        ax.plot([x, x + 0.12 * facing], [y + 0.025, y + 0.095], color=cloth, lw=2.0, solid_capstyle="round", zorder=21)
        ax.plot([x, x - 0.095 * facing], [y - 0.10, y - 0.22], color="#1c3448", lw=2.1, solid_capstyle="round", zorder=21)
        ax.plot([x, x + 0.095 * facing], [y - 0.10, y - 0.20], color="#1c3448", lw=2.1, solid_capstyle="round", zorder=21)

    def _draw_traffic_light(self, ax, x: float, y: float, status: str) -> None:
        """Kırmızı-sarı-yeşil lambası bulunan gerçekçi sinyal kutusu çizer."""
        box = FancyBboxPatch((x - 0.14, y - 0.31), 0.28, 0.62,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor="#1d2328", edgecolor="black", linewidth=0.8, zorder=30)
        ax.add_patch(box)
        states = {
            "red": ("#e13b3b", "#5b1919", "#1f542f"),
            "yellow": ("#5b1919", "#f0bb38", "#1f542f"),
            "green": ("#5b1919", "#725a20", "#29b362"),
        }
        colors = states[status]
        for yy, color in zip((y + 0.18, y, y - 0.18), colors):
            ax.add_patch(Circle((x, yy), 0.064, facecolor=color, edgecolor="#0d0d0d", linewidth=0.45, zorder=31))

    def render_rgb(self) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        ax.set_facecolor("#d9ead8")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
        ax.axis("off")

        # Yol, kavşak zemini ve kaldırımlar.
        ax.add_patch(Rectangle((-1.95, -6), 3.9, 12, color="#4b5157", zorder=0))
        ax.add_patch(Rectangle((-6, -1.95), 12, 3.9, color="#4b5157", zorder=0))
        ax.add_patch(Rectangle((-1.95, -1.95), 3.9, 3.9, color="#41474c", zorder=1))

        # Şerit ayırıcıları; giriş yönleri için üçer şerit görünür.
        for offset in (-0.65, 0.0, 0.65):
            ax.plot([offset, offset], [-6, -1.95], color="white", lw=1.25, alpha=0.83, zorder=2)
            ax.plot([offset, offset], [1.95, 6], color="white", lw=1.25, alpha=0.83, zorder=2)
            ax.plot([-6, -1.95], [offset, offset], color="white", lw=1.25, alpha=0.83, zorder=2)
            ax.plot([1.95, 6], [offset, offset], color="white", lw=1.25, alpha=0.83, zorder=2)

        # Zebra geçitleri: tek dikdörtgen yerine çizgili şeritler.
        for k in range(9):
            ax.add_patch(Rectangle((-1.65 + k * 0.38, 1.66), 0.19, 0.30, color="white", alpha=0.95, zorder=4))
            ax.add_patch(Rectangle((-1.65 + k * 0.38, -1.96), 0.19, 0.30, color="white", alpha=0.95, zorder=4))
            ax.add_patch(Rectangle((1.66, -1.65 + k * 0.38), 0.30, 0.19, color="white", alpha=0.95, zorder=4))
            ax.add_patch(Rectangle((-1.96, -1.65 + k * 0.38), 0.30, 0.19, color="white", alpha=0.95, zorder=4))

        # Trafik ışıkları. Sarı geçişte bütün yönlerde sarı görünür.
        active_lanes = set(PHASES[self.current_phase]["lanes"]) if self.current_phase != 4 else set()
        signal_positions = {0: (-2.30, 2.28), 1: (2.30, -2.28), 2: (2.28, 2.30), 3: (-2.28, -2.30)}
        for direction in range(4):
            direction_lanes = set(range(direction * 3, direction * 3 + 3))
            if self.signal_mode == "yellow":
                status = "yellow"
            elif self.current_phase == 4:
                status = "red"
            else:
                status = "green" if active_lanes & direction_lanes else "red"
            self._draw_traffic_light(ax, *signal_positions[direction], status)

        # Araç kuyrukları: her araç gerçek araba ikonu olarak gösterilir.
        for lane in range(self.num_lanes):
            direction = lane // 3
            movement = lane % 3
            q = int(self.vehicle_q[lane])
            for j in range(min(q, 6)):
                x, y = self._vehicle_position(direction, movement, j)
                is_emergency = lane == self.emergency_lane and j == 0
                self._draw_car(ax, x, y, direction, emergency=is_emergency)
            tx, ty = self._lane_label_position(direction, movement)
            ax.text(tx, ty, f"{lane_name(lane)}  ×{q}", fontsize=7.3, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#b0b0b0", alpha=0.93), zorder=35)
            if lane == self.closed_lane:
                cx, cy = self._vehicle_position(direction, movement, 2)
                ax.add_patch(Circle((cx, cy), 0.30, facecolor="#a91d1d", edgecolor="white", linewidth=1.4, zorder=40))
                ax.text(cx, cy, "X", color="white", fontsize=14, fontweight="bold", ha="center", va="center", zorder=41)

        # Yaya yoğunluğu: yayalar kaldırımda ya da yaya fazında zebra üzerinde yürür şeklinde görünür.
        ped_green = self.current_phase == 4 and self.signal_mode == "green"
        for idx in range(self.num_ped_crossings):
            pcount = int(self.ped_q[idx])
            for j in range(min(pcount, 6)):
                x, y, facing = self._pedestrian_position(idx, j, ped_green)
                self._draw_pedestrian(ax, x, y, facing)
            lx, ly = self._pedestrian_label_position(idx)
            avg_wait = self._current_ped_wait_vector()[idx]
            ax.text(lx, ly, f"Yaya {idx + 1}: {pcount} | {avg_wait:.0f} sn",
                    fontsize=7.3, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#d7f0dc" if ped_green else "white",
                              ec="#9cb39f", alpha=0.94), zorder=35)

        closure_text = "Yok" if self.closed_lane < 0 else lane_name(self.closed_lane)
        emergency_text = "Yok" if self.emergency_lane < 0 else f"{lane_name(self.emergency_lane)} (AMBULANS)"
        subtitle = (
            f"t={self.time:03d} sn | Faz: {PHASES[self.current_phase]['name']} | "
            f"Acil araç: {emergency_text} | Kapalı şerit: {closure_text}"
        )
        ax.set_title("DRL Tabanlı Akıllı Kavşak", fontsize=16, fontweight="bold", pad=12)
        ax.text(0, -5.68, subtitle, fontsize=9.2, ha="center",
                bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#c6c6c6", alpha=0.97), zorder=50)
        ax.text(
            0, 5.58,
            f"Araç kuyruğu: {int(self.vehicle_q.sum())} | Yaya kuyruğu: {int(self.ped_q.sum())} | "
            f"Acil bekleme: {self.emergency_wait:.0f} sn | Override: {self.last_override_reason}",
            fontsize=9.0, ha="center", zorder=50,
        )

        fig.tight_layout(pad=0.35)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return rgb

    @staticmethod
    def _lane_label_position(direction: int, movement: int) -> Tuple[float, float]:
        lane_offset = (movement - 1) * 0.65
        if direction == 0:
            return lane_offset, 5.66
        if direction == 1:
            return -lane_offset, -5.66
        if direction == 2:
            return 5.58, -lane_offset
        return -5.58, lane_offset

    @staticmethod
    def _vehicle_position(direction: int, movement: int, index: int) -> Tuple[float, float]:
        """index=0 kavşağa en yakın araçtır; diğerleri geride sıralanır."""
        lane_offset = (movement - 1) * 0.65
        spacing = 0.64
        if direction == 0:
            return lane_offset, 2.40 + index * spacing
        if direction == 1:
            return -lane_offset, -2.40 - index * spacing
        if direction == 2:
            return 2.40 + index * spacing, -lane_offset
        return -2.40 - index * spacing, lane_offset

    @staticmethod
    def _pedestrian_position(crossing: int, index: int, walking: bool) -> Tuple[float, float, int]:
        row, col = divmod(index, 3)
        if crossing == 0:  # üst geçit, yatay yürüme
            return (-0.86 + col * 0.56, 1.80 if walking else 2.38 + row * 0.34, 1)
        if crossing == 1:  # alt geçit
            return (0.86 - col * 0.56, -1.80 if walking else -2.38 - row * 0.34, -1)
        if crossing == 2:  # sağ geçit, dikey yürüme
            return (1.80 if walking else 2.38 + row * 0.34, -0.86 + col * 0.56, 1)
        return (-1.80 if walking else -2.38 - row * 0.34, 0.86 - col * 0.56, -1)

    @staticmethod
    def _pedestrian_label_position(crossing: int) -> Tuple[float, float]:
        return {0: (0, 2.88), 1: (0, -2.88), 2: (3.08, 0), 3: (-3.08, 0)}[crossing]


# ============================================================
# 3) DQN
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, float, float]] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, discount) -> None:
        self.memory.append((state, action, reward, next_state, float(done), discount))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, discounts = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(discounts, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def epsilon_by_step(step: int) -> float:
    ratio = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + ratio * (EPSILON_END - EPSILON_START)


def select_action(model: DQN, state: np.ndarray, epsilon: float, action_size: int) -> int:
    if random.random() < epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        tensor_state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(torch.argmax(model(tensor_state), dim=1).item())


def train_dqn() -> Tuple[DQN, List[float], List[float]]:
    env = TrafficIntersectionEnv(seed=SEED)
    online = DQN(env.observation_size, env.action_size).to(DEVICE)
    target = DQN(env.observation_size, env.action_size).to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = optim.Adam(online.parameters(), lr=LEARNING_RATE)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    global_step = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset(seed=SEED + episode)
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            epsilon = epsilon_by_step(global_step)
            action = select_action(online, state, epsilon, env.action_size)
            next_state, reward, done, info = env.step(action)
            discount = GAMMA_PER_SECOND ** info["elapsed"]
            replay.add(state, action, reward, next_state, done, discount)
            state = next_state
            total_reward += reward
            global_step += 1

            if len(replay) >= TRAIN_AFTER:
                states, actions, rewards, next_states, dones, discounts = replay.sample(BATCH_SIZE)
                q_values = online(states).gather(1, actions)
                with torch.no_grad():
                    # Double DQN: aksiyon online ağdan, değer target ağdan alınır.
                    next_actions = torch.argmax(online(next_states), dim=1, keepdim=True)
                    next_q_values = target(next_states).gather(1, next_actions)
                    td_target = rewards + (1.0 - dones) * discounts * next_q_values

                loss = nn.SmoothL1Loss()(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()
                losses.append(float(loss.item()))

                if global_step % TARGET_UPDATE_EVERY == 0:
                    target.load_state_dict(online.state_dict())

        episode_rewards.append(total_reward)
        episode_losses.append(float(np.mean(losses)) if losses else 0.0)

        if episode == 1 or episode % 25 == 0:
            print(
                f"Bölüm {episode:03d}/{EPISODES} | ödül={total_reward:8.1f} | "
                f"epsilon={epsilon_by_step(global_step):.3f} | "
                f"araç={env.total_passed_vehicles} | yaya={env.total_crossed_pedestrians} | "
                f"acil geçti={env.emergency_served}"
            )

    torch.save(online.state_dict(), OUTPUT_DIR / "traffic_dqn_model.pt")
    return online, episode_rewards, episode_losses


# ============================================================
# 4) GRAFİK, DEĞERLENDİRME VE GIF
# ============================================================

def save_training_plot(rewards: List[float], losses: List[float]) -> Path:
    path = OUTPUT_DIR / "training_curve.png"
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=135)
    ax.plot(rewards, label="Bölüm ödülü")
    if len(rewards) >= 20:
        moving = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        ax.plot(range(19, len(rewards)), moving, label="20 bölümlük hareketli ortalama", linewidth=2.4)
    ax.set_title("DQN Eğitim Ödülü")
    ax.set_xlabel("Bölüm")
    ax.set_ylabel("Toplam ödül")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def evaluate_and_create_gif(model: DQN, seed: int = 2026) -> Tuple[Path, Dict]:
    # Demo süresi eğitim bölümünden kısa tutulur; GIF hem okunur hem de dosya boyutu kontrol altında kalır.
    env = TrafficIntersectionEnv(seed=seed, max_sim_seconds=GIF_EVALUATION_SECONDS)
    state = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    frames = [env.render_rgb()]
    decisions = []

    def add_frame(frame: np.ndarray) -> None:
        # Aynı kareyi tekrar eklememek için sadece belirlenen maksimum sayıya kadar sakla.
        if len(frames) < GIF_MAX_FRAMES:
            frames.append(frame)

    while not done:
        action = select_action(model, state, epsilon=0.0, action_size=env.action_size)
        next_state, reward, done, info = env.step(
            action,
            frame_callback=add_frame,
            frame_every_seconds=GIF_FRAME_EVERY_SECONDS,
        )
        total_reward += reward
        selected_phase, selected_duration = env.decode_action(action)
        decisions.append(
            {
                "time": env.time,
                "phase": PHASES[selected_phase]["name"],
                "duration": selected_duration,
                "override": info["override_reason"],
            }
        )
        state = next_state

    # Pillow ile milisaniye seviyesinde süre yazılır; bazı GIF oynatıcılarında hızlanma sorunu yaşanmaz.
    gif_path = OUTPUT_DIR / "traffic_control_demo_slow.gif"
    pil_frames = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
        disposal=2,
        optimize=False,
    )

    summary = {
        "toplam_odul": round(total_reward, 2),
        "toplam_gecen_arac": env.total_passed_vehicles,
        "toplam_gecen_yaya": env.total_crossed_pedestrians,
        "acil_arac_gecisi": env.emergency_served,
        "faz_degisimi": env.total_phase_switches,
        "son_arac_kuyrugu": int(env.vehicle_q.sum()),
        "son_yaya_kuyrugu": int(env.ped_q.sum()),
        "gif_kare_sayisi": len(frames),
        "gif_kare_suresi_ms": GIF_FRAME_DURATION_MS,
        "kararlar": decisions,
    }
    return gif_path, summary


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Cihaz: {DEVICE}")
    print("DQN eğitimi başlıyor...")
    model, rewards, losses = train_dqn()

    plot_path = save_training_plot(rewards, losses)
    gif_path, summary = evaluate_and_create_gif(model)

    print("\n===== DEĞERLENDİRME SONUCU =====")
    for key, value in summary.items():
        if key != "kararlar":
            print(f"{key}: {value}")

    print(f"\nModel: {OUTPUT_DIR / 'traffic_dqn_model.pt'}")
    print(f"Eğitim grafiği: {plot_path}")
    print(f"GIF: {gif_path}")

    # Colab / Jupyter ekranında GIF'i doğrudan gösterir.
    try:
        from IPython.display import Image, display
        display(Image(filename=str(gif_path)))
    except Exception:
        pass


if __name__ == "__main__":
    main()

files.download("outputs_drl_traffic/training_curve.png")
display(Image(filename="outputs_drl_traffic/training_curve.png"))

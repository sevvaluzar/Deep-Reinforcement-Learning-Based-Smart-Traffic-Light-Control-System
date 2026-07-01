# Deep Reinforcement Learning for Adaptive Traffic Signal Control

A custom **Double Deep Q-Network (Double DQN)** implementation for adaptive traffic signal control at a four-way intersection. The agent learns to select traffic phases and green-light durations while balancing vehicle queues, pedestrian demand, emergency vehicle priority, and temporary lane closures.

> This project uses a custom Python simulation environment and does not require SUMO. It is designed to be easy to run in Google Colab and can also be adapted for local execution.

<p align="center">
  <img src="traffic_control_demo_slow.gif" alt="DRL traffic intersection simulation" width="560" />
</p>

<p align="center"><em>Simulation output: vehicle, pedestrian, emergency vehicle, and lane-closure events are visualized during policy evaluation.</em></p>

---

## Project Highlights

- Four-way intersection with **12 approach lanes**
  - North, South, East, and West approaches
  - Dedicated straight, left-turn, and right-turn movements
- Pedestrian crossings with queue and waiting-time tracking
- Adaptive green-light durations: **8, 12, or 16 seconds**
- Dedicated pedestrian phase
- Ambulance / fire-truck priority mechanism
- Emergency safety override after a maximum waiting threshold
- Temporary lane-closure scenario
- Yellow transition phase between conflicting signal phases
- Time-varying traffic demand, including a peak-traffic interval
- Double DQN with experience replay and a target network
- GIF-based visualization and training-reward plotting

---

## Simulation Environment

The traffic intersection is modeled as a Markov Decision Process (MDP).

| MDP Component | Design |
|---|---|
| **Agent** | Double DQN traffic-signal controller |
| **Environment** | Custom four-way road intersection |
| **State** | Vehicle queues, average waiting times, pedestrian queues, current phase, lane closure, emergency vehicle location, demand profile, etc. |
| **Action** | Select one of five phases and one of three green durations |
| **Reward** | Penalizes queues and waiting time; rewards cleared vehicles and pedestrians; strongly penalizes emergency delay |
| **Episode** | Up to 360 simulation seconds |

### Intersection Phases

| Phase | Description |
|---:|---|
| 0 | North–South straight and right-turn movements |
| 1 | East–West straight and right-turn movements |
| 2 | North–South protected left turns |
| 3 | East–West protected left turns |
| 4 | Pedestrian crossing phase |

The action space has **15 actions**:

```text
5 signal phases × 3 green durations (8 s, 12 s, 16 s)
```

---

## State Space

The environment uses a 67-dimensional observation vector:

```text
12 vehicle queue values
12 lane-level average vehicle waiting times
 4 pedestrian queue values
 4 pedestrian average waiting times
 5 current-phase one-hot values
 1 current-phase elapsed time
12 lane-closure one-hot values
 1 remaining lane-closure time
12 emergency-vehicle lane one-hot values
 1 emergency waiting time
 3 traffic-demand profile values
--------------------------------
67 total state features
```

---

## Reward Function

The reward function is multi-objective. It aims to reduce congestion and waiting time while preserving pedestrian and emergency-vehicle priority.

```text
Reward =
  - vehicle queue penalty
  - pedestrian queue penalty
  - vehicle waiting-time penalty
  - pedestrian waiting-time penalty
  + vehicle throughput reward
  + pedestrian throughput reward
  - emergency vehicle delay penalty
  - lane-closure congestion penalty
```

Important design choices:

- Pedestrian queues receive a stronger penalty than vehicle queues.
- Emergency vehicles receive a high waiting-time penalty.
- A safety override activates when the emergency vehicle waits too long.
- A pedestrian override activates when a pedestrian crossing reaches its maximum waiting threshold.

---

## Safety and Priority Rules

The learned policy is protected by a rule-based safety layer:

- **Emergency vehicle priority:** if an ambulance or fire truck waits for 10 seconds, the phase serving its lane is forced.
- **Pedestrian fairness:** if any pedestrian crossing waits for 34 seconds, the pedestrian phase is forced.
- **Yellow transition:** every phase switch includes a 3-second yellow-light transition.
- **Lane closures:** a randomly selected lane can be closed temporarily during an episode.

---

## Double DQN Architecture

The model is a fully connected neural network:

```text
Input:  67 state features
Hidden: 160 neurons + ReLU
Hidden: 160 neurons + ReLU
Output: 15 Q-values
```

The training procedure includes:

- Epsilon-greedy exploration
- Replay buffer with a capacity of 50,000 transitions
- Target network updates every 400 training steps
- Huber loss (`SmoothL1Loss`)
- Gradient clipping
- Per-second discount adjustment for actions with different durations

---

## Training Result

The training curve below shows the episode reward and its 20-episode moving average. A rising reward trend indicates that the policy is learning to manage queues and priority events more effectively.

<p align="center">
  <img src="training_curve.png" alt="DQN training reward curve" width="780" />
</p>

---

## Installation

### Google Colab

Upload `DRL_FinalProject.py` to Colab and run:

```python
!pip install numpy torch matplotlib imageio pillow
!python /content/DRL_FinalProject.py
```

The script automatically creates an `outputs_drl_traffic/` directory and displays the generated GIF in the notebook.

### Local Python Environment

```bash
pip install numpy torch matplotlib imageio pillow
python DRL_FinalProject.py
```

> **Note:** The provided version includes `google.colab` import and download commands at the end for Colab convenience. For local execution, comment out or remove these lines:
>
> ```python
> from google.colab import files
> files.download("outputs_drl_traffic/training_curve.png")
> ```

---

## Main Hyperparameters

| Parameter | Value |
|---|---:|
| Episodes | 500 |
| Maximum simulation time | 360 s |
| Batch size | 128 |
| Replay capacity | 50,000 |
| Learning rate | 0.001 |
| Discount factor per second | 0.995 |
| Target network update interval | 400 steps |
| Epsilon start / end | 1.0 / 0.05 |
| Green durations | 8 s, 12 s, 16 s |
| Yellow duration | 3 s |
| Maximum pedestrian waiting time | 34 s |
| Maximum emergency waiting time | 10 s |

---

## Generated Outputs

After training, the following files are produced in `outputs_drl_traffic/`:

```text
traffic_dqn_model.pt           # Trained Double DQN model weights
training_curve.png             # Episode reward and moving-average graph
traffic_control_demo_slow.gif  # Visual policy demonstration
```

---

## Repository Structure

```text
.
├── DRL_FinalProject.py
├── README.md
├── assets/
│   ├── traffic_control_demo_slow.gif
│   └── training_curve.png
└── outputs_drl_traffic/       # Created after execution
    ├── traffic_dqn_model.pt
    ├── training_curve.png
    └── traffic_control_demo_slow.gif
```

---

## Possible Extensions

- Compare Double DQN with fixed-time and heuristic controllers
- Add `training_loss.png`, queue-over-time, throughput, and emergency-response plots
- Model public-transport priority for buses or trams
- Add weather conditions, road accidents, and multi-lane blockage scenarios
- Extend the environment to multiple connected intersections
- Replace the custom simulation with SUMO and TraCI integration
- Apply PPO, Dueling DQN, or multi-agent reinforcement learning

---

## Limitations

This is a controlled simulation, not a deployment-ready traffic system. The vehicle arrivals, lane capacities, and pedestrian behavior are simplified. Real-world deployment would require calibrated traffic data, safety validation, traffic-engineering constraints, and integration with physical signal infrastructure.

---

## License

This project is provided for academic and educational use.

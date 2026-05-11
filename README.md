# Lunar Lander — Neural Network Controller

A Pygame-based lunar lander simulation where a feedforward neural network, trained entirely from scratch (no ML libraries), learns to autonomously pilot the spacecraft to the landing pad.

Built as the CE889 Neural Networks & Deep Learning assignment at the University of Essex.

---

## Project Purpose

The project has two distinct phases:

1. **Data Collection** — A human pilot plays the game manually. Every frame, the lander's positional error relative to the landing pad is recorded alongside the pilot's control inputs (thrust and turning). This produces a CSV training dataset.

2. **Neural Network Training** — `NeuralNetworkFinal.py` reads the CSV, trains a 2-input → 8-hidden → 2-output multilayer perceptron using backpropagation with momentum, and saves the learned weights to `lander.txt`.

3. **Autonomous Flight** — The game reloads `lander.txt` at startup and uses the trained model to control the lander in real time, replacing human input entirely.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Game engine | Pygame |
| Neural network | Custom implementation (pure Python, no NumPy/TensorFlow) |
| Activation function | Sigmoid |
| Optimiser | Stochastic Gradient Descent + Momentum |
| Configuration | Custom `.con` file parser |

---

## Architecture

### Neural Network

```
Input Layer        Hidden Layer       Output Layer
─────────────      ────────────────   ─────────────
x_pos_to_target ─→ 8 sigmoid neurons ─→ y_velocity (thrust)
y_pos_to_target ─→                   ─→ x_velocity (turning)
```

- Inputs are min-max normalised to [0, 1] before inference.
- Outputs are denormalised back to game-space velocities.
- Training uses 70 / 15 / 15 train / validation / test split with early stopping (patience = 20 epochs, evaluated every 5 epochs).

### Game Modules

```
Main.py           ← Entry point; loads Config.con
GameLoop.py       ← Main game loop; handles all three modes
Lander.py         ← Lander sprite, physics, collision
Surface.py        ← Terrain and landing pad generation
GameLogic.py      ← Physics update (velocity, gravity)
Controller.py     ← Input abstraction (human or AI)
EventHandler.py   ← Keyboard event dispatch
DataCollection.py ← Records (x_err, y_err, vel_y, vel_x) per frame
NeuralNetHolder.py← Loads lander.txt and wraps inference
NeuralNetworkFinal.py ← Standalone trainer script
```

---

## Setup

### Prerequisites

- Python 3.8 or later
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### (Optional) Train the network

If `lander.txt` is not present, or you want to retrain:

1. Run the game, select **Data Collection**, and land the lander manually several times to build up `ce889_dataCollection.csv`.
2. Then run the trainer:

```bash
python NeuralNetworkFinal.py
```

This will print per-epoch RMSE and write `lander.txt` when training is complete.

---

## Running the Game

```bash
python Main.py
```

The main menu offers three modes:

| Mode | Description |
|---|---|
| **Play Game** | Manual control — arrow keys to thrust and turn |
| **Data Collection** | Manual control — flight data is saved to CSV on landing |
| **Neural Net** | AI autopilot using the trained model |

### Controls (manual modes)

| Key | Action |
|---|---|
| `↑` | Thrust upward |
| `←` | Rotate left |
| `→` | Rotate right |

---

## Configuration

Edit `Files/Config.con` to change display settings:

```
SCREEN_HEIGHT = 1000, SCREEN_WIDTH = 1600
LANDER_IMG_PATH = Sprites/rocket_lander.png
BACKGROUND_IMG_PATH = Sprites/BackGround.bmp
FULLSCREEN = FALSE
ALL_DATA = FALSE
```

Set `FULLSCREEN = TRUE` to run fullscreen (Windows only). `ALL_DATA = TRUE` switches data collection to an extended feature set (speed, angle, distance to surface).

---

## Project Structure

```
.
├── Main.py                   # Entry point
├── GameLoop.py               # Core game loop
├── Lander.py                 # Lander physics & sprite
├── Surface.py                # Terrain generation
├── GameLogic.py              # Physics engine
├── CollisionUtility.py       # Line-segment intersection & collision helpers
├── Controller.py             # Input abstraction
├── EventHandler.py           # Event handling
├── DataCollection.py         # Training data recorder
├── NeuralNetHolder.py        # Inference wrapper
├── NeuralNetworkFinal.py     # Standalone trainer
├── Vector.py                 # 2D vector utility
├── MainMenu.py               # Main menu UI
├── ResultMenu.py             # Win/lose screen UI
├── Files/
│   └── Config.con            # Game configuration
├── Sprites/                  # Game images
├── requirements.txt
└── lander.txt                # Trained model weights (generated)
```

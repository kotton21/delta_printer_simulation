# RobotArm

Kinematics simulation for non-standard delta robots. Includes:

- A PySide6 GUI (`main.py`) that visualizes a linear delta robot's kinematics, letting you drag tower heights or the effector position and see the resulting pose in real time.
- `linear_delta_kinematics.py` / `conical_delta_kinematics.py`: standalone scripts that derive forward/inverse kinematics symbolically (sympy) and pickle the resulting model for reuse.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

GUI visualizer:

```bash
python3 main.py
```

Regenerate a kinematics model (writes a `.pkl` file):

```bash
python3 linear_delta_kinematics.py
python3 conical_delta_kinematics.py
```

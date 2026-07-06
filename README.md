# RobotArm

Kinematics simulation for non-standard delta robots. Includes:

- A PySide6 GUI (`main.py`) that visualizes a linear delta robot's kinematics, letting you drag tower heights or the effector position and see the resulting pose in real time.
- `linear_delta_kinematics.py` / `conical_delta_kinematics.py` / `adjustable_conical_delta_kinematics.py`: standalone scripts that derive forward/inverse kinematics symbolically (sympy) and pickle the resulting model for reuse.
- The adjustable-rod conical delta keeps each tower's rod (arm) length as an independent free parameter instead of one fixed constant shared by all three, exposing 3 real-time arm-length sliders alongside the 3 tower-rail sliders. This is a real redundant DOF per tower: for a fixed effector target, IK normally uses whatever arm length is already set on each tower, and only nudges the tower(s) that need it (to the minimum extent needed, leaving the others untouched) when the target isn't reachable at their current length -- e.g. shortening an arm can increase the maximum reachable on-axis height for this geometry.

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
python3 adjustable_conical_delta_kinematics.py
```

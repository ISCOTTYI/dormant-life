# Dormant Life

## Installation Instructions
* Create and activate virtual environment using `python3 -m venv ./venv && source venv/bin/activate`.
* Install requirements using `pip install -r requirements.txt`.

## Running a Simulation
* Run a simulation using `python simulation.py`.
* The initial grid can be randomly generated in line 8 or alternatively specified by hand in line 9 in `simulation.py`.
* A `1` in the grid encodes `ALIVE`, a `0` encodes `DEAD`, a `2` encodes `DORM`. The respective aliases are also imported and may be used.

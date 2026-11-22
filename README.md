
# Reinforcement Learning Network Simulator

 A network simulator which performs simulated packet routing across a multi-node network which uses Reinforcement Learning to reduce congestion and improve throughput compared to a standard Dijkstra algorithm approach.


## Features

- Easily customizable network topology and simulator settings allowing nodes/link to be added or removed and network parameters such as tick rate to be easily adjusted via. .YAML files
- Simulation of data transfers across a multi-node network using three modes: 
     1. Baseline (Routing based on Djikstra's algorithm alone),
     2. RL-Enhanced (RL-agent routing used where more efficient than Djikstra's)
     3. Evaluation (RL-agent routing + auto-evaluation of transfer performance every 30 seconds)
- Structured report generation for all simulated data
- PyQT-based Dashboard for visualization of data transfers, average network utilization, node usage and controls for network transfers & RL-usage


## Installation

Clone the repo and move to the project directory

```bash
  git clone https://github.com/Dylan-M-afk/RL-Network-Simulator
  cd RL-Network-Simulator/project
```

Install pip required packages
```
pip install -r requirements.txt
```

    
## Usage/Examples

Start the visualization tool
```
python module_c/QtDashboard.py 
```

Use run.py to start the simulator

```
python run.py
```

Use the menu to selected the desired option

```
══════════════════════════════════════════════════════════
  RL Network Simulation — Launcher
══════════════════════════════════════════════════════════
  history.csv  : 452 ticks total  (380 RL,  72 dijkstra)
  qtable.pkl   : exists — agent will resume from saved state
  tick speed   : 3.0s per tick
──────────────────────────────────────────────────────────

  Select an option:

    1. Simulator only                  (Dijkstra baseline, no RL)
    2. Simulator + RL agent            (live routing with Q-learning)
    3. Simulator + RL agent + evaluate (auto-report every 30s)
    4. Train offline                   (rebuild Q-table from history.csv)
    5. Evaluate                        (print report, no simulation)
    6. Exit

  Enter choice [1-6]: 
```

Example Usage of option 2
```
============================================================
Module B — RL Agent (online Q-learning)
============================================================
Graph: 11 nodes  14 edges
Shared dir: /RL-Network-Simulator/project/shared
State file: /RL-Network-Simulator/project/shared/state.json
Decision file: /RL-Network-Simulator/project/shared/routing_decision.json

Loaded Q-table from /RL-Network-Simulator/project/module_b/qtable.pkl  (759 states)
Agent running — press Ctrl+C to stop
------------------------------------------------------------
Tick    0 | strategy=dijkstra             | util= 58.7% avg  225.7% max |                   congested=4/14 | delay=8.0ms avg
Tick    0 | sim_tick=1    | flows=25 | util=65.0% avg | delay=7.6ms | congested=6/14 | sample_reward=-2.503 | ε=0.2997 | states=759
Tick    1 | strategy=dijkstra             | util= 65.0% avg  209.8% max |                   congested=6/14 | delay=7.6ms avg
Tick    1 | sim_tick=2    | flows=25 | util=48.9% avg | delay=4.1ms | congested=1/14 | sample_reward=-0.412 | ε=0.2994 | states=759
Tick    2 | strategy=rl                   | util= 48.9% avg   76.0% max |                   congested=1/14 | delay=4.1ms avg
Tick    2 | sim_tick=3    | flows=25 | util=56.5% avg | delay=8.0ms | congested=3/14 | sample_reward=-1.419 | ε=0.2991 | states=759
Tick    3 | strategy=rl                   | util= 56.5% avg  237.9% max |                   congested=3/14 | delay=8.0ms avg
Tick    3 | sim_tick=4    | flows=25 | util=64.8% avg | delay=9.1ms | congested=3/14 | sample_reward=-0.401 | ε=0.2988 | states=760
Tick    4 | strategy=rl                   | util= 64.8% avg  267.9% max |                   congested=3/14 | delay=9.1ms avg
Tick    4 | sim_tick=5    | flows=25 | util=69.9% avg | delay=7.6ms | congested=7/14 | sample_reward=-1.525 | ε=0.2985 | states=760
Tick    5 | strategy=rl                   | util= 69.9% avg  144.5% max |                   congested=7/14 | delay=7.6ms avg
Tick    5 | sim_tick=6    | flows=25 | util=59.4% avg | delay=7.5ms | congested=4/14 | sample_reward=-0.533 | ε=0.2982 | states=760
```


## Screenshots

Network Visualization

![Network Visualization](https://i.imgur.com/tSYQF7n.png)

Network Usage
![Network Usage](https://i.imgur.com/21QHzyj.png)

Node Usage
![Node Usage](https://i.imgur.com/yAtp7Go.png)
## Authors

- [Dylan Mercer](https://github.com/Dylan-M-afk)
- [Julia Forward](https://github.com/UnusualFrog)
- [Parker-Wallace](https://github.com/Parker-Wallace)

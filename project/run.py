# ========== Launcher ==========

# Interactive launcher for the RL Network Simulation project.
# Handles starting the simulator, RL agent, and evaluation report
# as coordinated subprocesses from a single terminal.
#
# Usage:
#   python run.py

import os
import sys
import time
import signal
import subprocess
import threading
import queue
import re


# ─── Paths ────────────────────────────────────────────────────────────────────

LAUNCHER_DIR  = os.path.dirname(os.path.abspath(__file__))
MODULE_A_DIR  = os.path.join(LAUNCHER_DIR, "module_a")
MODULE_B_DIR  = os.path.join(LAUNCHER_DIR, "module_b")
SIMULATOR     = os.path.join(MODULE_A_DIR, "simulator.py")
RL_AGENT      = os.path.join(MODULE_B_DIR, "rl_agent.py")
TRAINER       = os.path.join(MODULE_B_DIR, "train.py")
EVALUATOR     = os.path.join(MODULE_B_DIR, "evaluate.py")
QTABLE        = os.path.join(MODULE_B_DIR, "qtable.pkl")
SHARED_DIR    = os.path.join(LAUNCHER_DIR, "shared")
HISTORY_PATH  = os.path.join(SHARED_DIR, "history.csv")

PYTHON        = sys.executable  # Use the same python env that launched this script
TOPOLOGY_PATH = os.path.join(LAUNCHER_DIR, "module_a", "topology.yaml")
NETWORK_PARAMETERS_PATH = os.path.join(LAUNCHER_DIR, "module_a", "network_parameters.yaml")

# Minimum tick interval when the RL agent is running
RL_MIN_TICK   = 1

# ─── Tick speed helpers ──────────────────────────────────────────────────────────────────

def read_current_tick_speed():
    """Read tick_interval_seconds from network_parameters.yaml."""
    try:
        import yaml
        with open(NETWORK_PARAMETERS_PATH) as f:
            config = yaml.safe_load(f)
        return config["simulation"].get("tick_interval_seconds", 5)
    except Exception:
        return 5

def write_tick_speed(seconds):
    """Overwrite tick_interval_seconds in topology.yaml, preserving all other content."""
    import yaml
    with open(NETWORK_PARAMETERS_PATH) as f:
        config = yaml.safe_load(f)
    config["simulation"]["tick_interval_seconds"] = seconds
    with open(NETWORK_PARAMETERS_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def prompt_tick_speed(with_rl):
    """
    Ask the user for a tick interval before launching.
    Returns the chosen float, or None to keep the current value.
    with_rl: if True, enforce RL_MIN_TICK floor and warn about fast speeds.
    """
    current = read_current_tick_speed()
    floor   = RL_MIN_TICK if with_rl else 0

    print(f"\n  Current tick speed : {current}s per tick")
    if with_rl:
        print(f"  Minimum with RL    : {floor}s  (agent needs time to read/write files)")
    else:
        print("  Set to 0 for maximum speed (useful for generating baseline history quickly)")
    print()

    raw = input(f"  Enter tick interval in seconds [Enter to keep {current}s]: ").strip()

    if raw == "":
        return None  # keep existing

    try:
        val = float(raw)
    except ValueError:
        print("  Invalid input — keeping current speed.")
        time.sleep(1)
        return None

    if val < 0:
        print("  Cannot be negative — keeping current speed.")
        time.sleep(1)
        return None

    if with_rl and val < floor:
        print(f"  Clamped to {floor}s minimum (RL agent requires at least {floor}s per tick).")
        time.sleep(1)
        val = floor

    return val

# ─── Helpers ──────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def sep(char="─", width=58):
    print(char * width)

def header():
    clear()
    sep("═")
    print("  RL Network Simulation — Launcher")
    sep("═")

def status_line():
    """Print a quick status summary of existing data and model files."""
    history_rows = 0
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history_rows = sum(1 for line in f) - 1  # subtract header

    qtable_exists = os.path.exists(QTABLE)
    rl_rows = 0
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("strategy") == "rl":
                    rl_rows += 1

    tick_speed = read_current_tick_speed()
    print(f"  history.csv  : {history_rows} ticks total  ({rl_rows} RL,  {history_rows - rl_rows} dijkstra)")
    print(f"  qtable.pkl   : {'exists — agent will resume from saved state' if qtable_exists else 'not found — agent will start fresh'}")
    print(f"  tick speed   : {tick_speed}s per tick")
    sep()

def wait_for_key():
    input("\n  Press Enter to return to the menu...")

def run_subprocess(cmd, cwd, extra_env=None):
    """
    Run a command as a subprocess in the given working directory.
    Returns the Popen object so the caller can manage it.
    extra_env: optional dict of additional environment variables to pass.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(cmd, cwd=cwd, env=env)

def wait_for_procs(procs, labels):
    """
    Block until all processes finish or the user hits Ctrl+C.
    On Ctrl+C, sends SIGTERM (Windows: terminate) to each process gracefully.
    """
    print()
    sep()
    running_labels = ", ".join(labels)
    print(f"  Running: {running_labels}")
    print("  Press Ctrl+C to stop.")
    sep()

    try:
        while True:
            # Check if all processes have finished on their own
            all_done = all(p.poll() is not None for p in procs)
            if all_done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  Stopping processes...")
        for p in procs:
            if p.poll() is None:
                if os.name == "nt":
                    p.terminate()
                else:
                    p.send_signal(signal.SIGTERM)
        # Give them up to 5 seconds to shut down cleanly
        deadline = time.time() + 5
        for p in procs:
            remaining = max(0, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()

    print("  All processes stopped.")

def run_ordered_subprocess(cmd, cwd, label, output_queue, extra_env=None):
    """Run a subprocess and forward its output to the ordered-output queue."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    def read_output():
        try:
            for line in proc.stdout:
                output_queue.put((label, line))
        finally:
            proc.stdout.close()
            output_queue.put((label, None))

    threading.Thread(target=read_output, daemon=True).start()
    return proc


def wait_for_procs_ordered(procs, labels, output_queue, evaluate=False,
                           show_header=True, stop_text="Stopping processes...",
                           done_text="All processes stopped."):
    """Display simulator and RL tick lines in simulation-tick order."""
    if show_header:
        print()
        sep()
        running_labels = ", ".join(labels)
        print(f"  Running: {running_labels}")
        print("  Press Ctrl+C to stop.")
        sep()

    tick_buffers = {"Simulator": {}, "RL Agent": {}}
    next_tick = 0
    finished_readers = set()
    last_eval = time.time()
    tick_pattern = re.compile(r"^\s*Tick\s+(\d+)\s*\|")

    def display_ready_ticks(force=False):
        nonlocal next_tick

        # Hold each process's tick lines until both have reported the same
        # tick number, then print the simulator line followed by the RL line.
        while all(next_tick in tick_buffers[label] for label in labels):
            print(tick_buffers["Simulator"].pop(next_tick), end="", flush=True)
            print(tick_buffers["RL Agent"].pop(next_tick), end="", flush=True)
            next_tick += 1

        if force:
            remaining_ticks = sorted(
                tick for per_process in tick_buffers.values()
                for tick in per_process
            )
            for tick in remaining_ticks:
                for label in labels:
                    line = tick_buffers[label].pop(tick, None)
                    if line is not None:
                        print(line, end="", flush=True)

    def handle_output(label, line):
        if line is None:
            finished_readers.add(label)
            return

        tick_match = tick_pattern.match(line)
        if tick_match and label in tick_buffers:
            tick_buffers[label][int(tick_match.group(1))] = line
        else:
            print(line, end="", flush=True)

        display_ready_ticks()

    try:
        while len(finished_readers) < len(procs):
            try:
                label, line = output_queue.get(timeout=0.1)
                handle_output(label, line)
            except queue.Empty:
                pass

            if evaluate and time.time() - last_eval >= 30:
                print("\n  [Auto-evaluate]")
                eval_proc = run_subprocess([PYTHON, EVALUATOR], cwd=MODULE_B_DIR)
                eval_proc.wait()
                last_eval = time.time()

        display_ready_ticks(force=True)

    except KeyboardInterrupt:
        print(f"\n  {stop_text}")
        for proc in procs:
            if proc.poll() is None:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)

        # Give them up to 5 seconds to shut down cleanly
        deadline = time.time() + 5
        for proc in procs:
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Continue reading until both subprocess output streams have closed.
        while len(finished_readers) < len(procs):
            try:
                label, line = output_queue.get(timeout=0.1)
                handle_output(label, line)
            except queue.Empty:
                pass

        display_ready_ticks(force=True)

    print(f"  {done_text}")

# ─── Menu options ─────────────────────────────────────────────────────────────

def option_simulator_only():
    """Start only the Module A simulator."""
    header()
    print("  Starting: Simulator only (Module A)")
    print("  Module A will write state.json and history.csv to /shared.")
    print("  Routing will use Dijkstra — no RL agent.")
    sep()
    new_speed = prompt_tick_speed(with_rl=False)
    if new_speed is not None:
        write_tick_speed(new_speed)
        print(f"  Tick speed set to {new_speed}s")
    sep()
    sim = run_subprocess([PYTHON, SIMULATOR], cwd=MODULE_A_DIR)
    wait_for_procs([sim], ["Simulator"])
    wait_for_key()

def option_simulator_and_rl():
    """Start the simulator and RL agent together."""
    header()
    print("  Starting: Simulator + RL Agent (Module A + B)")
    sep()

    # Offer to run train.py first if qtable doesn't exist or user wants a refresh
    if not os.path.exists(QTABLE):
        print("  No qtable.pkl found.")
        print("  Running train.py to warm-start from history.csv first...")
        print()
        train_proc = run_subprocess([PYTHON, TRAINER], cwd=MODULE_B_DIR)
        train_proc.wait()
        print()
    else:
        print("  qtable.pkl found — RL agent will resume from saved Q-table.")
        print("  Tip: delete qtable.pkl before launching if you want a fresh start.")
        sep()

    new_speed = prompt_tick_speed(with_rl=True)
    if new_speed is not None:
        write_tick_speed(new_speed)
        print(f"  Tick speed set to {new_speed}s")
    tick_speed = read_current_tick_speed()
    sep()

    output_queue = queue.Queue()
    sim = run_ordered_subprocess(
        [PYTHON, SIMULATOR], MODULE_A_DIR, "Simulator", output_queue
    )

    # Small delay so the simulator writes its first state.json before the agent starts
    print()
    print("  Waiting 3 seconds for simulator to initialise...")
    time.sleep(3)

    # Pass tick interval to agent via env so it stays in sync with the simulator
    agent = run_ordered_subprocess(
        [PYTHON, RL_AGENT], MODULE_B_DIR, "RL Agent", output_queue,
        extra_env={"RL_TICK_INTERVAL": str(tick_speed)}
    )
    wait_for_procs_ordered([sim, agent], ["Simulator", "RL Agent"], output_queue)
    wait_for_key()

def option_train_only():
    """Run train.py standalone to rebuild the Q-table from history.csv."""
    header()
    print("  Running: Offline trainer (train.py)")
    print("  Reads history.csv and saves qtable.pkl.")
    sep()
    train_proc = run_subprocess([PYTHON, TRAINER], cwd=MODULE_B_DIR)
    train_proc.wait()
    wait_for_key()

def option_evaluate():
    """Run evaluate.py and display the report inline."""
    header()
    print("  Running: Evaluation report (evaluate.py)")
    sep()
    eval_proc = run_subprocess([PYTHON, EVALUATOR], cwd=MODULE_B_DIR)
    eval_proc.wait()
    wait_for_key()

def option_simulator_and_evaluate():
    """Run the simulator and periodically re-run evaluate.py every N ticks."""
    header()
    print("  Starting: Simulator + live evaluation (Module A only, no RL)")
    print("  evaluate.py will re-run every 30 seconds automatically.")
    sep()

    sim = run_subprocess([PYTHON, SIMULATOR], cwd=MODULE_A_DIR)

    print("  Running: Simulator (Dijkstra baseline only)")
    print("  Press Ctrl+C to stop.")
    sep()

    try:
        last_eval = time.time()
        while sim.poll() is None:
            if time.time() - last_eval >= 30:
                print("\n  [Auto-evaluate]")
                eval_proc = run_subprocess([PYTHON, EVALUATOR], cwd=MODULE_B_DIR)
                eval_proc.wait()
                last_eval = time.time()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping...")
        if sim.poll() is None:
            sim.terminate() if os.name != "nt" else sim.kill()
            sim.wait()

    wait_for_key()

# ─── Main menu ────────────────────────────────────────────────────────────────

MENU = [
    ("Simulator only",                    option_simulator_only),
    ("Simulator + RL agent",              option_simulator_and_rl),
    ("Simulator + RL agent + evaluate",   None),   # handled inline below
    ("Train offline (rebuild Q-table)",   option_train_only),
    ("Evaluate (report only)",            option_evaluate),
    ("Exit",                              None),
]

def option_simulator_rl_and_evaluate():
    """Start simulator + RL agent, and re-run evaluate every 30 seconds."""
    header()
    print("  Starting: Simulator + RL Agent + live evaluation")
    sep()

    if not os.path.exists(QTABLE):
        print("  No qtable.pkl found — running train.py first...")
        train_proc = run_subprocess([PYTHON, TRAINER], cwd=MODULE_B_DIR)
        train_proc.wait()
        print()
    else:
        print("  qtable.pkl found — resuming from saved Q-table.")
        sep()

    new_speed = prompt_tick_speed(with_rl=True)
    if new_speed is not None:
        write_tick_speed(new_speed)
        print(f"  Tick speed set to {new_speed}s")
    tick_speed = read_current_tick_speed()
    sep()

    output_queue = queue.Queue()
    sim = run_ordered_subprocess(
        [PYTHON, SIMULATOR], MODULE_A_DIR, "Simulator", output_queue
    )
    print("  Waiting 3 seconds for simulator to initialise...")
    time.sleep(3)
    agent = run_ordered_subprocess(
        [PYTHON, RL_AGENT], MODULE_B_DIR, "RL Agent", output_queue,
        extra_env={"RL_TICK_INTERVAL": str(tick_speed)}
    )

    print()
    print("  Running: Simulator + RL Agent")
    print("  evaluate.py will re-run every 30 seconds.")
    print("  Press Ctrl+C to stop everything.")
    sep()

    wait_for_procs_ordered(
        [sim, agent], ["Simulator", "RL Agent"], output_queue,
        evaluate=True, show_header=False,
        stop_text="Stopping all processes...", done_text="All stopped."
    )
    wait_for_key()


def main():
    while True:
        header()
        status_line()
        print()
        print("  Select an option:\n")
        print("    1. Simulator only                  (Dijkstra baseline, no RL)")
        print("    2. Simulator + RL agent            (live routing with Q-learning)")
        print("    3. Simulator + RL agent + evaluate (auto-report every 30s)")
        print("    4. Train offline                   (rebuild Q-table from history.csv)")
        print("    5. Evaluate                        (print report, no simulation)")
        print("    6. Exit")
        print()

        choice = input("  Enter choice [1-6]: ").strip()

        if choice == "1":
            option_simulator_only()
        elif choice == "2":
            option_simulator_and_rl()
        elif choice == "3":
            option_simulator_rl_and_evaluate()
        elif choice == "4":
            option_train_only()
        elif choice == "5":
            option_evaluate()
        elif choice == "6":
            print("\n  Goodbye.\n")
            break
        else:
            print("\n  Invalid choice — enter a number between 1 and 6.")
            time.sleep(1)


if __name__ == "__main__":
    main()

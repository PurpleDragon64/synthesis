import os
import subprocess
from datetime import datetime
import sys

onebyone = "--onebyone" in sys.argv

# List of models
posmg_model = ["obstacles-maze-4", "obstacles-maze-4-2", "obstacles-maze-4-3", "refuel-04", "refuel-04-slip", "rover-100", "dpm-switch-q10"]

# Create experiments folder if it doesn't exist
experiments_dir = "experiments"
if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)

# Get current date timestamp
timestamp = datetime.now().strftime("%Y-%m-%d")

# Create timestamped folder inside experiments
timestamped_dir = os.path.join(experiments_dir, f"{timestamp}-pomdp-family")
if onebyone:
    timestamped_dir += "-onebyone"
if not os.path.exists(timestamped_dir):
    os.makedirs(timestamped_dir)

# Iterate over the list and run the command
for model in posmg_model:
    log_file = os.path.join(timestamped_dir, f"{model}.log")

    # Skip the model if log_file already exists and --override is not used
    if os.path.exists(log_file) and "--override" not in sys.argv:
        print(f"Skipping {model}, log file already exists.")
        continue

    command = f"gtimeout 3600 python3 paynt.py models/pomdp/sketches/{model}"
    if onebyone:
        command += " --method onebyone"

    with open(log_file, "w") as log:
        process = subprocess.Popen(command, shell=True, stdout=log, stderr=log)
        process.communicate()

    print(f"Finished analysis for {model}")
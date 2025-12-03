# activate venv!

# Script runs paynt on `models`, analyzes outtput from stderr and stores results in log.csv
# Used to analyze value vectors between iterations in AR

import subprocess
import os
import csv

models = [
    # posmg
    'dice-and-coin',
    'dice-and-coin2',
    'test-game',
    'test-game2',
    'dodge-6-lines',
    'dodge-6-lines-position'
    'attacker-one-defender',
]

# open csv file to write results
with open('log.csv', mode='w') as log:
    writer = csv.writer(log)
    writer.writerow(
        ['model', 'game',
         'states left', 'states removed',
         'diff min', 'diff max', 'diff avg', 'diff std',
         'nonzero diff count'])

    for model in models:
        model_path = os.path.join('models/posmg', model)
        command = f'python3 -m paynt {model_path}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        error = result.stderr

        game = 0
        states_left = None
        states_removed = None
        diff_min = None
        diff_max = None
        diff_avg = None
        diff_std = None
        nonzero_diff_count = None
        for line in error.split('\n'):
            if 'Number of states left:' in line:
                states_left = int(line.split(':')[-1].strip())
            elif 'Number of states removed:' in line:
                states_removed = int(line.split(':')[-1].strip())
            elif 'Diff min:' in line:
                diff_min = float(line.split(':')[-1].strip())
            elif 'Diff max:' in line:
                diff_max = float(line.split(':')[-1].strip())
            elif 'Diff avg:' in line:
                diff_avg = float(line.split(':')[-1].strip())
            elif 'Diff std:' in line:
                diff_std = float(line.split(':')[-1].strip())
            elif 'Number of nonzero diffs:' in line:
                nonzero_diff_count = int(line.split(':')[-1].strip())
                # assume one game per such block
                writer.writerow([
                    model, game,
                    states_left, states_removed,
                    diff_min, diff_max, diff_avg, diff_std,
                    nonzero_diff_count
                ])
                game += 1

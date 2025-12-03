# activate venv!

# Script runs paynt on `models`, analyzes outtput from stderr and stores results in log.csv
# Used to analyze value iteration form Tempest

import subprocess
import os
import csv

models = [
    # posmg
    'posmg/dice-and-coin',
    'posmg/dice-and-coin2',
    'posmg/grid-walls',
    'posmg/attacker-one-defender',
    # policy tree
    'archive/atva24-policy-trees/obstacles-10-6-skip-easy',
    'archive/atva24-policy-trees/avoid-8-2-easy',
    # pomdp family - not present in current version of paynt
    # interval pomdp - not present in current version of paynt
]

# open csv file to write results
with open('log.csv', mode='w') as log:
    writer = csv.writer(log)
    writer.writerow(
        ['model type', 'model', 'game', 'value vector size', 'iteration', 'diff count', 'total diff'])

    for model in models:
        model_path = os.path.join('models', model)
        command = f'python3 -m paynt {model_path}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        error = result.stderr

        model_type = '/'.join(model.split('/')[0:-1])
        model_name = model.split('/')[-1]
        game = 0
        value_vector_size = None
        iteration = None
        diff_count = None
        total_diff = None
        for line in error.split('\n'):
            if 'Started VI' in line:
                game += 1
            if 'Value vector size:' in line:
                value_vector_size = int(line.split(':')[1].strip())
            if 'Iteration' in line:
                iteration = int(line.split('Iteration')[1].strip())
                diff_count = 0
                total_diff = 0.0
            if 'Number of diffs:' in line:
                diff_count += int(line.split(':')[1].strip())
            if 'Total diff:' in line:
                total_diff += float(line.split(':')[1].strip())
                if iteration > 0:
                    writer.writerow(
                        [model_type, model_name, game, value_vector_size, iteration, diff_count, total_diff])

# Model Design
import agentpy as ap 
import numpy as np 
import argparse
from boids_model import BoidsModel
import multiprocessing as mp

# Visualisation
from visualisation import animation_plot, animation_plot_single

parser = argparse.ArgumentParser()
parser.add_argument('--threads', type=int, help="Number of threads")
parser.add_argument('--iterations', type=int, help="Number of iterations")
parser.add_argument('--new_experiment', action="store_true", help="New experiment")
parser.add_argument('--reporters_file', type=str, default="reporters.csv", help="Reporters file")
args = parser.parse_args()

# Parameter definitions
parameters2D = {
    'size': 50,
    'seed': 123,
    'steps': 200,
    'ndim': 2,
    'population': 200,
    'inner_radius': 3,
    'outer_radius': 10,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'seperation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}

parameters_multi = dict(parameters2D)
parameters_multi.update({
    'cohesion_strength': ap.Values(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.5, 1),
    'seperation_strength': ap.Values(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15),
    'alignment_strength': ap.Values(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    'border_strength': ap.Values(0.05, 0.1, 0.5, 1, 2, 3, 5, 7)
})
# parameters_multi.update({
#     'cohesion_strength': ap.Values(0.001, 0.0025, 0.005),
# })

sample = ap.Sample(parameters_multi)

exp = ap.Experiment(BoidsModel, sample, iterations=args.iterations, record=True, reporters_file=args.reporters_file)
# results = exp.run()

results = exp.run(n_jobs=args.threads, new_experiment=args.new_experiment, verbose=10)
results.reporters.to_csv('boids_statistics_results.csv')

# results.reporters.save()

# animation_plot(BoidsModel, parameters2D)
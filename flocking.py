# Model Design
import agentpy as ap 
import numpy as np 
from boids_model import BoidsModel

# Visualisation
from visualisation import animation_plot, animation_plot_single


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

model = BoidsModel(parameters2D)
results = model.run()
print(results)
print(results.variables.BoidsModel.head(-1))
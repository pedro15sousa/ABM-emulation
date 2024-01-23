# Model Design
import agentpy as ap 
import numpy as np 

from boid import Boid

class BoidsModel(ap.Model):
    """
    An agent-based model of animals' flocking behavior,
    based on Craig Reynolds' Boids Model [1]
    and Conrad Parkers' Boids Pseudocode [2].

    [1] http://www.red3d.com/cwr/boids/
    [2] http://www.vergenet.net/~conrad/boids/pseudocode.html
    """

    def setup(self):
        """ Initializes the agents and network of the model. """
        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)

    def update(self):
        """ Record agents' positions after setup and each step."""
        # self.record('positions', self.agents.pos)
        for i, agent in enumerate(self.agents):
            self.record(f'agent_{i}', list(agent.pos))

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction
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

        # Report initial alignment
        initial_alignment = np.var([agent.velocity for agent in self.agents])
        self.report('initial_aligment', initial_alignment)

        # Report initial cohesion
        center = np.mean([agent.pos for agent in self.agents], axis=0)
        initial_cohesion = np.mean([np.linalg.norm(agent.pos - center) for agent in self.agents])
        self.report('initial_cohesion', initial_cohesion)

        # Report initial seperation
        distances = [np.linalg.norm(agent.pos - other.pos) for agent in self.agents for other in self.agents if agent != other]
        initial_separation_min = np.min(distances)
        initial_separation_max = np.max(distances)
        initial_separation_avg = np.mean(distances)
        self.report('initial_separation_min', initial_separation_min)
        self.report('initial_separation_max', initial_separation_max)
        self.report('initial_separation_avg', initial_separation_avg)

        # Report initial border distance
        border_distances = [min(agent.pos[i], self.space.shape[i] - agent.pos[i]) for agent in self.agents for i in range(self.p.ndim)]
        initial_border_distance_min = np.min(border_distances)
        initial_border_distance_max = np.max(border_distances)
        initial_border_distance_avg = np.mean(border_distances)
        self.report('initial_border_distance_min', initial_border_distance_min)
        self.report('initial_border_distance_max', initial_border_distance_max)
        self.report('initial_border_distance_avg', initial_border_distance_avg)

        # Report initial parameters
        self.report('cohesion_strength', self.p.cohesion_strength)
        self.report('seperation_strength', self.p.seperation_strength)
        self.report('alignment_strength', self.p.alignment_strength)
        self.report('border_strength', self.p.border_strength)

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction

    def update(self):
        """ Record agents' positions after setup and each step."""
        # self.record('positions', self.agents.pos)
        for i, agent in enumerate(self.agents):
            self.record(f'agent_{i}', list(agent.pos))

    def end(self):
        # Final alignment
        final_alignment = np.var([agent.velocity for agent in self.agents])
        self.report('final_alignment', final_alignment)

        # Final cohesion
        center = np.mean([agent.pos for agent in self.agents], axis=0)
        final_cohesion = np.mean([np.linalg.norm(agent.pos - center) for agent in self.agents])
        self.report('final_cohesion', final_cohesion)

        # Final seperation
        distances = [np.linalg.norm(agent.pos - other.pos) for agent in self.agents for other in self.agents if agent != other]
        final_separation_min = np.min(distances)
        final_separation_max = np.max(distances)
        final_separation_avg = np.mean(distances)
        self.report('final_separation_min', final_separation_min)
        self.report('final_separation_max', final_separation_max)
        self.report('final_separation_avg', final_separation_avg)

        # Final border distance
        border_distances = [min(agent.pos[i], self.space.shape[i] - agent.pos[i]) for agent in self.agents for i in range(self.p.ndim)]
        final_border_distance_min = np.min(border_distances)
        final_border_distance_max = np.max(border_distances)
        final_border_distance_avg = np.mean(border_distances)
        self.report('final_border_distance_min', final_border_distance_min)
        self.report('final_border_distance_max', final_border_distance_max)
        self.report('final_border_distance_avg', final_border_distance_avg)


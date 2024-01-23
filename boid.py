# Model Design
import agentpy as ap 
import numpy as np 

def normalise(v):
    """Normalise a vector to length 1."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Boid(ap.Agent):
    """
    An agent with a position and velocity in a continuous space that
    follows Craig Reynolds' flocking rules (3 rules) plus a fourth rule
    to avoid the edges of the simulation space.
    """
    
    def setup(self):
        self.velocity = normalise(
            self.model.nprandom.random(self.p.ndim) - 0.5)
        
    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        pos = self.pos
        ndim = self.p.ndim

        # Rule 1 - Cohesion
        nbs = self.neighbors(self, distance=self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_array = np.array(nbs.pos)
        nbs_vec_array = np.array(nbs.velocity)
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # Rule 2 - Seperation
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self, distance=self.p.inner_radius):
            v2 -= nb.pos - pos
        v2 *= self.p.seperation_strength

        # Rule 3 - Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)

        # Rule 4 - Borders
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalise(self.velocity)

    def update_position(self):
        self.space.move_by(self, self.velocity)
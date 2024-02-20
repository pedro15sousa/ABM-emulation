import agentpy as ap 
import numpy as np

import matplotlib.pyplot as plt
import IPython
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation


def animation_plot_single(m, ax):
    ndim = m.p.ndim
    ax.set_title(f"Boids Flocking Model {ndim}D t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(pos)).T  # Transform
    ax.scatter(*pos, s=1, c='black')
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if ndim == 3:
        ax.set_zlim(0, m.p.size)
    ax.set_axis_off()

def animation_plot(m, p):
    projection = '3d' if p['ndim'] == 3 else None
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection=projection)
    animation = ap.animate(m(p), fig, ax, animation_plot_single)

    # Save the animation as a video file
    writer = FFMpegWriter(fps=20)
    animation.save("animation_cona.mp4", writer=writer)
    # return IPython.display.HTML(animation.to_jshtml(fps=20))

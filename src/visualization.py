import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d.art3d import Line3D, Path3DCollection

from src.body import Body
from src.utils import compute_relative_marker_size

colors = mpl.colormaps["Set3"].colors
plt.style.use('dark_background')


def create_frame(
        frame_num: int,
        trajectories: np.ndarray,
        trajectory_traces: list[Line3D],
        position_traces: list[Path3DCollection],
) -> list[Artist]:

    for i in range(trajectories.shape[0]):
        trajectory_traces[i].set_data(
            trajectories[i, :frame_num, 0],
            trajectories[i, :frame_num, 1],
        )
        trajectory_traces[i].set_3d_properties(trajectories[i, :frame_num, 2])

        position_traces[i].set_offsets(
            trajectories[i, frame_num, 0:2],
        )
        position_traces[i].set_3d_properties(trajectories[i, frame_num, 2], zdir="z")

    return trajectory_traces + position_traces


def plot_animation(bodies: list[Body]):
    """
    Add one to this variable if you struggled trying to edit this function -> STRUGGLES = 5
    """
    trajectories = np.stack([b.states for b in bodies])
    masses = [b.mass for b in bodies]
    marker_sizes = [compute_relative_marker_size(m, max(masses)) for m in masses]
    n_frames = trajectories.shape[1]
    xm = np.min(trajectories[:, :, 0])
    xM = np.max(trajectories[:, :, 0])
    ym = np.min(trajectories[:, :, 1])
    yM = np.max(trajectories[:, :, 1])
    zm = np.min(trajectories[:, :, 2])
    zM = np.max(trajectories[:, :, 2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set(xlim3d=(xm, xM), xlabel="X")
    ax.set(ylim3d=(ym, yM), xlabel="Y")
    ax.set(zlim3d=(zm, zM), xlabel="Z")

    trajectory_traces = []
    position_traces = []

    for i in range(trajectories.shape[0]):
        color = np.array(colors[i]).reshape(1, -1)
        trajectory_traces += ax.plot3D(
            trajectories[i, 0, 0],
            trajectories[i, 0, 1],
            trajectories[i, 0, 2],
            c=color,
        )
        position_traces.append(ax.scatter(
            trajectories[i, 0, 0],
            trajectories[i, 0, 1],
            trajectories[i, 0, 2],
            c=color,
            marker='o',
            s=marker_sizes[i]
        ))

    anim = FuncAnimation(
        fig=fig,
        func=create_frame,
        fargs=(trajectories, trajectory_traces, position_traces),
        frames=n_frames,
        interval=1,
        repeat=False
    )
    # anim.save("animation.gif", fps=10)
    plt.show()


def plot_orbits(bodies):
    trajectories = np.stack([b.states for b in bodies])
    plt.style.use('dark_background')
    colors = mpl.colormaps["Set3"].colors

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(trajectories.shape[0]):
        color = np.array(colors[i]).reshape(1, -1)
        ax.plot3D(
            trajectories[i, :, 0],
            trajectories[i, :, 1],
            trajectories[i, :, 2],
            c=color,
        )

    plt.show()

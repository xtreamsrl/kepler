import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.body import Body
from src.utils import compute_relative_marker_size

colors = matplotlib.colormaps["Set3"].colors
plt.style.use('dark_background')


def create_frame(
        frame_num: int,
        trajectories: np.ndarray,
        trajectory_traces: list[plt.Artist],
        position_traces: list[plt.Artist],
) -> list[plt.Artist]:

    for i in range(trajectories.shape[0]):
        trajectory_traces[i].set_data(
            trajectories[i, 0, :frame_num],
            trajectories[i, 1, :frame_num],
        )
        trajectory_traces[i].set_3d_properties(trajectories[i, 2, :frame_num])

        position_traces[i].set_offsets(
            trajectories[i, 0:2, frame_num],
        )
        position_traces[i].set_3d_properties(trajectories[i, 2, frame_num], zdir="z")

    return trajectory_traces + position_traces


def plot_animation(bodies: list[Body]):
    trajectories = np.stack([b.states for b in bodies])
    masses = [b.mass for b in bodies]
    marker_sizes = [compute_relative_marker_size(m, max(masses)) for m in masses]
    n_frames = trajectories.shape[-1]
    xm = np.min(trajectories[:, 0, :])
    xM = np.max(trajectories[:, 0, :])
    ym = np.min(trajectories[:, 1, :])
    yM = np.max(trajectories[:, 1, :])
    zm = np.min(trajectories[:, 2, :])
    zM = np.max(trajectories[:, 2, :])

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
            trajectories[i, 1, 0],
            trajectories[i, 2, 0],
            c=color,
        )
        position_traces.append(ax.scatter(
            trajectories[i, 0, 0],
            trajectories[i, 1, 0],
            trajectories[i, 2, 0],
            c=color,
            marker='o',
            s=marker_sizes[i]
        ))

    fig = plt.figure()
    anim = FuncAnimation(
        fig=fig,
        func=create_frame,
        fargs=(trajectories, trajectory_traces, position_traces),
        frames=n_frames,
        interval=100,
        repeat=False
    )
    return anim
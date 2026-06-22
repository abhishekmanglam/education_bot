import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np


def draw_triangle():
    fig, ax = plt.subplots()

    x = [0, 4, 2, 0]
    y = [0, 0, 3, 0]

    ax.plot(x, y)
    ax.set_aspect("equal")

    return fig


def draw_circle():
    fig, ax = plt.subplots()

    circle = plt.Circle(
        (0, 0),
        1,
        fill=False
    )

    ax.add_patch(circle)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

    return fig


def draw_coordinate_plane():
    fig, ax = plt.subplots()

    ax.axhline(0)
    ax.axvline(0)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    return fig


def draw_bar_graph():
    fig, ax = plt.subplots()

    categories = ["A", "B", "C", "D"]
    values = [10, 15, 7, 12]

    ax.bar(categories, values)

    return fig


def draw_line_graph():
    fig, ax = plt.subplots()

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 3, 6, 8]

    ax.plot(x, y)

    return fig


def draw_angle():
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 0])
    ax.plot([0, 0.8], [0, 0.8])

    ax.set_aspect("equal")

    return fig
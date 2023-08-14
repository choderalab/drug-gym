import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

def plot(deck, utility_function, plot_cycle=True):

    design_cycle = deck.annotations['design'] if plot_cycle else None

    oracles = utility_function.oracles

    # create basic plot
    g = sns.jointplot(
        x=oracles[0](deck),
        y=oracles[1](deck),
        hue=design_cycle
    )

    # add evaluator boundaries
    evals = utility_function.evaluators
    ideal = g.ax_joint.add_patch(
        make_box(
            ranges=[evals[0].ideal, evals[1].ideal],
            color='green', label='Ideal'
        )
    )
    acceptable = g.ax_joint.add_patch(
        make_box(
            ranges=[evals[0].acceptable, evals[1].acceptable],
            color='#c0c0c0', label='Acceptable'
        )
    )

    # move to back
    plt.setp(g.ax_joint.patches, zorder=-1)

    # create legend
    g.ax_joint.legend(
        bbox_to_anchor=(1.2, 1),
        loc='upper left'
    )

    # labels
    plt.xlabel(oracles[0].name)
    plt.ylabel(oracles[1].name)

    # formatting
    g.fig.set_figwidth(6)
    g.fig.set_figheight(3)
    g.fig.set_dpi(300)

    return g


def make_box(ranges, color, label):
    
    abs_diff = lambda a, b: abs(a - b)
    width = abs_diff(*ranges[0])
    height = abs_diff(*ranges[1])

    x = np.mean(ranges[0]) - width/2
    y = np.mean(ranges[1]) - height/2
    
    edge_color = to_rgba(color, alpha=1)
    face_color = to_rgba(color, alpha=0.1)

    return patches.Rectangle(
        xy=(x, y),
        width=width,
        height=height,
        facecolor=face_color,
        edgecolor=edge_color,
        linestyle='--',
        label=label
    )

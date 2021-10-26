import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class Utils:
    def __init__(self):
        pass

    def plot_many_lines(ys, colors, labels, legend_font_size=18):
        plt.figure(figsize=(20,7))
        for y, color, label in zip(ys, colors, labels):
            plt.plot(y, color=color, label=label)
        plt.legend(fontsize=legend_font_size)

    def plot_two_diffs_ACF(y1, y2, y3):
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(3, 1, 1)
        fig = plot_acf(y1, ax=ax1, title="original")
        ax2 = fig.add_subplot(3, 1, 2)
        fig = plot_acf(y2, ax=ax2, title="first diff")
        ax3 = fig.add_subplot(3, 1, 3)
        fig = plot_acf(y3, ax=ax3, title="second diff")
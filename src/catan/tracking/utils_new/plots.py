# import numpy as np
# from matplotlib import colors, pyplot as plt


# def plot_with_confidence(ax, x_data, y_data, CI, col="k", ls="-", lw=1, label=None):

#     col_fill = np.minimum(np.array(colors.to_rgb(col)) + np.ones(3) * 0.3, 1)
#     if len(CI.shape) > 1:
#         ax.fill_between(x_data, CI[0, :], CI[1, :], color=col_fill, alpha=0.2)
#     else:
#         ax.fill_between(x_data, y_data - CI, y_data + CI, color=col_fill, alpha=0.2)
#     ax.plot(x_data, y_data, color=col, linestyle=ls, linewidth=lw, label=label)

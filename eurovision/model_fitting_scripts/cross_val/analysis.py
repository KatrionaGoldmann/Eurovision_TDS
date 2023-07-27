import arviz as az
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import spearmanr

num_samples = 1000
num_chains = 4

df = pd.read_csv('eurovision/df_main.csv')
def format_votes_to_scores(x):
  if x == 12.:
    return 10
  elif x == 10.:
    return 9
  return int(x)
df['indexed_votes'] = df['points'].apply(format_votes_to_scores) + 1

ro = []
ro_non_z = []
for test_year in [x for x in range(1998,2022)]:
    if test_year == 2020 :
        # no competition in 2020
        continue

    # load output from fitted model
    az_fit = az.from_json(f"eurovision/model_output/1998_to_2022_cross_val/model_3_voter_bias_vector_{num_samples}_samples_{num_chains}_chains_test_on_{test_year}.json")

    df_test = df.loc[ df['year'] == test_year ]
    
    y_test = np.asarray(df_test['indexed_votes'].values)

    y_pred = az_fit.predictions.y_pred.to_numpy()

    # before doing rank correlation, the variables should be re-mapped back to 0->12 (skipping 9 and 11)
    def format_scores_to_votes(x):
      if x == 10.:
        return 12
      elif x == 9.:
        return 10
      return int(x)
    v_test = [ format_scores_to_votes(x) for x in (y_test - 1) ]
    v_pred = np.zeros_like(y_pred)
    for c in range(y_pred.shape[0]):
      for d in range(y_pred.shape[1]):
        v_pred[c,d,:] = [ format_scores_to_votes(x) for x in (y_pred[c,d,:] - 1) ]

    # analyse the non-zero modelling performance
    y_test_non_z = y_test[y_test != 1]
    y_pred_non_z = np.zeros((y_pred.shape[0],y_pred.shape[1],y_test_non_z.shape[0]))

    for ci in range(y_pred.shape[0]):   # loop chains
        for si in range(y_pred.shape[1]):   # loop samples
            y_pred_non_z[ci][si] = y_pred[ci][si][y_test != 1]

    v_test_non_z = [ format_scores_to_votes(x) for x in (y_test_non_z - 1) ]
    v_pred_non_z = np.zeros_like(y_pred_non_z)
    for c in range(y_pred_non_z.shape[0]):
      for d in range(y_pred_non_z.shape[1]):
        v_pred_non_z[c,d,:] = [ format_scores_to_votes(x) for x in (y_pred_non_z[c,d,:] - 1) ]

    ro_of_fold = np.zeros((v_pred.shape[0],v_pred.shape[1]))
    for ci in range(v_pred.shape[0]):   # loop chains
        for si in range(v_pred.shape[1]):   # loop samples
            ro_of_fold[ci][si] = spearmanr(v_test, v_pred[ci][si])[0]
            
    ro.extend([(x,test_year) for x in ro_of_fold.flatten()])


    # subset of data where ground truth vote != 0
    ro_non_z_of_fold = np.zeros((v_pred_non_z.shape[0],v_pred_non_z.shape[1]))
    for ci in range(v_pred_non_z.shape[0]):   # loop chains
        for si in range(v_pred_non_z.shape[1]):   # loop samples
            ro_non_z_of_fold[ci][si] = spearmanr(v_test_non_z, v_pred_non_z[ci][si])[0]

    ro_non_z.extend([(x,test_year) for x in ro_non_z_of_fold.flatten()])

# plot ridgeplot
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
x_ax_label = "spearman's rank correlation coefficient"

df_ro = pd.DataFrame({
   x_ax_label: list(map(lambda x: x[0], ro)),
   'ro_non_z': list(map(lambda x: x[0], ro_non_z)),
   'g': list(map(lambda x: x[1], ro))})

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(23, rot=-.25, light=.7)
g = sns.FacetGrid(df_ro, row="g", hue="g", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, x_ax_label,
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, x_ax_label, clip_on=False, color="w", lw=2, bw_adjust=.5)

g.map(sns.kdeplot, "ro_non_z",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=0.2, linewidth=0.8)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, x_ax_label)

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

g._figure.suptitle("Cross validation for each year hold-out set")
g.savefig('eurovision/model_fitting_scripts/cross_val/spearmans_plot')

    
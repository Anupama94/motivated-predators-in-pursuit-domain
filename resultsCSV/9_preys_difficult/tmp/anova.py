import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import scikit_posthocs as sp

mainLi = []

for name in os.listdir("."):
    li = []
    if name != "summary.py" and name != "anova.py" and name != 'stats.html' and name != 'yieldperunittension.csv' and name != 'tension.csv' and\
            name != 'yield.csv' and name != "posthoc.html":
        # '111' - total steps
        # '333' - prey yield per step
        # '784' - yield per unit tension
        # '805' - tension per step
        field = '111'
        df = pd.read_csv(name, usecols=[field])
        formattedNameWithoutCSV = name.split(".")[0]
        formattedNameWithEncoding = formattedNameWithoutCSV.split("_")[-1]
        df.rename({field: formattedNameWithEncoding}, axis=1, inplace=True)
        mainLi.append(df)

csvFileName = "tension.csv"
with open('./' + csvFileName, 'w') as f:
    pd.concat(mainLi, axis=1).to_csv(f, index=False)

names = os.listdir("../..")


names = []
for name in os.listdir("../tmp"):
    if name != "summary.py" and name != "anova.py" and name != "yieldperunittension.csv" and name != "stats.html" and\
            name != "yield.csv" and name != 'tension.csv':
        formattedNameWithoutCSV = name.split(".")[0]
        formattedNameWithEncoding = formattedNameWithoutCSV.split("_")[-1]
        names.append(formattedNameWithEncoding)
df = pd.read_csv('tension.csv')


data = [df[colName] for colName in df.columns]
H, p = ss.kruskal(*data)


df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=names)
# replace column names
df_melt.columns = ['index', 'treatments', 'value']

pc = sp.posthoc_conover(df_melt, val_col='value', group_col='treatments', p_adjust='holm')

htmlFileName = "posthoc.html"
with open('./' + htmlFileName, 'w') as f:
    pc.to_html(f)

heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3], \
                'xticklabels': True, 'yticklabels': True}
sp.sign_plot(pc, **heatmap_args)
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.2)
plt.show()

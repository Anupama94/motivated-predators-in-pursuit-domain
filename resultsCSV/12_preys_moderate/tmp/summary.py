import math

import pandas as pd
import glob
import os
import plotly.figure_factory as ff
import plotly
import numpy as np

aff = [] # aff
pow = [] # pow
ach = [] # ach
numOfSteps = []
preyYield = []
tension = []

newYield = []
incentiveTensionPerStep = []


li = []
for name in os.listdir("."):
    if name != "summary.py" and name != "posthoc.html" and name != "anova.py" and name != "stats.html" and name != "tension.csv":
        li = []
        print(name)
        removeExt = name.split(".")[0]
        ratioStr = removeExt.split("_")[-1]
        eachRatio = ratioStr.split("-")
        aff.append(int(eachRatio[0])/12)
        pow.append(int(eachRatio[1])/12)
        ach.append(int(eachRatio[2])/12)

        frame = pd.read_csv(name, index_col=None, header=0)

        pd.set_option('display.max_colwidth', None)
        # print("MEAN of ", name, frame['Prey_Yield'].mean())
        numOfSteps.append(frame['111'].mean())
        preyYield.append(frame['333'].mean())
        newYield.append(frame['784'].mean())
        incentiveTensionPerStep.append(frame['805'].mean())

fig = ff.create_ternary_contour(np.array([np.array(aff), np.array(ach), np.array(pow)]), np.array(numOfSteps),
                                pole_labels=['Aff', 'Ach', 'Pow'],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,
                                title='Total Steps')
fig.show()

fig2 = ff.create_ternary_contour(np.array([np.array(aff), np.array(ach), np.array(pow)]), np.array(preyYield),
                                pole_labels=['Aff', 'Ach', 'Pow'],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,
                                title='Prey Yield per Step')
fig2.show()


normalizedNewYield = [(math.log(x)) for x in newYield]
fig3 = ff.create_ternary_contour(np.array([np.array(aff), np.array(ach), np.array(pow)]), np.array(normalizedNewYield),
                                pole_labels=['Aff', 'Ach', 'Pow'],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,
                                title='Yield per unit Tension')
fig3.show()


fig4 = ff.create_ternary_contour(np.array([np.array(aff), np.array(ach), np.array(pow)]), np.array(incentiveTensionPerStep),
                                pole_labels=['Aff', 'Ach', 'Pow'],
                                interp_mode='cartesian',
                                ncontours=20,
                                colorscale='Viridis',
                                showscale=True,
                                title='Perceived Tension per Step')
fig4.show()

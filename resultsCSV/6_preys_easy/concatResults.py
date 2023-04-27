import pandas as pd
import glob
import os
mainLi = []

for name in os.listdir("./"):
    li = []
    if name != "tmp" and name != "concatResults.py" and name != "concatSlurm.slurm":
        all_files = glob.glob(name + "/*.csv")
        print(name)
        for filename in all_files:
            df = pd.read_csv(filename)
            li.append(df)

        csvFileName = name + ".csv"
        with open('./tmp/' + csvFileName, 'w') as f:
            pd.concat(li, axis=0).to_csv(f, index=False)
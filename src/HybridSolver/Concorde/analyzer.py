import pandas as pd
import numpy as np
import ast
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp


def analyze():
    # open file "results.csv"
    df = pd.read_csv("results.csv")

    for i in range(len(df)):
        if df.iloc[i]["Experiment"] == "Classic":
            classic_bbnodes = df.iloc[i]["BBNodes"]
            classic_bbnodes = np.array(ast.literal_eval(classic_bbnodes), dtype=int)
            classic_instance = df.iloc[i]["Instance"]
            for j in range(len(df)):
                experiment = df.iloc[j]["Experiment"]
                if experiment != "Classic" and df.iloc[j]["Instance"] == classic_instance:
                    hybrid_bbnodes = df.iloc[j]["BBNodes"]
                    hybrid_bbnodes = np.array(ast.literal_eval(hybrid_bbnodes), dtype=int)
                    print(classic_bbnodes - hybrid_bbnodes)
                    if not np.array_equal(classic_bbnodes, hybrid_bbnodes):
                        wilc = wilcoxon(classic_bbnodes, hybrid_bbnodes).pvalue < 0.05
                        ttest = ttest_rel(classic_bbnodes, hybrid_bbnodes).pvalue < 0.05
                        
                        if wilc or ttest:
                            print(f"Instance: {classic_instance} \t Experiment: {df.iloc[j]['Experiment']} \t Wilcoxon p-value: {wilc} \t T-test p-value: {ttest}")


if __name__ == "__main__":
    analyze()
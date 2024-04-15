from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np
import pandas as pd
import subprocess
import os


def DEPs_ANOVA(df_data, labels):
    le = LabelEncoder()
    label_encoded = le.fit_transform(labels)

    ANOVA_res = []

    for i in range(df_data.shape[0]):
        a = df_data.iloc[i,:].to_numpy()
        data = np.c_[a,label_encoded]

        df = pd.DataFrame(data, columns=['value', 'treatment'])    

        # the "C" indicates categorical data
        res = anova_lm(ols('value ~ C(treatment)', df).fit())
        ANOVA_res.append((df_data.index[i],res.iloc[0,-2],res.iloc[0,-1]))

    ANOVA_arr = np.array(ANOVA_res)
    pval_adj = multipletests(ANOVA_arr[:,-1].astype(float), method="fdr_bh")[1]

    ANOVA_df = pd.DataFrame(np.c_[ANOVA_arr, pval_adj], columns=['protein','F','pval','pval_adj']).set_index('protein').astype('float64').reset_index().sort_values(by='pval_adj',ascending=True)
    ANOVA_df['rank'] = range(1,ANOVA_df.shape[0]+1)

    DEPs = ANOVA_df.loc[lambda x:x.pval_adj<0.1,'protein'].to_list()

    return DEPs

def run_NP(df_data, label_dict, outDir):
    import pdb;pdb.set_trace()
    labels = df_data.index.map(lambda x: label_dict[x])
    DEPs = DEPs_ANOVA(df_data, labels)

    if os.path.exists(outDir + '/tmp') == False:
        os.makedirs(outDir + '/tmp')

    with open(outDir + '/tmp/ANOVA_DEPs.txt', 'w') as f:
        f.write("\n".join(DEPs))

    NP_script = "network_propagation/network_propagation.py"
    network = "network_propagation/human_string_ppi_norm.nwk.over0.5"
    seed = "tmp/ANOVA_DEPs.txt"
    cmd = ["python", NP_script, network, seed, '-o', 'tmp/NP_with_DEPs.txt', '-e', '0.5', '-addBidirectionEdge', 'True', '-normalize', 'True', '-constantWeight', 'True']
    # !python 3_network_analysis.py STRING/human_string_ppi_norm.nwk.over0.5 latest/ANOVA_DEPs.txt -o latest/NP_with_DEPs.txt -e 0.5 -addBidirectionEdge True -normalize True -constantWeight True

    out = pd.read_csv("latest/NP_with_DEPs.txt", sep='\t', header=None, index_col=0, names=['NP_score'])#[0]

    return out
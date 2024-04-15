import argparse
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import os
from MI_gene_ranking import *
from NP_gene_ranking import *

def main(args):
    input_data = pd.read_csv(args.data, sep='\t', header=0, index_col=0)
    label_dict = dict(pd.read_csv(args.label, sep='\t',header=0).values)
    
    # Data split
    train, test = input_data.iloc[:-1,:], input_data.iloc[-1,:]

    # ===== Feature ranking ===== 
    # 1. Ranking by Mutual Information
    score_MI = pd.DataFrame(avg_mut_info(train,label_dict,100)).rename(columns={0:'MI_score'}).sort_values(by=['MI_score'],ascending=False)
    import pdb;pdb.set_trace()
    # 2. Ranking by Network Propagation
    score_NP = run_NP(train, label_dict, args.outDir)

    # 3. Combine two scores
    score = score_MI.join(score_NP, how='inner').sort_values(by=['MI_score','NP_score'],ascending=False)

    # ===== Recursive feature addition ===== 
    accuracy = []
    for k in range(1, input_data.shape[1]):
        X = train.iloc[:k,:].to_numpy()
        y = train.index.map(lambda x: label_dict[x])

        model = SVC(C=0.25, gamma='auto', kernel='rbf',decision_function_shape='ovo')
        accuracy.append(cross_val_score(model, X, y, cv=3).mean())

    k = np.argwhere(np.array(accuracy) == np.max(np.array(accuracy)))[0]

    # Test with in-distribution data

    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, help="Input: sample X feature matrix in tsv format")
    parser.add_argument("-label", type=str, help="Labels of sample")
    parser.add_argument("-outDir",type=str)
    args = parser.parse_args()
    if os.path.exists(args.outDir) == False:
        os.makedirs(args.outDir)
    main(args)
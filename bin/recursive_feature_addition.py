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
    data_split = pd.read_csv(args.data_split, sep='\t', header=0, index_col=0)
    train_idx = list(set(input_data.index).intersection(set(data_split.loc[lambda x:x.train==True,:].index)))
    test_idx = list(set(input_data.index).intersection(set(data_split.loc[lambda x:x.test==True,:].index)))
    train, test = pd.DataFrame(input_data.loc[train_idx,:]), pd.DataFrame(input_data.loc[test_idx,:])

    # ===== Feature ranking ===== 
    # 1. Ranking by Mutual Information
    score_MI = pd.DataFrame(avg_mut_info(train,label_dict,100)).rename(columns={0:'MI_score'}).sort_values(by=['MI_score'],ascending=False)
    
    # 2. Ranking by Network Propagation
    score_NP = run_NP(train, label_dict, args.outDir)

    # 3. Combine two scores
    score = score_MI.join(score_NP, how='inner').sort_values(by=['MI_score','NP_score'],ascending=False)

    # ===== Recursive feature addition ===== 
    accuracy = []
    for k in range(1, input_data.shape[1]):
        genes = score.index[:k]
        X = train.loc[:,genes].to_numpy()
        y = train.index.map(lambda x: label_dict[x])

        model = SVC(C=0.25, gamma='auto', kernel='rbf',decision_function_shape='ovo')
        accuracy.append(cross_val_score(model, X, y, cv=3).mean())

    k_optimal = np.argwhere(np.array(accuracy) == np.max(np.array(accuracy)))[0,0]
    
    genes_final = score.index[:(k_optimal+1)]
    X_final = train.loc[:,genes_final].to_numpy()
    y_final = train.loc[:,genes_final].index.map(lambda x: label_dict[x])
    model_final = SVC(C=0.25, gamma='auto', kernel='rbf',decision_function_shape='ovo').fit(X_final, y_final)

    # Test with in-distribution data
    X_test = test.loc[:,genes_final].to_numpy()
    y_test = test.index.map(lambda x: label_dict[x])
    y_pred = model_final.predict(X_test)

    with open(args.outDir+'/recursive_feature_addition.txt', 'w') as f:
        f.write("Optimal number of features: "+str(k_optimal)+'\n')
        f.write("Selected features: "+", ".join(genes_final)+'\n')
        f.write("Accuracy: "+str(np.max(np.array(accuracy)))+'\n')
        f.write("Test result (y_predicted): "+'\t'.join(map(str,y_pred))+'\n')
        f.write("Test result (y_true): "+'\t'.join(map(str,y_test)))

    return 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, help="Input: sample X feature matrix in tsv format")
    parser.add_argument("-label", type=str, help="Labels of sample")
    parser.add_argument("-data_split", type=str, help="Data split file in tsv format")
    parser.add_argument("-outDir",type=str)
    args = parser.parse_args()
    if os.path.exists(args.outDir) == False:
        os.makedirs(args.outDir)
    main(args)

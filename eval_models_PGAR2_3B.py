#Evaluate learned model profits against test and hold out data
import numpy as np
import pandas as pd
import sys
#!pip install tpot
#from tpot import TPOTClassifier

from utils_3B import eval_test_ho, gen_ho

def eval_models(tour_ind, r_ind, g_ind, df_flat, X_train, X_train_norm, Y_train, X_test, X_test_norm, Y_test, X_features, Y_classes, y_dict, scaler,
                x_cols, y_cols, tourn_cols, pl_id, pl_col, p012_cols, reduced_cols, p_res):   
    df_results = None
    file_name = tour_ind+'_R'+str(r_ind)+'_'+str(g_ind)+'ball_model_results.csv'
    df_results = pd.read_csv(file_name)
    col_lst = list(df_results.columns)
    print(f'\nInitial load df_results = {df_results.shape}\n')
    #sys.exit()
    '''
    ['model', 'sc_metric', 'ties', 'm_test', 'm_ho', 'pred_ror_test', 'pred_ror_ho', 'pred_ror_comb', 'fav_ror_test', 'fav_ror_ho',
    'pred_prec_test', 'pred_prec_ho', 'fav_prec_test', 'fav_prec_ho', 'acc_test', 'acc_ho', 'p0_freq_test', 'p0_freq_ho']
    '''
    #Generate HO pred and eval results
    df_flat_ho, X_test_ho, Y_test_ho, Y_test_oh_ho = gen_ho(df_flat, X_train, g_ind, tour_ind, r_ind, x_cols, y_cols, X_features, Y_classes, tourn_cols, pl_id, pl_col, p012_cols)
    #Normalise X_test_ho
    X_test_norm_ho = scaler.transform(X_test_ho)
    print(f'\nX_test_norm_ho {X_test_norm_ho.shape}, {X_test_norm_ho[-1,:5]}')
    #If len(reduced_cols)>0 generate reduced X arrays for train, test and hold sets
    if len(reduced_cols)>0:
        reduced_inds = [i for i in range(len(X_features)) if X_features[i] in reduced_cols]
        print(f'\nreduced_cols={len(reduced_cols)}: {reduced_inds}')
        print(f'Check... reduced_cols[0]={X_features[reduced_inds[0]]}, reduced_cols[-1]={X_features[reduced_inds[-1]]}')
        print(f'Inititial: X_train_norm={type(X_train_norm)},{X_train_norm.shape}')
        print(f'Inititial: X_test_norm={type(X_test_norm)},{X_test_norm.shape}')
        print(f'Inititial: X_test_norm_ho={type(X_test_norm_ho)},{X_test_norm_ho.shape}')
        X_train_norm=X_train_norm[:,reduced_inds]
        X_test_norm=X_test_norm[:,reduced_inds]
        X_test_norm_ho=X_test_norm_ho[:,reduced_inds]
        print(f'Post reduction: X_train_norm={type(X_train_norm)},{X_train_norm.shape}')
        print(f'Post reduction: X_test_norm={type(X_test_norm)},{X_test_norm.shape}')
        print(f'Post reduction: X_test_norm_ho={type(X_test_norm_ho)},{X_test_norm_ho.shape}')

    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.tree import DecisionTreeClassifier
    #from tpot.builtins import StackingEstimator
    #from tpot.export_utils import set_param_recursive
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import FunctionTransformer
    from copy import copy
    from xgboost import XGBClassifier
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    #from tpot.builtins import ZeroCount
    from sklearn.decomposition import FastICA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFwe
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import Binarizer
    from sklearn.cluster import FeatureAgglomeration
        
    ##Evaluate Models##
    ##Model DD)
    ev = 1
    if ev==1:
        from sklearn.neural_network import MLPClassifier
        v = '0'
        tpot_flag=0
        base = 'MLPClassifier'
        for a in ['identity', 'logistic', 'tanh', 'relu']: #['identity', 'logistic', 'tanh', 'relu']
            for s in ['lbfgs', 'sgd', 'adam']:   #['lbfgs', 'sgd', 'adam']
                for lr in ['constant', 'invscaling', 'adaptive']:
                    sc_metric = a+'_'+s+'_'+lr
                    print(f'Running {base} with hyperparameters {sc_metric}...')
                    msetup = MLPClassifier(random_state=0, activation=a, solver=s, learning_rate=lr)
                    model = msetup.fit(X_train_norm, Y_train.astype('int'))
                    df_results = eval_test_ho(df_flat, X_train, X_train_norm, Y_train, X_test, X_test_norm, Y_test, y_dict, df_results, base, sc_metric, v, tpot_flag, model,
                                              p_res, g_ind, df_flat_ho, X_test_ho, Y_test_ho, X_test_norm_ho, reduced_cols, r_ind)
                    print(f'\ndf_results = {df_results.shape}\n')
                    
    ##Model CC)
    ev = 1
    if ev==1:
        from sklearn.linear_model import LogisticRegression
        v = '0'
        tpot_flag=0
        base = 'LogisticRegression'
        for k in ['newton-cg', 'lbfgs', 'sag', 'saga']:   #['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
                sc_metric = k+'_c'+str(c)
                print(f'Running {base} with hyperparameters {sc_metric}...')
                msetup = LogisticRegression(random_state=0, class_weight='balanced', solver=k, C=c, multi_class='multinomial')
                model = msetup.fit(X_train_norm, Y_train.astype('int'))
                df_results = eval_test_ho(df_flat, X_train, X_train_norm, Y_train, X_test, X_test_norm, Y_test, y_dict, df_results, base, sc_metric, v, tpot_flag, model,
                                          p_res, g_ind, df_flat_ho, X_test_ho, Y_test_ho, X_test_norm_ho, reduced_cols, r_ind)
                print(f'\ndf_results = {df_results.shape}\n')
    
    ##Model BB)
    ev = 1
    if ev==1:
        from sklearn.svm import SVC
        v = '0'
        tpot_flag=0
        base = 'SVM SVC'
        for k in ['linear', 'poly', 'rbf', 'sigmoid']:   #['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
            for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
                for g in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
                    sc_metric = k+'_c'+str(c)+'_g'+str(g)
                    print(f'Running {base} with hyperparameters {sc_metric}...')
                    svmm = SVC(random_state=0, class_weight='balanced', kernel=k, C=c, gamma=g)
                    model = svmm.fit(X_train_norm, Y_train.astype('int'))
                    df_results = eval_test_ho(df_flat, X_train, X_train_norm, Y_train, X_test, X_test_norm, Y_test, y_dict, df_results, base, sc_metric, v, tpot_flag, model,
                                              p_res, g_ind, df_flat_ho, X_test_ho, Y_test_ho, X_test_norm_ho, reduced_cols, r_ind)
                    print(f'\ndf_results = {df_results.shape}\n')
                
    ##Model AA)
    ev = 1
    if ev==1:
        from sklearn.svm import LinearSVC
        v = '0'
        tpot_flag=0
        base = 'SVM LinearSVC'
        for cl in ['ovr', 'crammer_singer']:
            for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
            #for c in [0.0001, 0.0005]:
                sc_metric = cl+'_'+str(c)
                print(f'Running {base} with hyperparameters {sc_metric}...')
                svmm = LinearSVC(random_state=0, tol=1e-2,class_weight='balanced', multi_class=cl, C=c)
                model = svmm.fit(X_train_norm, Y_train.astype('int'))
                df_results = eval_test_ho(df_flat, X_train, X_train_norm, Y_train, X_test, X_test_norm, Y_test, y_dict, df_results, base, sc_metric, v, tpot_flag, model,
                                          p_res, g_ind, df_flat_ho, X_test_ho, Y_test_ho, X_test_norm_ho, reduced_cols, r_ind)
                print(f'\ndf_results = {df_results.shape}\n')


    #Print Results and output file
    print(f'\nTop Results: ')
    df_results['pred_ror_comb'] = pd.to_numeric(df_results['pred_ror_comb'])
    res = df_results.loc[df_results['pred_ror_comb'].idxmax(), ['model', 'pred_ror_comb']].values
    print(f'Model with best average profit (overall): {res}')

    df_res_high_m_test = df_results.loc[df_results['m_test']>=0.7*len(X_test)].copy()
    res = df_res_high_m_test.loc[df_res_high_m_test['pred_ror_comb'].idxmax(), ['model', 'pred_ror_comb']].values
    print(f'\nModel with best average profit (where m_test > 0.7*X_test): {res}')

    df_res_best_prec = df_results.loc[(df_results['pred_prec_test']>=df_results['fav_prec_test']) & (df_results['pred_prec_ho']>=df_results['fav_prec_ho'])].copy()
    res = df_res_best_prec.loc[df_res_best_prec['pred_ror_comb'].idxmax(), ['model', 'pred_ror_comb']].values
    print(f'\nModel with best average profit (where pred_prec_test >= fav_prec_test AND pred_prec_ho >= fav_prec_ho): {res}')

    df_res_best_prec_70p = df_results.loc[(df_results['pred_prec_test']>=df_results['fav_prec_test']) & (df_results['pred_prec_ho']>=df_results['fav_prec_ho']) & (df_results['m_test']>=0.7*len(X_test))].copy()
    res = df_res_best_prec_70p.loc[df_res_best_prec_70p['pred_ror_comb'].idxmax(), ['model', 'pred_ror_comb']].values
    print(f'\nModel with best average profit (where pred_prec_test >= fav_prec_test AND pred_prec_ho >= fav_prec_ho AND m_test is >=0.7*len(X_test)): {res}')
    
    df_results.sort_values(by=['pred_ror_comb', 'pred_prec_test'], ascending=[False, False], inplace=True)
    try: df_results.drop('Unnamed: 0', inplace=True)
    except KeyError: None 
    df_results.to_csv(file_name, index=False)
    return df_results

''' run local
tour_ind = 'PGA'
r_ind = 2
g_ind = 3
df_results = eval_models(None, tour_ind, r_ind, g_ind)
'''

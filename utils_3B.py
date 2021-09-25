#Utility functions for model evals
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import accuracy_score as accuracy

def eval_test_ho(df_flat, X_train, X_train_norm, Y_train, X_test, X_test_norm, Y_test, y_dict, df_results, base, sc_metric, v, tpot_flag, model,
                 p_res, g_ind, df_flat_ho, X_test_ho, Y_test_ho, X_test_norm_ho, reduced_cols, r_ind):
    #1) Generate test & ho results including ties
    if len(reduced_cols)==0: mod_name = base+'_'+sc_metric+'_v'+v
    else: mod_name = base+'_reduced'+str(len(reduced_cols))+'_'+sc_metric+'_v'+v
    if not(mod_name in list(df_results['model'])):
        df_results = df_results.append(pd.Series(), ignore_index=True)
        row_ind = len(df_results)-1
        df_results.loc[row_ind,'model']=mod_name
        df_results.loc[row_ind,'sc_metric']=sc_metric
        df_results.loc[row_ind,'ties']=1
        df_results.loc[row_ind,'m_test']=len(X_test)
        df_results.loc[row_ind,'m_ho']=len(X_test_ho)
    else:
        row_ind = df_results.index[df_results['model']==mod_name].values[0]
        df_results.loc[row_ind,'sc_metric']=sc_metric
        df_results.loc[row_ind,'ties']=1
        df_results.loc[row_ind,'m_test']=len(X_test)
        df_results.loc[row_ind,'m_ho']=len(X_test_ho)
        
    if tpot_flag==1:
        pred = model.predict(X_test.astype('float'))
        pred_ho = model.predict(X_test_ho.astype('float'))
    else:
        pred = model.predict(X_test_norm)
        pred_ho = model.predict(X_test_norm_ho)
    #Test Results
    pred_ror_test, fav_ror_test, pred_prec_test, fav_prec_test, acc_test, p0_freq_test = eval_predictions(Y_test.astype('int'), pred, y_dict, p_res, df_flat, X_train, g_ind, r_ind)
    df_results.loc[row_ind,'pred_ror_test']=pred_ror_test
    df_results.loc[row_ind,'fav_ror_test']=fav_ror_test
    df_results.loc[row_ind,'pred_prec_test']=pred_prec_test
    df_results.loc[row_ind,'fav_prec_test']=fav_prec_test
    df_results.loc[row_ind,'acc_test']=acc_test
    df_results.loc[row_ind,'p0_freq_test']=p0_freq_test
    #HO Results
    pred_ror_ho, fav_ror_ho, pred_prec_ho, fav_prec_ho, acc_ho, p0_freq_ho = eval_predictions(Y_test_ho.astype('int'), pred_ho, y_dict, p_res, df_flat_ho, X_train, g_ind, r_ind)
    df_results.loc[row_ind,'pred_ror_ho']=pred_ror_ho
    df_results.loc[row_ind,'fav_ror_ho']=fav_ror_ho
    df_results.loc[row_ind,'pred_prec_ho']=pred_prec_ho
    df_results.loc[row_ind,'fav_prec_ho']=fav_prec_ho
    df_results.loc[row_ind,'acc_ho']=acc_ho
    df_results.loc[row_ind,'p0_freq_ho']=p0_freq_ho
    #Average of test & ho ror
    df_results.loc[row_ind,'pred_ror_comb']=(pred_ror_test+pred_ror_ho)/2
    #print(df_results.iloc[row_ind])

    #2) Generate test & ho results excluding ties
    if len(reduced_cols)==0: mod_name = base+'_'+sc_metric+'_v'+v+'_exties'
    else: mod_name = base+'_reduced'+str(len(reduced_cols))+'_'+sc_metric+'_v'+v+'_exties'
    if not(mod_name in list(df_results['model'])):
        df_results = df_results.append(pd.Series(), ignore_index=True)
        row_ind = len(df_results)-1
        df_results.loc[row_ind,'model']=mod_name
        df_results.loc[row_ind,'sc_metric']=sc_metric
        df_results.loc[row_ind,'ties']=0
    else:
        row_ind = df_results.index[df_results['model']==mod_name].values[0]
        df_results.loc[row_ind,'sc_metric']=sc_metric
        df_results.loc[row_ind,'ties']=0
    #Test excluding ties
    Y_test_exties, Y_pred_exties, df_flat_exties = rem_ties(Y_test, pred, df_flat.copy(), X_train)
    pred_ror_test, fav_ror_test, pred_prec_test, fav_prec_test, acc_test, p0_freq_test = eval_predictions(Y_test_exties.astype('int'), Y_pred_exties, y_dict, p_res, df_flat_exties, X_train, g_ind, r_ind)
    df_results.loc[row_ind,'m_test']=len(Y_pred_exties)
    df_results.loc[row_ind,'pred_ror_test']=pred_ror_test
    df_results.loc[row_ind,'fav_ror_test']=fav_ror_test
    df_results.loc[row_ind,'pred_prec_test']=pred_prec_test
    df_results.loc[row_ind,'fav_prec_test']=fav_prec_test
    df_results.loc[row_ind,'acc_test']=acc_test
    df_results.loc[row_ind,'p0_freq_test']=p0_freq_test
    #HO excluding ties
    Y_test_ho_exties, Y_pred_ho_exties, df_flat_ho_exties = rem_ties(Y_test_ho, pred_ho, df_flat_ho.copy(), X_train)
    pred_ror_ho, fav_ror_ho, pred_prec_ho, fav_prec_ho, acc_ho, p0_freq_ho = eval_predictions(Y_test_ho_exties.astype('int'), Y_pred_ho_exties, y_dict, p_res, df_flat_ho_exties, X_train, g_ind, r_ind)
    df_results.loc[row_ind,'m_ho']=len(Y_pred_ho_exties)
    df_results.loc[row_ind,'pred_ror_ho']=pred_ror_ho
    df_results.loc[row_ind,'fav_ror_ho']=fav_ror_ho
    df_results.loc[row_ind,'pred_prec_ho']=pred_prec_ho
    df_results.loc[row_ind,'fav_prec_ho']=fav_prec_ho
    df_results.loc[row_ind,'acc_ho']=acc_ho
    df_results.loc[row_ind,'p0_freq_ho']=p0_freq_ho
    #Average of test & ho ror
    df_results.loc[row_ind,'pred_ror_comb']=(pred_ror_test+pred_ror_ho)/2
    #print(df_results.iloc[row_ind])   

    return df_results

def rem_ties(Y_test, pred, df_flat, X_train):
  print(f'\n------ Evaluating model excluding predicted ties------\n')
  temp_pred = pred
  df_orig = df_flat.copy()
  print(f'a) df_flat.shape={df_flat.shape}, df_orig.shape={df_orig.shape}')

  df_flat['temp_pred']=[None]*len(X_train)+list(temp_pred)
  print(f'b) df_flat.shape={df_flat.shape}, df_orig.shape={df_orig.shape}')

  #Remove row where ties are predicted from df_flat, Y_test & Y_pred
  rem_inds = df_flat.index[df_flat['temp_pred']>2].tolist()
  np_rem_inds = list(np.subtract(rem_inds, len(X_train)))
  df_flat = df_flat.drop(rem_inds)
  df_flat.reset_index(inplace=True)
  Y_test_temp = np.delete(Y_test, np_rem_inds, axis=0)
  Y_pred_temp = np.delete(temp_pred, np_rem_inds, axis=0)
  print(f'c) df_flat.shape={df_flat.shape}, df_orig.shape={df_orig.shape}, Y_test{Y_test.shape}, Y_test_temp{Y_test_temp.shape}, temp_pred{temp_pred.shape}, Y_pred_temp{Y_pred_temp.shape}')
  df_flat_exties = df_flat.loc[:, list(df_orig.columns)]
  print(f'd) df_flat_exties.shape = {df_flat_exties.shape}')
  print('---------------------------------------------------------------------------')

  return Y_test_temp, Y_pred_temp, df_flat_exties

def prof_calc(win_array, pred_array, odds_array, stake=1.0):
    prof = 0 
    bet_won = 0
    if np.sum(win_array)==1: tie_flag=0
    if np.sum(win_array)==2: tie_flag=2
    if np.sum(win_array)==3: tie_flag=3

    if np.sum(pred_array)==1: #Single winner predicted
      for i in range(len(pred_array)):
          if win_array[i] == 1 and pred_array[i] == 1:
              bet_won = 1
              if tie_flag == 0: prof = stake*(odds_array[i] - 1)                      #P/L Outright Win
              if tie_flag == 2: prof = ((stake/2)*(odds_array[i] - 1)) - (stake/2)    #P/L 2-way tie
              if tie_flag == 3: prof = ((stake/3)*(odds_array[i] - 1)) - (stake*2/3)  #P/L 3-way tie
    else: #Multiple winners predicted
      t_prof, n_win = 0,0
      n_pred = np.sum(pred_array)
      t_stake = stake / n_pred
      for i in range(len(pred_array)):
          if win_array[i] == 1 and pred_array[i] == 1:
              bet_won = 1
              n_win+=1
              if tie_flag == 0: t_prof += t_stake*(odds_array[i] - 1)                        #P/L Outright Win
              if tie_flag == 2: t_prof += ((t_stake/2)*(odds_array[i] - 1)) - (t_stake/2)    #P/L 2-way tie
              if tie_flag == 3: t_prof += ((t_stake/3)*(odds_array[i] - 1)) - (t_stake*2/3)  #P/L 3-way tie
      if bet_won == 1:
        prof = t_prof - ((n_pred - n_win)*t_stake)

    if bet_won == 0:
        prof = -stake                                                           #P/L Loss
        
    return prof

def eval_predictions(y_target, y_pred, y_dict, p_res, df_flat, X_train, g_ind, r_ind):
  pred_ror, fav_ror, pred_prec, fav_prec, acc, p0_freq = 0,0,0,0,0,0
  acc   = accuracy(y_target, y_pred)
  print(f'Accuracy = {acc}')

  pred_res = np.zeros((y_target.shape[0], g_ind))
  y_check = np.zeros((y_target.shape[0], g_ind))
  for i in range(pred_res.shape[0]):
    pred_res[i]=y_dict[y_pred[i]]
    y_check[i]=y_dict[y_target[i]]

  print('\ny_check win percentages for bookies favourite, mid and weakest players [Precision : TP / predicted yes]')
  print(f'fav_w={round(np.divide(np.sum(y_check[:,0]), y_check.shape[0]),4)}, mid_w={round(np.divide(np.sum(y_check[:,1]), y_check.shape[0]),4)}, weak_w={round(np.divide(np.sum(y_check[:,2]), y_check.shape[0]),4)} across {y_check.shape[0]} groups (*including ties)')
  fav_prec = np.divide(np.sum(y_check[:,0]), y_check.shape[0])
  print()
  print('pred_res frequencies for p0, p1 and p2 players')
  print(f'p0={round(np.divide(np.sum(pred_res[:,0]), pred_res.shape[0]),4)}, p1={round(np.divide(np.sum(pred_res[:,1]), pred_res.shape[0]),4)}, p2={round(np.divide(np.sum(pred_res[:,2]), pred_res.shape[0]),4)} across {pred_res.shape[0]} groups (*including ties)')
  p0_freq = round(np.divide(np.sum(pred_res[:,0]), pred_res.shape[0]),4)

  print('\npred_res overall win percentage for p0, p1 and p2 player predictions [Precision : TP / predicted yes]')
  w_cnt=0
  pred_pl, fav_pl = 0,0
  for i in range(pred_res.shape[0]):
    if np.matmul(pred_res[i], y_check[i])>0: w_cnt+=1
    temp_id=df_flat.loc[i+X_train.shape[0], 'pod_id_d'+str(r_ind)]
    temp_prof_loss = prof_calc(y_check[i], pred_res[i], 
                               df_flat.loc[i+X_train.shape[0], ['p0_pl_back_d'+str(r_ind), 'p1_pl_back_d'+str(r_ind), 'p2_pl_back_d'+str(r_ind)]].values)
    pred_pl += temp_prof_loss
    fav_prof_loss = prof_calc(y_check[i], [1,0,0],
                              df_flat.loc[i+X_train.shape[0], ['p0_pl_back_d'+str(r_ind), 'p1_pl_back_d'+str(r_ind), 'p2_pl_back_d'+str(r_ind)]].values)
    fav_pl += fav_prof_loss
    if p_res==1: print(f'[{i}] pred_res{pred_res[i]} {y_check[i]}y_check | ({y_pred[i]}) ({y_target[i]}), w_cnt={w_cnt} ({temp_id}), pred_pl = {round(temp_prof_loss, 2)}[{round(pred_pl,2)}], fav_pl = {round(fav_prof_loss,2)}[{round(fav_pl,2)}]')

  try: pred_prec = w_cnt/pred_res.shape[0]
  except ZeroDivisionError: None
  try: fav_ror = fav_pl/pred_res.shape[0]
  except ZeroDivisionError: None
  try: pred_ror = pred_pl/pred_res.shape[0]
  except ZeroDivisionError: None
  print(f'\nPrecision = {round(pred_prec,4)}, fav_pl = {round(fav_pl,2)}, pred_pl = {round(pred_pl,2)}')

  print(f'\nSummary: Accurcacy = {round(acc,4)}, Bookies Fav Precision = {round(fav_prec,4)}, Model Precision = {round(pred_prec,4)}, fav_ror = {round(fav_ror,4)}, pred_ror = {round(pred_ror,4)}')

  return pred_ror, fav_ror, pred_prec, fav_prec, acc, p0_freq

def gen_ho(df_flat, X_train, g_ind, tour_ind, r_ind, x_cols, y_cols, X_features, Y_classes, tourn_cols, pl_id, pl_col, p012_cols):
    file_name = tour_ind+'_R'+str(r_ind)+'_'+str(g_ind)+'ball_classification_ho.csv'
    print(f'\nGenerating df_flat_ho from {file_name}')
    data_ho = pd.read_csv(file_name)
    # number of groups with favourite, mid and weak player identified based on bookies odds (when available)
    m_grp_with_p012_ho = (len(data_ho) - data_ho.loc[data_ho['fav_ind_d'+str(r_ind)].isin([8,9])].shape[0]) / 3 
    #Calculate win percentages for bookies favourite, mid and weakest players
    fav_w_ho = data_ho.loc[(data_ho['fav_ind_d'+str(r_ind)]==1) & (data_ho['win_ind_d'+str(r_ind)]==1)].shape[0] / m_grp_with_p012_ho
    mid_w_ho = data_ho.loc[(data_ho['mid_ind_d'+str(r_ind)]==1) & (data_ho['win_ind_d'+str(r_ind)]==1)].shape[0] / m_grp_with_p012_ho
    weak_w_ho = data_ho.loc[(data_ho['weak_ind_d'+str(r_ind)]==1) & (data_ho['win_ind_d'+str(r_ind)]==1)].shape[0] / m_grp_with_p012_ho
    print(f'\nfav_w_ho={round(fav_w_ho,4)}, mid_w_ho={round(mid_w_ho,4)}, weak_w_ho={round(weak_w_ho,4)} across {m_grp_with_p012_ho} groups (*including ties)')

    #Drop groups without favourite, mid and weak player identified
    data_with_p012_ho = data_ho.loc[(data_ho['fav_ind_d'+str(r_ind)]!=8) & (data_ho['fav_ind_d'+str(r_ind)]!=9)]
    print(f'\ndata_with_p012_ho.shape={data_with_p012_ho.shape}')

    #Flatten each group (currently represented by 3 rows per group) to into a single row per group
    df_flat_ho = pd.DataFrame({'pod_id_d'+str(r_ind):list(data_with_p012_ho['pod_id_d'+str(r_ind)].unique())})
    for i in range(len(df_flat_ho)):
    #for i in [0]:
      if i==0: #initialize all columns
          for c in tourn_cols: df_flat_ho[c]=[None]*len(df_flat_ho)
          for pl in pl_id: 
            df_flat_ho[pl+'_'+pl_col]=[None]*len(df_flat_ho)   
          for pl in pl_id:
              for x in x_cols: 
                df_flat_ho[pl+'_'+x]=[None]*len(df_flat_ho)
          for pl in pl_id:
              for y in y_cols: 
                df_flat_ho[pl+'_'+y]=[None]*len(df_flat_ho)
          df_flat_ho['y_target']=[None]*len(df_flat_ho)

      #Populate df_flat_ho
      temp_df = data_with_p012_ho[data_with_p012_ho['pod_id_d'+str(r_ind)]==df_flat_ho.loc[i,'pod_id_d'+str(r_ind)]].reset_index()
      df_flat_ho.loc[i,tourn_cols] = temp_df.loc[0,tourn_cols]
      #get fav, mid & weak inds
      p0_ind = temp_df.loc[temp_df['fav_ind_d'+str(r_ind)]==1].index[0]
      p1_ind = temp_df.loc[temp_df['mid_ind_d'+str(r_ind)]==1].index[0]
      p2_ind = temp_df.loc[temp_df['weak_ind_d'+str(r_ind)]==1].index[0]
      p012_ind=[p0_ind, p1_ind, p2_ind]
      for j in range(len(pl_id)):
        df_flat_ho.loc[i,p012_cols[j][0]] = temp_df.loc[p012_ind[j],'pod_pls_d'+str(r_ind)]
        df_flat_ho.loc[i,p012_cols[j][1:-6]] = temp_df.loc[p012_ind[j],x_cols].values
        df_flat_ho.loc[i,p012_cols[j][-6:]] = temp_df.loc[p012_ind[j],y_cols].values
      #Calculate y_target
      win_lst = list(df_flat_ho.loc[i,Y_classes])
      if [1,0,0]==win_lst: df_flat_ho.loc[i,'y_target']=0
      if [0,1,0]==win_lst: df_flat_ho.loc[i,'y_target']=1
      if [0,0,1]==win_lst: df_flat_ho.loc[i,'y_target']=2
      if [1,1,0]==win_lst: df_flat_ho.loc[i,'y_target']=3
      if [1,0,1]==win_lst: df_flat_ho.loc[i,'y_target']=4
      if [0,1,1]==win_lst: df_flat_ho.loc[i,'y_target']=5
      if [1,1,1]==win_lst: df_flat_ho.loc[i,'y_target']=6
    print()
    print(df_flat_ho.columns)
    print(f'\ndf_flat_ho.shape={df_flat_ho.shape}, null_values={df_flat_ho.isnull().values.any()}\n')

    #Veracity Check Data
    if r_ind==1:
        grp_id = '2021_1_14_v0_d1_grp0_PGA'
        #grp_id = '2021_8_26_v0_d1_grp9_PGA'
    if r_ind==2:
        grp_id = '2021_8_19_v0_d2_grp8_PGA'
        #grp_id = '2021_1_14_v0_d2_grp0_PGA'
    print(data_with_p012_ho.loc[data_with_p012_ho['pod_id_d'+str(r_ind)]==grp_id, ['pod_pls_d'+str(r_ind),'pl_back_d'+str(r_ind),'Length_sg1','tb12_p1','R'+str(r_ind)+'_scr', 'win_ind_d'+str(r_ind)]])
    print()
    print(df_flat_ho.loc[df_flat_ho['pod_id_d'+str(r_ind)]==grp_id, ['p0_pod_pls_d'+str(r_ind),'p0_pl_back_d'+str(r_ind),'p0_Length_sg1','p0_tb12_p1','p0_R'+str(r_ind)+'_scr', 'p0_win_ind_d'+str(r_ind)]])
    print()
    print(df_flat_ho.loc[df_flat_ho['pod_id_d'+str(r_ind)]==grp_id, ['p1_pod_pls_d'+str(r_ind),'p1_pl_back_d'+str(r_ind),'p1_Length_sg1','p1_tb12_p1','p1_R'+str(r_ind)+'_scr', 'p1_win_ind_d'+str(r_ind)]])
    print()
    print(df_flat_ho.loc[df_flat_ho['pod_id_d'+str(r_ind)]==grp_id, ['p2_pod_pls_d'+str(r_ind),'p2_pl_back_d'+str(r_ind),'p2_Length_sg1','p2_tb12_p1','p2_R'+str(r_ind)+'_scr', 'p2_win_ind_d'+str(r_ind)]])
    print()
    print(df_flat_ho.loc[df_flat_ho['pod_id_d'+str(r_ind)]==grp_id, 'y_target'])

    #Generate X_ho & Y_ho Dataframes
    X_ho = df_flat_ho.loc[:,X_features]     # pl_back_d2 -> R1_vFavg for p012
    Y_ho = df_flat_ho.loc[:,'y_target']     # y_target for p012
    Y_oh_ho = df_flat_ho.loc[:,Y_classes]   # win_ind_d2 for p012

    print(f'\nX_ho={type(X_ho)}, {X_ho.shape}, {X_ho.values[0,0:5]}')
    print(f'Y_ho={type(Y_ho.values)}, {Y_ho.shape}, {Y_ho.values[0]}')
    print(f'Y_oh_ho={type(Y_oh_ho.values)}, {Y_oh_ho.shape}, {Y_oh_ho.values[0]}\n')

    #Update df_flat and test array with holdout data
    df_flat_temp = df_flat.copy()
    df_flat = df_flat.drop(list(range(len(X_train), len(df_flat_temp))))
    print(f'pre ho append: df_flat{df_flat.shape}, df_flat_temp{df_flat_temp.shape}')
    df_flat = df_flat.append(df_flat_ho, ignore_index = True)
    X_test_ho = X_ho.values
    Y_test_ho = Y_ho.values
    Y_test_oh_ho = Y_oh_ho.values

    print(f'post ho append: df_flat{df_flat.shape}, df_flat_temp{df_flat_temp.shape}')
    print(f'\nX_test_ho={type(X_test_ho)}, {X_test_ho.shape}, {X_test_ho[-1,0:5]}')
    print(f'\nY_test_ho={type(Y_test_ho)}, {Y_test_ho.shape}, {Y_test_ho[-1]}')
    print(f'\nY_test_oh_ho={type(Y_test_oh_ho)}, {Y_test_oh_ho.shape}, {Y_test_oh_ho[-1]}')
    df_flat_ho = df_flat

    return df_flat_ho, X_test_ho, Y_test_ho, Y_test_oh_ho

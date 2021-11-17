from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast

def make_bars_all(best_mod, df_res, k_fold):
  plt.rcParams["figure.figsize"] = [15,8]
  df_ind=df_res[df_res['model']==best_mod].index.values[0]

  pred_lst = list(df_res.loc[df_ind, ['p0_freq_train','p0_freq_cv', 'p0_freq_test', 'p0_freq_ho']])
  pred_lst += list(df_res.loc[df_ind, ['pred_prec_train', 'pred_prec_cv', 'pred_prec_test', 'pred_prec_ho']])
  pred_lst += list(df_res.loc[df_ind, ['pred_ror_train', 'pred_ror_cv', 'pred_ror_test', 'pred_ror_ho']])

  fav_lst = [0,0,0,0]
  fav_lst += list(df_res.loc[df_ind, ['fav_prec_train', 'fav_prec_cv', 'fav_prec_test', 'fav_prec_ho']])
  fav_lst += list(df_res.loc[df_ind, ['fav_ror_train', 'fav_ror_cv', 'fav_ror_test', 'fav_ror_ho']])

  index = ['p0_freq_train','p0_freq_cv', 'p0_freq_test', 'p0_freq_ho', 'prec_train', 'prec_cv', 'prec_test', 'prec_ho', 'ror_train', 'ror_cv', 'ror_test', 'ror_ho']
  df_bar = pd.DataFrame({'model prediction':pred_lst,
                         'bookies favorite':fav_lst}, index=index)
  
  #https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
  x = np.arange(len(index))  # the label locations
  width = 0.35  # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.barh(x - width/2, list(np.round(np.array(pred_lst),3)), width, label='model prediction', edgecolor='black')
  rects2 = ax.barh(x + width/2, list(np.round(np.array(fav_lst),3)), width, label='bookies favorite', edgecolor='black')
  # Add some text for labels, title and custom x-axis tick labels, etc.%
  ax.set_ylabel('Model Prediction vs Bookies Favourite comparisons\n(across train, cv, test & ho datasets)\n', fontsize=14)
  ax.set_xlabel('\nPercentages', fontsize=14)
  ax.set_title('Model Performance Summary ['+df_res.loc[df_ind,'model'][:100]+'...]\n', fontweight='bold', fontsize=16)
  ax.set_yticks(x)
  ax.set_yticklabels(index)
  ax.tick_params(labelsize=12)
  ax.legend(fontsize=14)
  ax.grid()

  ax.bar_label(rects1, padding=3)
  ax.bar_label(rects2, padding=3)
  fig.tight_layout()
  sns.set_style('dark')

  reduced_cols = ast.literal_eval(df_res.loc[df_ind,'reduced_cols'])
  if len(reduced_cols) <= 20: n, ht = 3, 0.45
  else: n, ht = 3, 0.25
  col_split = [reduced_cols[i:i + n] for i in range(0, len(reduced_cols), n)]
  col_str = ''
  for i in col_split: col_str += '\n'+str(i)
  
  if len(reduced_cols) > 30:
    n, ht = 3, 0.15
    col_split = [reduced_cols[i:i + n] for i in range(0, len(reduced_cols[:30]), n)]
    col_str = ''
    for i in col_split: col_str += '\n'+str(i)
  
  misc_lst = ['m_ho','m_test','m_cv','m_train','stdev_pred_ror_cv','k_fold','n_features']
  misc_val = list(df_res.loc[df_ind,['m_ho','m_test','m_cv','m_train','stdev_pred_ror_cv']])
  misc_val += [k_fold, len(reduced_cols)]
  misc_str = '\n'.join((misc_lst[0]+'='+str(int(misc_val[0])), misc_lst[1]+'='+str(int(misc_val[1])), misc_lst[2]+'='+str(int(misc_val[2])), 
                        misc_lst[3]+'='+str(int(misc_val[3])), misc_lst[4]+'='+str(misc_val[4]), misc_lst[5]+'='+str(misc_val[5]),
                        misc_lst[6]+'='+str(misc_val[6]), col_str))
  
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  ax.text(1.025, ht, misc_str, transform=ax.transAxes, fontsize=14, bbox=props)
  
  plt.show()

  return df_bar



import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=7000, n_features=10, n_informative=5, n_redundant=2,
                           n_repeated=0, scale=None, shift=None, shuffle=False, class_sep=0.5, 
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def woe_line(X,         # 1-d array with independent feature 
             y,         # 1-d array with target feature
             n_buckets, # number of buckets to plot
             var_nm,    # optional, var_name to show above the plot
             target_nm, # optional, target_name to show above the plot
             plot_hist  # optional, if True — histogram of X is displayed
            ):
  
  if plot_hist == True:
    plt.hist(X, 30)
    plt.show()
  array_X_y = [list(a) for a in zip(X, y)]
  array_X_y = sorted(array_X_y, key = lambda x: x[0])
  arrays = np.array_split(array_X_y, n_buckets)
  xx, yy = [], []
  i = 0
  for arr in arrays:
    if arr[0][0] == 0 : break
    num_0, num_1 = 0, 0
    for el in arr:
      if el[1] == 0: num_0 += 1
      else: num_1 += 1
    xx.append(np.median(arr, axis = 0)[0])
    yy.append(-np.log(num_1/num_0))
    plt.scatter(xx[i], yy[i], color = 'firebrick')
    plt.plot([xx[i], xx[i]], [yy[i] - abs(yy[i]/2), yy[i] + abs(yy[i]/2)],
             linestyle = '-', marker = '_', color = 'firebrick')
    i += 1
  array = [list(a) for a in zip(xx,yy)]
  xx = [x[0] for x in array]
  yy = [x[1] for x in array]
  plt.plot(xx, yy, linestyle = 'dashed', color = 'firebrick')
  plt.title("{} | {} AUC = {}".format(var_nm, target_nm, round(1 - roc_auc_score(y, X), 3)))

res = woe_line(X_train[:, 0], y_train, n_buckets=10, plot_hist=False, 
               var_nm = 'X_0', target_nm = 'y')
plt.show()

res = woe_line(np.clip(X_train[:, 0], -200, 0), y_train, n_buckets=10, plot_hist=False, 
               var_nm = 'X_0', target_nm = 'y')

plt.show()


#ЗАДАНИЕ 2
for i in range (10):
  #res = woe_line(X_train[:, i], y_train, n_buckets=10, plot_hist=False, 
               #var_nm = f'X_{i}', target_nm = 'y')
  res = woe_line(np.clip(X_train[:, i], -200, 0), y_train, n_buckets=10, plot_hist=False, 
               var_nm = f'X_{i}', target_nm = 'y')
  plt.show()
#X3, X5, X7, X8, X9 не монотонны, у X4 маленький AUC
#теперь надо линеаризовать...
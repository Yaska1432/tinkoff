import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pylab import *
import seaborn as sns
import csv

data_test, data_control = [], []
with open('test_3.csv', "r") as csv_file:
	text = csv.reader(csv_file)
	for row in text:
		if row[0] == 'test' : data_test.append(int(row[2]))
		elif row[0] == 'control' : data_control.append(int(row[2]))

data = pd.read_csv('test_3.csv')

table = pd.crosstab(
    data['group'],
    data['click_flg'],
    margins = True
)
obs1 = np.append(table[0][1], table[1][1])
rows = table.iloc[0:2,2].values



f = plt.figure(figsize=(12,5))

ax = f.add_subplot(121)
sns.kdeplot(data = data_control, color = 'olivedrab', ax = ax)
conv_control = obs1[0] / rows[0]
ax.set_title(f'conversion in control group = {conv_control:.4}')
ax = f.add_subplot(122)
sns.kdeplot(data = data_test, color = 'gold', ax = ax)
conv_test = obs1[1] / rows[1]
ax.set_title(f'conversion in test group = {conv_test:.4}')
show()

stat, p_value, num_of_fr, array = stats.chi2_contingency(table)

if p_value > 0.01:
	print('Конверсия стат значимо изменилась', 'Хи-квадрат критерий=', stat,'\np-value = ', p_value)
	if conv_test > conv_control: print('конверсия увеличилась')
	else: print('конверсия уменьшилась')
else:
    print('Конверсия стат значимо не изменилась','\nХи-квадрат критерий=', stat,'\np-value = ', p_value)
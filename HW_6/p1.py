import csv
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import scipy.stats as stats
from pylab import *

data_test = []
data_control = []


with open('test_1.csv', "r") as csv_file:
	text = csv.reader(csv_file)
	for row in text:
		if row[0] == 'test' : data_test.append(float(row[2]))
		elif row[0] == 'control' : data_control.append(float(row[2]))


f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
stats.probplot(data_test, dist="norm", plot=ax1)
stats.probplot(data_control, dist="norm", plot=ax2)
show()

for group in [data_test, data_control]:
    W_value,p_value = stats.shapiro(group)
    if p_value > 0.01:
        print('Normal','W =',round(W_value,4),'p-value = ',round(p_value,4))
    else:
        print('Not normal','W =',round(W_value,4),'p-value = ',round(p_value,4))

#данные распределены нормально, можем воспользоваться критериями фишера и Стьюдента


stat, p_val = stats.levene(data_test, data_control)
if p_value > 0.01:
	print('Статистически значимой разницы дисперсий нет','\np-value', round(p_val, 4), 'Levene-критерий',stat)
else:
	print('Разница дисперсий статистически значима','\np-value', round(p_val, 4), 'Levene-критерий',stat)

#проверили равенство дисперсий

print(ttest_ind(data_test, data_control))
stat, p_val = stats.mannwhitneyu(data_test, data_control, alternative = 'two-sided')
if p_value > 0.01:
	print('Статистически значимой разницы средних нет','\np-value', round(p_val, 4), 'T-критерий',stat)
else:
	print('Разница средних статистически значима','\np-value', round(p_val, 4), 'T-критерий',stat)

#убедились в равенстве средних


import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
from pylab import *

data_test = []
data_control = []


with open('test_2.csv', "r") as csv_file:
	text = csv.reader(csv_file)
	for row in text:
		if row[0] == 'test' : data_test.append(float(row[2]))
		elif row[0] == 'control' : data_control.append(float(row[2]))


f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
stats.probplot(data_test, dist = "norm", plot = ax1)
stats.probplot(data_control, dist = "norm", plot = ax2)
show()


stat, p_val = stats.mannwhitneyu(data_test, data_control, alternative = "greater")
if p_val > 0.01:
    print('Статистически значимой разницы нет, сумма чека не изменилась','\np-value', round(p_val, 4), '\nU-критерий',stat)
else:
    print('Разница статистически значима, значит сумма чека увеличилась','\np-value', round(p_val, 4), '\nU-критерий',stat)

print(mean(data_test))
print(mean(data_control))
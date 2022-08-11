import numpy as np
import pandas as pd
import scipy
import math
import matplotlib.pyplot as plt

from statsmodels.stats.weightstats import _tconfint_generic

df = pd.read_csv('data.csv', sep = ',')

plt.figure()
plt.subplot (1, 2, 1)
plt.hist(df.required_amt[df.required_amt < df.required_amt.quantile(0.99)],
           bins = 15,
           color = 'b',
           alpha = 0.9,
           label = 'Запрошенная сумма')
plt.legend()

plt.subplot (1, 2, 2)
plt.hist(df.monthly_income_amt[df.monthly_income_amt < df.monthly_income_amt.quantile(0.99)],
           bins = 15,
           color = 'g',
           alpha = 0.9,
           label = 'Доход клиента')
plt.legend()

x1, y1 = [], []
for x in df.monthly_income_amt:
	x1.append(math.log(x))
for y in df.required_amt:
	y1.append(math.log(y))

monthly_income_conf_int = _tconfint_generic (np.array(x1).mean(),
                                            np.array(x1).std(ddof = 1) / np.sqrt(len(df)),
                                            len(df) - 1,
                                            0.05,
                                            'two-sided')
array = np.array(y1)

required_conf_int = _tconfint_generic (array[~np.isnan(array)].mean(),
										array[~np.isnan(array)].std(ddof = 1) / np.sqrt(len(df)),
										len(df) - 1,
										0.05,
										'two-sided')
                                      

print(f'monthly_income_amt 99% confidence interval: [{math.exp(monthly_income_conf_int[0]):,.2f} - {math.exp(monthly_income_conf_int[1]):,.2f}]')
print(f'required_amt 99% confidence interval: [{math.exp(required_conf_int[0]):,.2f} - {math.exp(required_conf_int[1]):,.2f}]')

plt.show()


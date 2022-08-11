import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
from pylab import *
from math import factorial as fact
from scipy.integrate import quad
from scipy import integrate

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

distr = input('нормальное (E, D) / равномерное (a, b) / биномиальное (n, p) / экспоненциальное (l):\n')
N = int(input('размер выборки: '))

sample = np.linspace(-3, 3, N)
def norm_p (x, loc, scale):
    return exp ((-(x - loc) ** 2) / (2 * scale)) / (math.sqrt(2 * math.pi) * scale)
def bin (n, k):
	return fact(n) / (fact(k) * fact(n - k))


if distr.split()[0] == 'нормальное' :
	loc = float(distr.split('(')[1].split(',')[0])
	scale = float(distr.split()[2].split(')')[0])
	ax1.plot (sample, norm_p (sample, loc, scale), color = 'red')
	n_sample = sps.norm (loc, scale).rvs(size = N)
	ax1.hist(n_sample, density = True, edgecolor = 'black', bins = int(sqrt(N)))
	ax1.set_xlabel('Значения')
	ax1.set_ylabel('Плотность')
	def f (x, loc, scale):
		return x * norm_p(x, loc, scale)
	E = integrate.quad (f, -np.inf, np.inf, args = (loc, scale))[0]
	aver = sum(t / N for t in n_sample)
	plt.text(-3, 0.35, f'Теор. матож.: {E}\nСр. арифм.: {aver}\nРазница: {aver - E}')


elif distr.split()[0] == 'равномерное' :
	a = float(distr.split('(')[1].split(',')[0])
	b = float(distr.split()[2].split(')')[0])
	ax1.plot([a, b], [1 / (b - a), 1 / (b - a)], color = 'red')
	ax1.plot([a - 0.5, a], [0, 0], color = 'red')
	ax1.plot([b, b + 0.5], [0, 0], color = 'red')
	ax1.set_xlim([a - 1, b + 1])
	n_sample = np.random.uniform (a, b, N)
	def f (x, a, b): return x / (b - a)
	E = integrate.quad (f, a, b, args = (a, b))[0]
	aver = sum(t / N for t in n_sample)
	ax1.hist(n_sample, density = True, edgecolor = 'black', bins = int(sqrt(N)))
	plt.text(1, 1.1, f'Теор. матож.: {E}\nСр. арифм.: {aver}\nРазница: {aver - E}')

elif distr.split()[0] == 'биномиальное' :
	n = int(distr.split('(')[1].split(',')[0])
	p = float(distr.split()[2].split(')')[0])
	n_sample = np.random.binomial(n, p, N)
	for i in range (n):
		ax1.scatter(i, bin(n, i) * p ** i * (1 - p) ** (n - i), c = 'red', s = 2)
	ax1.hist(n_sample, density = True, edgecolor = 'black', bins = int(sqrt(N)))
	E = n * p
	aver = sum(t / N for t in n_sample)
	plt.text(40, 0.06, f'Теор. матож.: {E}\nСр. арифм.: {aver}\nРазница: {aver - E}')

elif distr.split()[0] == 'экспоненциальное' :
	l = float(distr.split('(')[1].split(')')[0])
	sample = np.random.exponential(scale = l, size = N)
	grid = np.linspace(sps.expon.ppf(scale = l, q = 0.001), 
	                   sps.expon.ppf(scale = l, q = 0.99), 100)
	ax1.plot(grid, sps.expon.pdf(grid, scale = l))
	ax1.hist(sample, density = True, edgecolor = 'black', bins = int(sqrt(N)))
	def f (x, l): return x * l * exp(-l * x)
	E = integrate.quad (f, 0, np.inf, args = (l))[0]
	aver = sum(t / N for t in sample)
	plt.text(0.6, 1.6, f'Теор. матож.: {E}\nСр. арифм.: {aver}\nРазница: {aver - E}')


ax1.grid('auto')
ax1.set_title(f'График плотности: {distr}')
ax1.legend(fontsize = 14, loc = 1)
plt.show()
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

df = pd.read_excel('/Users/yana/Downloads/1.xls')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data, data2 = [0 for i in range(len(df))], [0 for i in range(len(df))]

def age (born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

i = 0
for row in df.itertuples():
    data[i] = age(row[2])
    i += 1

df['Age'] = data

h_ages = sorted(df.Age.unique())
ages = []
l = 0
for k in range(int(len(h_ages) / 5) + 1):
	hh_ages = []
	for i in range (5):
		if (l + i < len(h_ages)):
			hh_ages.append(h_ages[l + i])
	l += 5
	ages.append(hh_ages)

colors = plt.cm.jet (np.linspace (0, 1, len(ages)))

print(df)

k = 0
labels = []
parts = []
for ag in ages:
	xx, yy = 0, 0
	for row in df.itertuples():
		if row[7] in ag :
			if pd.isnull(row[5]):
				yy += 1
			else :
				xx += 1
				yy += 1
	if yy != 0 :
		if (k + 1) < len(ages) :
			labels.append(f'{ag[0]} - {ag[4]} лет')
		else :
			labels.append(f'{ag[0]} лет')
	parts.append(xx / yy)
	k += 1

index = np.arange(len(labels))
ax.bar (index, parts)
ax.set_xticks(index - 0.4)
ax.set_xticklabels(labels)
plt.grid()
plt.xlabel('Доля конверсии в утилизацию')
plt.ylabel('Возрастной сигмент клиентов')
ax.set_title('Распределение конверсии в утилизацию в зависимости от возраста', weight = 'bold')
plt.show()


"""
df = pd.read_excel('1.xls')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data, data2 = [0 for i in range(len(df))], [0 for i in range(len(df))]

i = 0
for row in df.itertuples():
	if pd.isnull(row[4]): continue
	else : 
		timedelta = row[5].date() - row[3].date()
		data[i] = int(f'{timedelta.days}')
		data2[i] = int(row[4])
		i += 1

for i in range (len(df)):
	if data2[i] == 0 : break

ax.scatter (data[:i], data2[:i], color = 'orange', s = 10)
plt.xlabel('Время утилизации')
plt.ylabel('Сумма покупки')
ax.set_title('Зависимость суммы покупки первой покупки от скорости утилизации', weight = 'bold', size = 12)

plt.show()



df = pd.read_excel('/Users/yana/Downloads/1.xls')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data, data2 = [0 for i in range(len(df))], [0 for i in range(len(df))]

def age (born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

i = 0
for row in df.itertuples():
    data[i] = age(row[2])
    data2[i] = f'{row[3].strftime("%m/%d/%Y")}'
    i += 1

df['Age'] = data
df['Reg_date'] = data2

h_ages = sorted(df.Age.unique())
ages = []
l = 0
print(len(h_ages))
for k in range(int(len(h_ages) / 5) + 1):
	hh_ages = []
	for i in range (5):
		if (l + i < len(h_ages)):
			hh_ages.append(h_ages[l + i])
	l += 5
	ages.append(hh_ages)

colors = plt.cm.jet (np.linspace (0, 1, len(ages)))

k = 0
for ag in ages:
	xx, yy = [], []
	i = 0
	for row in df.groupby(['Reg_date', 'Age']).size().reset_index(name = 'counts').itertuples():
		if row[2] in ag :
			if row[1] not in xx :
				xx.append(row[1])
				yy.append(row[3])
			else :
				yy[i] += row[3]

	if (k + 1) < len(ages) : ax.scatter (xx, yy, color = colors[k], label = f'{ag[0]} - {ag[4]} лет')
	else : ax.scatter (xx, yy, color = colors[k], label = f'{ag[0]} лет')
	k += 1


ax.set_yscale('log')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(8))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(8))
plt.xlabel('День регистрации')
plt.ylabel('Кол-во клиентов')
ax.set_title('Распределение клиентов по возрасту в зависимости от времени регистрации')
plt.legend()
plt.show()"""



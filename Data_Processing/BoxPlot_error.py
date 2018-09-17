#By Zhenghang(Klaus) Zhong

#Box Plot of error distribution

from pandas import DataFrame
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot
# load results into a dataframe
filenames_128 = ['dis_diff_128.csv']
filenames_256 = ['dis_diff_256.csv']
filenames_512 = ['dis_diff_512.csv']
results = DataFrame()

for name in filenames_128:
	results_128 = read_csv(name, header=0,usecols = [1])
# describe all results, as 1 unit = 10cm, we want to transfer to meters, /10
results_128 = results_128.div(10, axis = 0)

for name in filenames_256:
	results_256 = read_csv(name, header=0,usecols = [1])
# describe all results
results_256 = results_256.div(10, axis = 0)

for name in filenames_512:
	results_512 = read_csv(name, header=0,usecols = [1])
# describe all results
results_512 = results_512.div(10, axis = 0)

print(results_128.describe())
print(results_256.describe())
print(results_512.describe())

# box and whisker plot
df = pd.DataFrame(np.concatenate((results_128,results_512),axis = 1),
columns=['128', '512'])

df.boxplot(sym='k',showmeans = True,showfliers = False,return_type='dict')
#results_256.boxplot(sym='k',showmeans = True,whis = [0,8],showfliers = False,return_type='dict')

pyplot.xlabel('Hidden node')
pyplot.ylabel('Error (m)')
pyplot.show()
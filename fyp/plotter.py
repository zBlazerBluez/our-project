import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv('logs/updated_rule_pretrained2 (copy).txt')
data2 = pd.read_csv('logs/updated_rule.txt')
#fig, ax = plt.subplot()
plt.plot([y*10000 for y in range(len(data.ix[:,1]))],[x for x in data.ix[:,1]], '-', label="continue")
plt.plot([y*20000 for y in range(len(data2.ix[:,1]))],[x for x in data2.ix[:,1]], '-',linestyle='dashed', label="from_scratch")
plt.legend()
plt.show()
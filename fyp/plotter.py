import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv('updated_rule_pretrained.txt')
plt.plot(range(len(data.ix[:,1])),[x for x in data.ix[:,1]], '-')

plt.show()
# -*- coding: utf-8 -*-
# @File       : tweet length statistic.py
# @Author     : Yuchen Chai
# @Date       : 2022/9/14 10:14
# @Description:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import clean_text

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Please adjust the following parameters before executing the code
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# Please specify the location you put the source data
DIR_INPUT = ''

# Please specify the location you store the output data
DIR_OUTPUT = ''


def split_count(x):
    return len(x.split(" "))


# Get statistic
dta = pd.read_csv(os.path.join(DIR_INPUT, "2021_sample tweets.csv"))
dta = clean_text(dta)
dta['text_len'] = dta['text'].apply(lambda x: split_count(x))
dta_statistic = dta.groupby(['text_len'])['lang'].count().reset_index()

# Draw cumulative figure
dta.columns = ['text length', 'n']
dta['cumsum'] = dta['n'].cumsum()
dta['ratio'] = dta['cumsum'] / max(dta['cumsum'])


plt.figure(figsize=(12,8))
plt.plot(dta['text length'], dta['ratio'], color="red")
plt.vlines(x=32, ymin=0, ymax=1, label="Text length = 32 words", color='green', linestyles="dashed")
plt.vlines(x=52, ymin=0, ymax=1, label="Text length = 52 words", color='blue', linestyles="dashed")
plt.hlines(y=0.9, xmin=0, xmax=120, label="Cumulative sum = 0.9", color="green", linestyles="dashed")
plt.hlines(y=0.99, xmin=0, xmax=120, label="Cumulative sum = 0.99", color="blue", linestyles="dashed")
plt.fill_between(dta['text length'], dta['ratio'], color="#ffbfbf")
plt.xlabel("Text length", fontsize=16)
plt.xticks(rotation=0)
plt.ylabel("Cumulative percentage of posts", fontsize=16)
sns.set_style("ticks")
sns.despine(offset=10)
plt.legend(loc=4)
plt.show()

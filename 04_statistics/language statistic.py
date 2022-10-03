# -*- coding: utf-8 -*-
# @File       : language statistic.py
# @Author     : Yuchen Chai
# @Date       : 2022/9/14 10:20
# @Description:


import os
import pandas as pd

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Please adjust the following parameters before executing the code
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# Please specify the location you put the source data
DIR_INPUT = ''

# Please specify the location you store the output data
DIR_OUTPUT = ''


files = os.listdir(DIR_INPUT)
ret = None
for ind, file in enumerate(files):
    try:
        file_path = os.path.join(DIR_INPUT, file)
        print(f"{ind}-{file_path}")
        df = pd.read_csv(file_path,
                         sep='\t',
                         encoding='UTF-8',
                         lineterminator='\n',
                         usecols=['tweet_lang'])
        df = df[df['tweet_lang'].str.len() <= 3]
        df_tweet_lang = df['tweet_lang'].value_counts().reset_index()
        df_tweet_lang.columns = ['tweet_lang', 'N']
        if ret is None:
            ret = df_tweet_lang
        else:
            ret = pd.concat([ret, df_tweet_lang])
    except:
        print('Error file')
        pass

ret = ret.groupby('tweet_lang')['N'].sum().reset_index()
ret = ret.sort_values(by=['N'], ascending=False)
ret.to_csv(os.path.join(DIR_OUTPUT,'language statistic.csv'))

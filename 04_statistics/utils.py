# -*- coding: utf-8 -*-
# @File       : utils.py
# @Author     : Yuchen Chai
# @Date       : 2022/9/14 9:07
# @Description:


import re
import html
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def standardize_username(string):
    string = re.sub(r'\@[A-z0-9\_]+', ' @user ', string) # replace user names by @user
    return string


def clean_for_content(string, lang):

    string = re.sub(r'\bhttps?\:\/\/[^\s]+', ' ', string) #remove websites

    string = html.unescape(string)

    string = deEmojify(string) # remove emojis

    # Classic replacements:
    string = re.sub(r'\&gt;', ' > ', string)
    string = re.sub(r'\&lt;', ' < ', string)
    string = re.sub(r'<\s?3', ' ❤ ', string)
    string = re.sub(r'\@\s', ' at ', string)

    string = standardize_username(string) # replace user names by @user

    if lang=='en':
        string = re.sub(r'(\&(amp)?|amp;)', ' and ', string)
        string = re.sub(r'(\bw\/?\b)', ' with ', string)
        string = re.sub(r'\brn\b', ' right now ', string)

    string = re.sub(r'\s+', ' ', string).strip()

    return string


def clean_text(p_df):
    mdf = p_df.copy()

    print("Cleaning training data")
    mdf['lang'] = 'en'
    mdf['text'] = [clean_for_content(text, lang) for text, lang in tqdm(zip(mdf['text'], mdf['lang']), total=mdf.shape[0])]

    mdf = mdf[mdf['text']!=''].reset_index(drop=True)

    return mdf

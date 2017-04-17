# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import datetime
import pickle
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import string
import jieba
jieba.load_userdict("../jiebaDATA/usr_dict/zhaojinciku.txt")
jieba.load_userdict("../jiebaDATA/usr_dict/wangluociku.txt")
jieba.load_userdict("../jiebaDATA/usr_dict/user.txt")
import jieba.posseg
import jieba.analyse
#jieba.analyse.set_idf_path("../extra_dict/idf.txt.big")
jieba.analyse.set_stop_words("../jiebaDATA/usr_dict/sentiment_stopword.txt")

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

def getTitleDf(data, ticker, start, end):

# Date map
    raw_d2td = pd.read_csv('../DataShare/csvData/daysToTradingDays.csv', index_col=0)
    d2td = raw_d2td.loc[start:end]
    dateIndex = d2td[d2td['currDay']==d2td['tradingDay']].index

    titleDf = pd.DataFrame(columns=["indexing", "title"])
    for i in dateIndex:
        #print('getTitleDf -> processing Date '+str(i))
        currentTradingDate = i
        currentDates = d2td[d2td['tradingDay']==currentTradingDate]
        titleSet = np.array([])
        for k in currentDates.index:
            sel = data[data['pubTime']==currentDates['currDay'][k]][['pubTime', 'title']]
            sel = sel.dropna()
            titleSet = np.append(titleSet, sel['title'].values)
        temp = pd.DataFrame(columns=["indexing", "title"], index=range(len(titleSet)))
        for j in temp.index:
            temp.loc[j]['indexing'] = ticker+"_"+currentTradingDate+"_"+str(j)
            temp.loc[j]['title'] = titleSet[j]
        titleDf = titleDf.append(temp, ignore_index=True)

    return(titleDf)


def funcCleanText(s):

    identify = string.maketrans('', '')
    delEStr = string.punctuation+string.digits+string.letters  #ASCII 标点符号，空格，数字，字母

    if sys.version[0] == '2': s = s.decode('utf-8')
    s = s.translate(identify, delEStr) 
    if len(s)<=4: s=""

    return(s)


def cleanTitleDf(titleDf):

    titleSeries = titleDf['title']
    titleDf['title'] = titleSeries.apply(funcCleanText)
    mask = titleDf['title']!=""
    titleDf = titleDf[mask]
    titleDf = titleDf.reset_index()

    return(titleDf)


def funcFilterText(s, keywords):

    seg_list = jieba.lcut(s)

    nn = 0
    for word in seg_list:
        if word in keywords: nn = nn + 1
    if nn == 0: s=""

    return(s)


def filterTitleDf(titleDf, keywords):

    titleSeries = titleDf['title']
    titleDf['title'] = titleSeries.apply(funcFilterText, keywords=keywords)
    mask = titleDf['title']!=""
    titleDf = titleDf[mask]
    titleDf = titleDf.reset_index()

    return(titleDf)


def funcFilterText_cut(s, keywords):

    seg_list = jieba.lcut(s)

    nn = 0
    for word in seg_list:
        if word in keywords: nn = nn + 1
    if nn == 0: seg_list=[]

    return("/".join(seg_list))


def filterTitleDf_cut(titleDf, keywords):

    titleSeries = titleDf['title']
    titleDf['title'] = titleSeries.apply(funcFilterText_cut, keywords=keywords)
    mask = titleDf['title']!=""
    titleDf = titleDf[mask]
    titleDf = titleDf.reset_index()

    return(titleDf)

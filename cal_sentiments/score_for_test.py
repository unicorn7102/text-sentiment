# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import datetime
import string

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import textprocessing as tp
import guba
import sentimentscoring as ss

#############################################################################################

pd.options.display.width = 200
pd.options.display.max_rows = 200
np.set_printoptions(linewidth=200)

# Load keywords for filter
keywords = ss.load_keywords()

#--------------------------------------------------------------------------------------------

raw_data = pd.read_csv('./RES/sampleTest/sample2.txt', encoding='gbk')
data = raw_data.copy()

data = guba.filterTitleDf(data, keywords)
sentimentDf = ss.getSentimentDf2(data)

resDf = pd.DataFrame(np.nan, index=data.index, columns=['real','jp','jn','jieba','rj','snow','rs','js','snowTrans','title'])
resDf['real'] = data['real']
resDf['title'] = data['title']
resDf['jp'] = sentimentDf['posSentScore']
resDf['jn'] = sentimentDf['negSentScore']

resDf['jieba'] = 0
resDf['snow'] = 0
resDf['rj'] = 0
resDf['rs'] = 0
resDf['js'] = 0

pmask = (sentimentDf['posSentScore'] > sentimentDf['negSentScore'])
nmask = (sentimentDf['posSentScore'] < sentimentDf['negSentScore'])
resDf.loc[resDf.index[pmask], 'jieba'] =  1
resDf.loc[resDf.index[nmask], 'jieba'] = -1

resDf['snowTrans'] = ( sentimentDf['snowScore'] - 0.5 ) * 2.
pmask = (resDf['snowTrans']>  1e-3)
nmask = (resDf['snowTrans']< -1e-3)
resDf.loc[resDf.index[pmask], 'snow'] =  1
resDf.loc[resDf.index[nmask], 'snow'] = -1
resDf = resDf.drop(['snowTrans'], axis=1)

mask1 = resDf['real'] == resDf['jieba']
mask2 = resDf['real'] * resDf['jieba'] == -1
resDf.loc[resDf.index[mask1], 'rj'] =  1
resDf.loc[resDf.index[mask2], 'rj'] = -1
mask1 = resDf['real'] == resDf['snow']
mask2 = resDf['real'] * resDf['snow'] == -1
resDf.loc[resDf.index[mask1], 'rs'] =  1
resDf.loc[resDf.index[mask2], 'rs'] = -1
mask1 = resDf['snow'] == resDf['jieba']
mask2 = resDf['snow'] * resDf['jieba'] == -1
resDf.loc[resDf.index[mask1], 'js'] =  1
resDf.loc[resDf.index[mask2], 'js'] = -1

NN = float(len(resDf.index))
nrj = resDf[resDf['rj']== 1]['rj'].sum()
nrs = resDf[resDf['rs']== 1]['rs'].sum()
njs = resDf[resDf['js']== 1]['js'].sum()
mrj = -resDf[resDf['rj']==-1]['rj'].sum()
mrs = -resDf[resDf['rs']==-1]['rs'].sum()
mjs = -resDf[resDf['js']==-1]['js'].sum()

print(resDf[resDf['rj']!=1])
print(resDf[resDf['rj']==1])
#print(resDf)

print("Total Number: "+str(NN))
print("Jieba - indentical: "+str(nrj)+"("+str(nrj/NN)+") inverse: "+str(mrj)+"("+str(mrj/NN)+")")
print("Snow - indentical: "+str(nrs)+"("+str(nrs/NN)+") inverse: "+str(mrs)+"("+str(mrs/NN)+")")
print("Compare two methods - indentical: "+str(njs)+"("+str(njs/NN)+") inverse: "+str(mjs)+"("+str(mjs/NN)+")")

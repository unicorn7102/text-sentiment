# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import datetime
import pickle
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import string

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

import textprocessing as tp
import guba

from snownlp import SnowNLP
import jieba
jieba.load_userdict("../jiebaDATA/usr_dict/zhaojinciku.txt")
jieba.load_userdict("../jiebaDATA/usr_dict/wangluociku.txt")
jieba.load_userdict("../jiebaDATA/usr_dict/user.txt")
import jieba.posseg
import jieba.analyse
#jieba.analyse.set_idf_path("../extra_dict/idf.txt.big")
jieba.analyse.set_stop_words("../jiebaDATA/usr_dict/sentiment_stopword.txt")

#############################################################################################
# Sentiment dictionary analysis basic function
#--------------------------------------------------------------------------------------------
def load_keywords():

        keywords = tp.get_txt_data("../jiebaDATA/keywords/myKeyword.txt", "lines")

        return(keywords)


def load_sentiment_dictionary():

# Load sentiment dictionary
        posdict = tp.get_txt_data("../jiebaDATA/sentiment_dict/myPos.txt", "lines")
        negdict = tp.get_txt_data("../jiebaDATA/sentiment_dict/myNeg.txt", "lines")

# Load adverbs of degree dictionary
        mostdict = tp.get_txt_data('../jiebaDATA/degree_inverse_dict/most.txt', 'lines')
        verydict = tp.get_txt_data('../jiebaDATA/degree_inverse_dict/very.txt', 'lines')
        moredict = tp.get_txt_data('../jiebaDATA/degree_inverse_dict/more.txt', 'lines')
        ishdict = tp.get_txt_data('../jiebaDATA/degree_inverse_dict/ish.txt', 'lines')
        insufficientdict = tp.get_txt_data('../jiebaDATA/degree_inverse_dict/insufficiently.txt', 'lines')
        inversedict = tp.get_txt_data('../jiebaDATA/degree_inverse_dict/inverse.txt', 'lines')

        return (posdict, negdict, mostdict, verydict, moredict, ishdict, insufficientdict, inversedict)


# Function of matching adverbs of degree and set weights
def match(word, sentiment_value):

#       if word in mostdict:
#               sentiment_value *= 2.0
#       elif word in verydict:
#               sentiment_value *= 1.5
#       elif word in moredict:
#               sentiment_value *= 1.25
#       elif word in ishdict:
#               sentiment_value *= 0.5
#       elif word in insufficientdict:
#               sentiment_value *= 0.25
        if word in inversedict:
                sentiment_value *= -1

        return sentiment_value


# Function of transforming negative score to positive score
# Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]
def transform_to_positive_num(poscount, negcount):

        if poscount < 0 and negcount >= 0:
                neg_count = negcount - poscount
                pos_count = 0
        elif negcount < 0 and poscount >= 0:
                pos_count = poscount - negcount
                neg_count = 0
        elif poscount < 0 and negcount < 0:
                neg_count = -poscount
                pos_count = -negcount
        else:
                pos_count = poscount
                neg_count = negcount

        return [pos_count, neg_count]


def sumup_sentence_sentiment_score(score_list):

        score_array = np.array(score_list) # Change list to a numpy array
        Pos = np.sum(score_array[:,0]) # Compute positive score
        Neg = np.sum(score_array[:,1])

        return [Pos, Neg]


def funcJiebaSentiment(text):
  
        single_sentence_senti_score = []
        cuted_text = tp.cut_sentence(text)

        for sent in cuted_text:
                seg_sent = tp.segmentation(sent, 'list')
                #print(" ".join(seg_sent))
                i = 0 # word position counter
                s = 0 # sentiment word position
                poscount = 0 # count a positive word
                negcount = 0 # count a negative word

                for word in seg_sent:
                        if word in posdict:
                                posi = 1
                                for w in seg_sent[s:i]:
                                        posi = match(w, posi)
                                poscount = poscount + posi
                                s = i + 1

                        elif word in negdict:
                                negi = 1
                                for w in seg_sent[s:i]:
                                        negi = match(w, negi)
                                negcount = negcount + negi
                                s = i + 1

                        # Match "!" in the review, every "!" has a weight of +2
                        #elif word == "！".decode('utf-8') or word == "!".decode('utf-8'):
                        #       for w2 in seg_sent[::-1]:
                        #               if w2 in posdict:
                        #                       poscount += 2
                        #                       break
                        #               elif w2 in negdict:
                        #                       negcount += 2
                        #                       break                    
                        i += 1

                if sys.version[0] == '2':
                    question_markc = "？".decode('utf-8')
                    question_marke = "?".decode('utf-8')
                question_markc = "？"
                question_marke = "?"
                if seg_sent[-1] == question_markc or question_marke:
                        poscount = 0
                        negcount = 0
                single_sentence_senti_score.append(transform_to_positive_num(poscount, negcount))
        score = sumup_sentence_sentiment_score(single_sentence_senti_score)

        return(score)


def funcSnowSentiment(text):

        snowText = SnowNLP(text)
        score = snowText.sentiments

        return(score)


def getSentimentDf(titleDf, method="jieba", parallel=False):

        if method=="jieba":
                wordDf = pd.DataFrame(columns=["indexing", "posSentScore", "negSentScore", "title"])
                wordDf['indexing'] = titleDf['indexing']
                if parallel==True:
                        pool = ThreadPool(2)
                        try:
                                sents = pool.map(funcJiebaSentiment, titleDf['title'])
                                pool.close()
                                pool.join()
                        except KeyboardInterrupt:
                                pool.terminate()
                                pool.join()
                        wordDf['sentScore'] = sents
                else:
                        for i in wordDf.index:
                                sents = funcJiebaSentiment(titleDf.loc[i]['title'])
                                wordDf.loc[i]['title'] = titleDf.loc[i]['title']
                                wordDf.loc[i]['posSentScore'] = sents[0]
                                wordDf.loc[i]['negSentScore'] = sents[1]

        elif method=="snow":
                wordDf = pd.DataFrame(columns=["indexing", "sentScore", "title"])
                wordDf['indexing'] = titleDf['indexing']
                if parallel==True:
                        pool = ThreadPool(2)
                        try:
                                sents = pool.map(funcSnowSentiment, titleDf['title'])
                                pool.close()
                                pool.join()
                        except KeyboardInterrupt:
                                pool.terminate()
                                pool.join()
                        wordDf['sentScore'] = sents
                else:
                        for i in wordDf.index:
                                sents = funcSnowSentiment(titleDf.loc[i]['title'])
                                wordDf.loc[i]['title'] = titleDf.loc[i]['title']
                                wordDf.loc[i]['sentScore'] = round(sents, 5)

        return(wordDf)


def getSentimentDf2(titleDf):

        wordDf = pd.DataFrame(columns=["indexing", "posSentScore", "negSentScore", "snowScore", "title"])
        wordDf['indexing'] = titleDf['indexing']
        for i in wordDf.index:
                sents = funcJiebaSentiment(titleDf.loc[i, 'title'])
                snowsent = funcSnowSentiment(titleDf.loc[i, 'title'])
                wordDf.loc[i, 'title'] = titleDf.loc[i, 'title']
                wordDf.loc[i, 'posSentScore'] = sents[0]
                wordDf.loc[i, 'negSentScore'] = sents[1]
                wordDf.loc[i, 'snowScore'] = round(snowsent, 5)

        return(wordDf)


# Load sentiment dictionary
(posdict, negdict, mostdict, verydict, moredict, ishdict, insufficientdict, inversedict) = load_sentiment_dictionary()

#############################################################################################

if __name__ == "__main__":

        pd.options.display.width = 200
        pd.options.display.max_rows = 200
        np.set_printoptions(linewidth=200)

        t1 = datetime.datetime.now()

# Researching period
        start = '2012-11-01'
        end = '2012-12-31'

# Load keywords for filter
        keywords = load_keywords()
#--------------------------------------------------------------------------------------------

        stocklist = sys.argv[1]
        f = open(stocklist, 'r')

        for line in f.readlines():

                ticker = str(line).strip()
                print(ticker)

                raw_data = pd.read_csv('../MOD_DATA/'+ticker+'/histbars.csv', index_col=0, encoding='gbk')
                data = raw_data.copy()

                titleDf = guba.getTitleDf(data, ticker, start, end)
                titleDf = guba.filterTitleDf(titleDf, keywords)
                sentimentDf = getSentimentDf2(titleDf)
#               print(sentimentDf)

                sentimentDf.to_csv('./RES/sentScores/score'+ticker+'.csv', header=sentimentDf.columns)

        f.close()

        t2 = datetime.datetime.now()
        print ("CPU TIME: "+str((t2-t1).seconds)+" s")

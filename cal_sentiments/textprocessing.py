# -*- coding: utf-8 -*-


import xlrd
import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

import jieba
jieba.load_userdict("../jiebaDATA/usr_dict/zhaojinciku.txt")
jieba.load_userdict("../jiebaDATA/usr_dict/wangluociku.txt")
jieba.load_userdict("../jiebaDATA/usr_dict/user.txt")
import jieba.posseg
import jieba.analyse
#jieba.analyse.set_idf_path("../extra_dict/idf.txt.big")
jieba.analyse.set_stop_words("../jiebaDATA/usr_dict/sentiment_stopword.txt")


def get_excel_data(filepath, sheetnum, colnum, para):

    table = xlrd.open_workbook(filepath)
    sheet = table.sheets()[sheetnum-1]
    data = sheet.col_values(colnum-1)
    rownum = sheet.nrows
    if para == 'data':
        return data
    elif para == 'rownum':
        return rownum


def get_txt_data(filepath, para):

    if para == 'lines':
        txt_file1 = open(filepath, 'r')
        txt_tmp1 = txt_file1.readlines()
        txt_tmp2 = ''.join(txt_tmp1)
        if sys.version[0] == '2': txt_tmp2 = txt_tmp2.decode('utf8')
        txt_data1 = txt_tmp2.split('\n')
        txt_file1.close()
        return txt_data1
    elif para == 'line':
        txt_file2 = open(filepath, 'r')
        txt_tmp = txt_file2.readline()
        if sys.version[0] == '2': txt_tmp = txt_tmp.decode('utf8')
        txt_data2 = txt_tmp
        txt_file2.close()
        return txt_data2


def segmentation(sentence, para):

    if para == 'str':
        seg_list = jieba.cut(sentence)
        seg_result = ' '.join(seg_list)
        return seg_result
    elif para == 'list':
        seg_list = jieba.cut(sentence)
        seg_result = []
        for w in seg_list:
            seg_result.append(w)
        return seg_result


def segmentation_filter(sentence, para):

    stopwords = get_txt_data("../jiebaDATA/usr_dict/sentiment_stopword.txt", 'lines')

    if para == 'str':
        seg_list = jieba.cut(sentence)
        seg_list1 = [word for word in seg_list if word not in stopwords and word != ' ']
        seg_result = ' '.join(seg_list1)
        return seg_result
    elif para == 'list':
        seg_list = jieba.cut(sentence)
        seg_list1 = [word for word in seg_list if word not in stopwords and word != ' ']
        seg_result = []
        for w in seg_list1:
            seg_result.append(w)
        return seg_result


def postagger(sentence, para):

    if para == 'list':
        pos_data1 = jieba.posseg.cut(sentence)
        pos_list = []
        for w in pos_data1:
            pos_list.append((w.word, w.flag)) #make every word and tag as a tuple and add them to a list
        return pos_list
    elif para == 'str':
        pos_data2 = jieba.posseg.cut(sentence)
        pos_list2 = []
        for w2 in pos_data2:
            pos_list2.extend([w2.word.encode('utf8'), w2.flag])
        pos_str = ' '.join(pos_list2)
        return pos_str


def cut_sentence(words):

    #words = (words).decode('utf8')
    start = 0
    i = 0 #i is the position of words
    token = 'meaningless'
    sents = []
    punt_list = '[]:,.!?;~：，。！？；～… '
    if sys.version[0] == '2': punt_list = punt_list.decode('utf8')
    for word in words:
        if word not in punt_list:
            i += 1
            token = list(words[start:i+2]).pop()
            #print token
        elif word in punt_list and token in punt_list:
            i += 1
            token = list(words[start:i+2]).pop()
        else:
            sents.append(words[start:i+1])
            start = i+1
            i += 1
    if start < len(words):
        sents.append(words[start:])
    return sents


##########################################################

if __name__ == "__main__":

    sent = sys.argv[1] #"小名是中国人，不是韩国人。他喜欢看电视。就是的。"
    seg_str = segmentation(sent, 'str')
    print(seg_str)
    seg_list = segmentation(sent, 'list')
    for word in seg_list:
        print(word)

    print("*****************************")

    seg_str = segmentation_filter(sent, 'str')
    print(seg_str)
    seg_list = segmentation_filter(sent, 'list')
    for word in seg_list:
        print(word)

 # -*- coding: utf-8 -*-
import re
import copy
import jieba
import jieba.posseg as pseg
from itertools import islice
from pprint import pprint
import traceback
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.pipeline import Pipeline 
from sklearn.svm import LinearSVC
from sklearn import metrics  
from sklearn import cross_validation
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
jieba.load_userdict("Dict/Dict_jieba.txt")



def sort_dict(d): 
    backitems=[[v[1],v[0]] for v in d.items] 
    backitems.sort() 
    return [ backitems[i][1] for i in range(0,len(backitems))] 



def sort_by_value(d):
    result = sorted(d.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    return result



def get_dict_number_and_name(listfilename):
    fsock_in = open(listfilename, "r")
    dict_filename_all={}
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        EachLine_unicode=unicode(EachLine,"utf8")
        EachLine_unicode_list= EachLine_unicode.split(',')  
        flie_name_number=EachLine_unicode_list[0]
        flie_name=EachLine_unicode_list[1]
        dict_filename_all[flie_name_number]=flie_name
    fsock_in.close() 
    return dict_filename_all



def to_cut_sentence(sentence):
    try:
        words= jieba.cut(sentence)
    except:
        words=[]
    sentence_cut = ' '.join(words)
    return sentence_cut



def get_file_name(listfilename):
    fsock_in = open(listfilename, "r")
    filename_all=[]
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n')  
        flie_name=unicode(EachLine,"utf8")
        filename_all.append(flie_name)
    fsock_in.close() 
    return filename_all



def To_Combined_into_text(filepath,filepath_sub,outname,number,target_number,listfilename,outname_listname):
    try:
        fsock_out = open(filepath + outname, "a")
        fsock_out_listname = open(filepath + outname_listname, "a")
        filename_list = get_file_name(listfilename)
        for filename in filename_list:
            number +=1
            str_number= '%d' %number
            fsock_out.write('#-#编号:'+str_number +'\n')
            fsock_out.write('#-#类别:'+target_number +'\n')
            fsock_out.write('[#-#-#-#-#-#-#\n')
            inpath = filepath_sub + filename
            fsock_in = open(inpath, "r")
            for EachLine in fsock_in:
                unicode_EachLine_dirty=unicode(EachLine,"utf8").replace(' ', '')
                sentence_cut=to_cut_sentence(unicode_EachLine_dirty)
                fsock_out.write(sentence_cut)
            fsock_in.close()
            fsock_out.write('\n#-#-#-#-#-#-#]\n')
            fsock_out_listname.write(str_number + ','+ filename[0:-4] +'\n')
        fsock_out.close()
        fsock_out_listname.close()
    except Exception, e:
        f_error=open(filepath+"error.txt",'a') 
        fsock_out.close()
        fsock_out_listname.close()
        print 'str(Exception):\t', str(Exception)
        print 'str(e):\t\t', str(e)
        print 'repr(e):\t', repr(e)
        print 'e.message:\t', e.message
        print 'traceback.print_exc():'; traceback.print_exc()
        print 'traceback.format_exc():\n%s' % traceback.format_exc()
        traceback.print_exc(file=f_error)  
        f_error.flush()  
        f_error.close()



def To_get_data(filename_all):
    fsock_in = open(filename_all, "r")
    flag=0
    data = []
    data_text = ''
    string_data_start = unicode('[#-#-#-#-#-#-#',"utf8")
    string_data_end = unicode('#-#-#-#-#-#-#]',"utf8")
    number = []
    number_text = ''
    string_number = unicode('#-#编号',"utf8")
    target_number_list = []
    target_number = 0
    string_target = unicode('#-#类别',"utf8")
    target_names = ['病例观察','对照试验','个案报道']
    result_dict = dict() 
    for EachLine in fsock_in:
        EachLine_unicode=unicode(EachLine,"utf8")
        if(flag==0):
            if(string_number==EachLine_unicode[0:9]):
                number_text=EachLine_unicode[10:-1]
                number.append(number_text)
                flag=1
        elif(flag==1):
            if(string_target==EachLine_unicode[0:5]):
                target_number=int(EachLine_unicode[6:10])
                target_number_list.append(target_number)
                flag=2
        elif(flag==2):
            if(string_data_start==EachLine_unicode[0:14]):
                flag=3
        elif(flag==3):
            if(string_data_end==EachLine_unicode[0:14]):
                data.append(data_text)
                data_text = ''
                flag=0
            else:
                data_text +=EachLine_unicode
    target=np.array(target_number_list)
    fsock_in.close()
    result_dict['data']=data
    result_dict['number']=number
    result_dict['target']=target
    result_dict['target_names']=target_names
    return result_dict



def To_process_classification_text(filepath,outname_train,outname_listname_train,outname_test,outname_listname_test,filepath_sub_PAPER):
    number = 1000
    target_number = '1'
    filepath_sub_Case_observation = filepath + u'病例观察/'
    listfilename_Case_observation = filepath + unicode('病例观察.txt',"utf8")
    To_Combined_into_text(filepath,filepath_sub_Case_observation,outname_train,number,target_number,listfilename_Case_observation,outname_listname_train)
    number = 2000
    target_number = '2'
    filepath_sub_Controlled_trial = filepath + u'对照试验/'
    listfilename_Controlled_trial = filepath + unicode('对照试验.txt',"utf8")
    To_Combined_into_text(filepath,filepath_sub_Controlled_trial,outname_train,number,target_number,listfilename_Controlled_trial,outname_listname_train)
    number = 3000
    target_number = '3'
    filepath_sub_Case_report = filepath + u'个案报道/'
    listfilename_Case_report = filepath + unicode('个案报道.txt',"utf8")
    To_Combined_into_text(filepath,filepath_sub_Case_report,outname_train,number,target_number,listfilename_Case_report,outname_listname_train)
    number = 60000
    target_number = '6'
    listfilename = filepath + unicode('待分类TXT文献列表.txt',"utf8")
    To_Combined_into_text(filepath,filepath_sub_PAPER,outname_test,number,target_number,listfilename,outname_listname_test)
    
    

def To_Get_Classified_files(stopwords_path,filename_test_data_listname,filename_test_data_cut,filename_train_data_cut,file_out):
    stopwords=[word.decode("utf-8") for word in open(stopwords_path).read().split()]
    dict_number_and_name = get_dict_number_and_name(filename_test_data_listname)
    dict_test = To_get_data(filename_test_data_cut)
    dict_train = To_get_data(filename_train_data_cut)
    docs_test = dict_test['data']    
    text_clf_LinearSVC = Pipeline(
        [('vect', CountVectorizer(stop_words = stopwords)),
         ('tfidf', TfidfTransformer(norm='l2',smooth_idf=True, sublinear_tf=True)),
         ('clf', LinearSVC(penalty='l2',loss='squared_hinge',dual=True,max_iter=5000,class_weight='balanced'))])
    text_clf_LinearSVC = text_clf_LinearSVC.fit(dict_train['data'], dict_train['target'])
    predicted_of_SVM = text_clf_LinearSVC.predict(docs_test)
    score_of_SVM = np.mean(predicted_of_SVM == dict_test['target']) 
    number_of_test = dict_test['number']
    fsock_out = open(file_out, "w")
    fsock_out.write("编号,文献名,所属类别\n")
    len_list = len(number_of_test)
    i=0
    while i<len_list:
        string_number = number_of_test[i]
        string_name = dict_number_and_name[string_number]
        num_predicted = predicted_of_SVM[i]
        string_predicted = '%d' %num_predicted
        fsock_out.write(string_number + ',' + string_name + ','+ string_predicted +'\n')
        i += 1
    fsock_out.close()



def get_acupoint_vocabulary_dict():
    fsock_in = open("Dict/Dict_acupoint_vocabulary_summary.txt", "r")
    words_acupoint_vocabulary_all=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        key_acupoint_vocabulary=unicode(EachLine,"utf8")
        words_acupoint_vocabulary_all[key_acupoint_vocabulary] = 0
    fsock_in.close()
    return words_acupoint_vocabulary_all



def get_illness_dict():
    fsock_in = open("Dict/Dict_illness.txt", "r")
    words_illness_all=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        key_illness=unicode(EachLine,"utf8")
        words_illness_all[key_illness] = 0
    fsock_in.close()
    return words_illness_all



def get_acupoint_vocabulary_Duplicate_removal_dict():
    fsock_in = open("Dict/Dict_acupoint_vocabulary_Duplicate_removal.txt", "r")
    words_acupoint_vocabulary_all=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        key_acupoint_vocabulary=unicode(EachLine,"utf8")
        words_acupoint_vocabulary_all[key_acupoint_vocabulary] = 0
    fsock_in.close()
    return words_acupoint_vocabulary_all



def get_acupoint_vocabulary_confused_dict():
    fsock_in = open("Dict/Dict_confused.txt", "r")
    words_acupoint_vocabulary_all=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        key_acupoint_vocabulary=unicode(EachLine,"utf8")
        words_acupoint_vocabulary_all[key_acupoint_vocabulary] = 0
    fsock_in.close()
    return words_acupoint_vocabulary_all



def get_dict_data(i,j):
    dict_data={}
    while i<j+1:
        str_i= '%d' %i
        dict_data[str_i]=0
        i += 1
    return dict_data



def get_convert_acupoint_vocabulary_dict():
    fsock_in = open("Dict/Dict_convert.txt", "r")
    dict_convert_acupoint_vocabulary=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine=unicode(EachLine,"utf8")
        EachLine_unicode_list = unicode_EachLine.split(',')  
        string_name= EachLine_unicode_list[0] 
        string_another_name_all = EachLine_unicode_list[1]
        list_another_name = string_another_name_all.split('、')  
        for string_another_name in list_another_name:
            dict_convert_acupoint_vocabulary[string_another_name]=string_name 
    fsock_in.close()
    return dict_convert_acupoint_vocabulary



def get_convert_wrong_dict():
    fsock_in = open("Dict/Dict_convert_wrong_word.txt", "r")
    dict_convert_wrong=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine=unicode(EachLine,"utf8")
        EachLine_unicode_list = unicode_EachLine.split(',')  
        string_name= EachLine_unicode_list[0] 
        string_another_name_all = EachLine_unicode_list[1]  
        list_another_name = string_another_name_all.split('、')  
        for string_another_name in list_another_name:
            dict_convert_wrong[string_another_name]=string_name  
    fsock_in.close()
    return dict_convert_wrong



def get_convert_combination_dict():
    fsock_in = open("Dict/Dict_convert_combination.txt", "r")
    dict_convert_combination=dict()
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine=unicode(EachLine,"utf8")
        EachLine_unicode_list = unicode_EachLine.split(',')  
        string_name= EachLine_unicode_list[0]  
        string_another_name_all = EachLine_unicode_list[1]  
        dict_convert_combination[string_name]=string_another_name_all   
    fsock_in.close()
    return dict_convert_combination



def to_Duplicate_removal_acupoint_vocabulary(sentence,dict_convert_acupoint_vocabulary):
    if sentence == '':
        return sentence
    else:
        string_clearn = ''
        x_word_temp=''
        dict_result ={}
        list_acupoint_vocabulary = sentence.split('、')  
        for string_acupoint_vocabulary in list_acupoint_vocabulary:
            if string_acupoint_vocabulary !='':
                if dict_convert_acupoint_vocabulary.has_key(string_acupoint_vocabulary):
                    x_word_temp=dict_convert_acupoint_vocabulary[string_acupoint_vocabulary]
                else:
                    x_word_temp=string_acupoint_vocabulary    
                dict_result[x_word_temp]=0
        for key in dict_result:
            string_clearn = string_clearn + key + '穴、' 
        string_acupoint_judgment=string_clearn[-1:]
        if string_acupoint_judgment=='、':
            string_clearn = string_clearn[:-1]
        return string_clearn



def to_Duplicate_removal_illness(sentence):
    if sentence == '':
        return sentence
    else:
        string_clearn = ''
        dict_result ={}
        list_acupoint_vocabulary = sentence.split('、')  
        for string_illness in list_acupoint_vocabulary:
            if string_illness !='':
                dict_result[string_illness]=0
        for key in dict_result:
            string_clearn = string_clearn + key + '、' 
        string_judgment=string_clearn[-1:]
        if string_judgment=='、':
            string_clearn = string_clearn[:-1]
        return string_clearn



def get_file_name_and_classificated(listfilename):
    fsock_in = open(listfilename, "r")
    list_result = []
    for EachLine in islice(fsock_in, 1, None):
        EachLine= EachLine.strip('\n') 
        EachLine_unicode=unicode(EachLine,"utf8")
        EachLine_unicode_list= EachLine_unicode.split(',')  
        string_number=EachLine_unicode_list[0]
        string_title=EachLine_unicode_list[1]
        string_classification=EachLine_unicode_list[2]
        list_result_part =[string_number,string_title,string_classification]
        list_result.append(list_result_part)
    return list_result



def Splitter(text):
    text_re_fuhao = text.replace("。", " ").replace("，", " ")
    splitted_sentences = text_re_fuhao.split()
    return splitted_sentences



def is_not_references(my_sentence):
    string_my=unicode('',"utf8")
    p1 = re.compile(r'\[\d{1,2}\]')
    p2 = re.compile(r'\〔\d{1,2}\〕')
    p3 = re.compile(r'\【\d{1,2}\】')
    str_p3= p3.findall(my_sentence)
    str_p2= p2.findall(my_sentence)
    str_p1= p1.findall(my_sentence)
    if len(str_p1)==0:
        if len(str_p2)==0:
            if len(str_p3)>0:
                string_my=str_p3[0]
        else:
            string_my=str_p2[0]
    else:
        string_my=str_p1[0]
    if len(string_my)==0:
        return 1
    else:
        return 0
  
    
    
def get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,unicode_EachLine):
    result_dict={}
    cutResult = jieba.cut(unicode_EachLine)
    resultList = list(cutResult) 
    the_word = ''
    type_of_word = 'N'
    ex_word = ''
    for x_word in resultList:
        add_word = ex_word + x_word[0]
        if x_word in List_ABgroup:
            if x_word in List_Agroup:
                type_of_word = 'A'
                the_word = x_word
            elif x_word in List_Bgroup:
                type_of_word = 'B'
                the_word = x_word
        elif add_word in List_ABgroup:
            if add_word in List_Agroup:
                type_of_word = 'A'
                the_word = add_word
            elif add_word in List_Bgroup:
                type_of_word = 'B'
                the_word = add_word
        ex_word=x_word
    result_dict['type_of_word']=type_of_word
    result_dict['the_word']=the_word
    return result_dict
        


def find_p(my_sentence):
    string_my=unicode('',"utf8")
    p1 = re.compile(r'\d{1}')
    p2 = re.compile(r'\d{1}\.\d{1}')
    p3 = re.compile(r'\d{1}\.\d{1}.\d{1}')
    str_p3= p3.findall(my_sentence)
    str_p2= p2.findall(my_sentence)
    str_p1= p1.findall(my_sentence)
    if len(str_p3)==0:
        if len(str_p2)==0:
            if len(str_p1)>0: string_my=str_p1[0]
        else:
            string_my=str_p2[0]
    else:
        string_my=str_p3[0]
    return string_my

  

def judge_can_read_Controlled_trial(flag_begin,flag_num_str,temp_sentence_top,dict_data,dict_number):
    temp_flag_begin = flag_begin
    temp_flag_num_str = flag_num_str
    temp_sentence = temp_sentence_top
    judge_flag_num_str=flag_num_str
    flag_first_word=0
    result_judge_dict = dict()    
    if temp_flag_begin==0:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        ex_word = ''
        for x_word in resultList:
            if (x_word=='方法') or (x_word=='取穴') or (x_word=='选穴') or (x_word=='针剌治疗') or (x_word=='针刺治疗') or (x_word=='穴位选择'):
                if (ex_word!='统计学') and (ex_word!='统计'):
                    temp_flag_begin = 1
                    temp_sentence = temp_sentence[:3]
                    temp_sentence=temp_sentence.replace("I", "1").replace("l", "1")
                    temp_flag_num_str=unicode('',"utf8")
                    words = pseg.cut(temp_sentence)
                    for word, flag in words:
                        if flag=='m':
                            temp_str=find_p(word)
                            if temp_str<>"":
                                temp_flag_num_str=temp_str                 
            ex_word = x_word
    elif temp_flag_begin == 1:
        temp_sentence = temp_sentence[:4]
        temp_sentence=temp_sentence.replace("I", "1").replace("l", "1")
        temp_sentence_one=temp_sentence[:1]
        if dict_number.has_key(temp_sentence_one):
            if dict_data.has_key(temp_sentence):
                temp_flag_begin = 1
            else:
                flag_is_not_references = is_not_references(temp_sentence)
                if flag_is_not_references:
                    words = pseg.cut(temp_sentence)
                    for word, flag in words:
                        flag_first_word += 1
                        if flag_first_word ==1:
                            if flag=='m':
                                temp_str=find_p(word)
                                if temp_str<>"":
                                    judge_flag_num_str=temp_str  
                    if judge_flag_num_str.find(temp_flag_num_str) == -1:
                        temp_flag_begin=2
    result_judge_dict['flag_begin']=temp_flag_begin
    result_judge_dict['flag_num_str']=temp_flag_num_str
    result_judge_dict['judge_flag_num_str']=judge_flag_num_str
    return result_judge_dict



def judge_can_read_Controlled_trial_normal(flag_begin,temp_sentence_top):
    temp_flag_begin = flag_begin
    temp_sentence = temp_sentence_top
    result_judge_dict = dict()    
    if temp_flag_begin==0:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        ex_word = ''
        for x_word in resultList:
            if (x_word=='方法') or (x_word=='取穴') or (x_word=='选穴') or (x_word=='针剌治疗') or (x_word=='针刺治疗') or (x_word=='穴位选择'):
                if (ex_word!='统计学') and (ex_word!='统计'):
                    temp_flag_begin = 1
            ex_word = x_word
    elif temp_flag_begin == 1:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        ex_word = ''
        for x_word in resultList:
            if (x_word=='治疗效果') or (x_word=='效果') or (x_word=='结果') or (x_word=='疗效') or (x_word=='参考文献') or (x_word=='讨论') or (x_word=='收稿日期'):
               temp_flag_begin = 2
    result_judge_dict['flag_begin']=temp_flag_begin
    return result_judge_dict



def judge_can_read_Case_observation(flag_begin,temp_sentence):
    temp_flag_begin = flag_begin
    result_judge_dict = dict()    
    if temp_flag_begin==0:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        ex_word = ''
        for x_word in resultList:
            if (x_word=='方法') or (x_word=='取穴') or (x_word=='选穴') or (x_word=='针剌治疗') or (x_word=='针刺治疗') or (x_word=='穴位选择'):
                if (ex_word!='统计学') and (ex_word!='统计'):
                    temp_flag_begin = 1
            ex_word = x_word
    elif temp_flag_begin == 1:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        ex_word = ''
        for x_word in resultList:
            if (x_word=='治疗效果') or (x_word=='效果') or (x_word=='结果') or (x_word=='体会') or (x_word=='疗效') or (x_word=='参考文献') or (x_word=='讨论') or (x_word=='收稿日期'):
               temp_flag_begin = 2
    result_judge_dict['flag_begin']=temp_flag_begin
    return result_judge_dict



def judge_can_read_Case_report(flag_begin,temp_sentence_long,string_title):
    temp_string_title = string_title[0:-4]
    temp_sentence = temp_sentence_long[0:5]
    temp_flag_begin = flag_begin
    result_judge_dict = dict()    
    if temp_flag_begin==0:
        if temp_sentence_long.find(temp_string_title) != -1:
            temp_flag_begin = 1
    elif temp_flag_begin == 1:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        for x_word in resultList:
            if (x_word=='治疗效果') or (x_word=='效果') or (x_word=='结果') or (x_word=='体会') or (x_word=='疗效')or (x_word=='收稿') or (x_word=='发稿') or (x_word=='参考文献') or (x_word=='讨论') or (x_word=='收稿日期'):
               temp_flag_begin = 2
    result_judge_dict['flag_begin']=temp_flag_begin
    return result_judge_dict



def judge_can_read_Case_report_normal(flag_begin,temp_sentence_long,string_title,string_illness):
    temp_string_title = string_illness
    temp_sentence = temp_sentence_long[0:5]
    temp_flag_begin = flag_begin
    result_judge_dict = dict()    
    if temp_flag_begin==0:
        if temp_sentence_long.find(temp_string_title) != -1:
            temp_flag_begin = 1
    elif temp_flag_begin == 1:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        for x_word in resultList:
            if (x_word=='治疗效果') or (x_word=='效果') or (x_word=='结果') or (x_word=='体会') or (x_word=='疗效')or (x_word=='收稿') or (x_word=='发稿') or (x_word=='参考文献') or (x_word=='讨论') or (x_word=='收稿日期'):
               temp_flag_begin = 2
    result_judge_dict['flag_begin']=temp_flag_begin
    return result_judge_dict



def judge_can_read_Case_report_normal2(flag_begin,temp_sentence_long,string_title):
    temp_string_title = string_title[-2:]
    temp_sentence = temp_sentence_long[0:5]
    temp_flag_begin = flag_begin
    result_judge_dict = dict()    
    if temp_flag_begin==0:
        if temp_sentence_long.find(temp_string_title) != -1:
            temp_flag_begin = 1
    elif temp_flag_begin == 1:
        cutResult = jieba.cut(temp_sentence)
        resultList = list(cutResult) 
        for x_word in resultList:
            if (x_word=='治疗效果') or (x_word=='效果') or (x_word=='结果') or (x_word=='体会') or (x_word=='疗效')or (x_word=='收稿') or (x_word=='发稿') or (x_word=='参考文献') or (x_word=='讨论') or (x_word=='收稿日期'):
               temp_flag_begin = 2
    result_judge_dict['flag_begin']=temp_flag_begin
    return result_judge_dict



def judge_which_feature_exists(inpath,string_title):
    dict_number = {u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    dict_number_chinese = {u'一、':0,u'二、':0,u'三、':0,u'四、':0,u'五、':0}
    dict_an = {u'按：':0,u'按:':0}
    dict_li = {u'一':0,u'二':0,u'三':0,u'四':0,u'五':0,u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    flag_number = 0 
    flag_an = 0 
    flag_li = 0 
    flag_fangfa = 0  
    flag_begin = 0 
    fsock_in = open(inpath, "r")
    result_dict = {}
    for EachLine in fsock_in:
        temp_sentence_top = ''
        temp_sentence_top_long = ''
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        len_sentence=len(unicode_EachLine)
        if len_sentence > 40:
            temp_sentence_top_long=unicode_EachLine[:40]
        else:
            temp_sentence_top_long=unicode_EachLine
        result_judge_dict = judge_can_read_Case_report_normal2(flag_begin,temp_sentence_top_long,string_title)
        flag_begin = result_judge_dict['flag_begin']
        if flag_begin==1:
            len_sentence=len(unicode_EachLine)
            if len_sentence > 10:
                temp_sentence_top=unicode_EachLine[:10]
            else:
                temp_sentence_top=unicode_EachLine
            if len(temp_sentence_top)>0:
                temp_sentence_top=temp_sentence_top.replace("I", "1").replace("l", "1")
                word_first = temp_sentence_top[0]
                word_two = temp_sentence_top[0:2]
                if dict_number.has_key(word_first):
                    word_to_judge=temp_sentence_top[1:3]
                    if len(unicode_EachLine)<10:
                        if (word_to_judge!=u'案例') and (word_to_judge!=u'体会') and (word_to_judge!=u'临床') and (word_to_judge!=u'总结'):
                            flag_number += 1
                if dict_number_chinese.has_key(word_two):    
                    word_to_judge=temp_sentence_top[2:4]
                    if len(unicode_EachLine)<10:
                        if (word_to_judge!=u'案例') and (word_to_judge!=u'体会') and (word_to_judge!=u'临床') and (word_to_judge!=u'总结'):
                            flag_number += 1
                if dict_an.has_key(word_two):
                    flag_an += 1        
                if word_first==u'例':
                    word_to_judge=temp_sentence_top[1:2]
                    if dict_li.has_key(word_to_judge):
                        flag_li += 1
                cutResult = jieba.cut(temp_sentence_top)
                resultList = list(cutResult) 
                ex_word = ''
                for x_word in resultList:
                    if (x_word=='方法') or (x_word=='取穴') or (x_word=='选穴'):
                        if (ex_word!='统计学') and (ex_word!='统计'):
                            flag_fangfa = 1
    if flag_number>=2:
        flag_number=1
    else:
        flag_number=0
    if flag_an>0:
        flag_an=1
    else:
        flag_an=0
    if flag_li>0:
        flag_li=1
    else:
        flag_li=0
    result_dict['flag_number']=flag_number
    result_dict['flag_an']=flag_an
    result_dict['flag_li']=flag_li
    result_dict['flag_fangfa']=flag_fangfa
    return result_dict



def get_illness_in_sentence(illness_dict,unicode_EachLine):
    flag_number = 0
    str_illness_infile=''
    cutResult = jieba.cut(unicode_EachLine)
    resultList = list(cutResult) 
    for x_word in resultList:
        if x_word in illness_dict:
            flag_number += 1
            str_illness_infile += x_word + '、' 
    if flag_number == 0:
        str_illness_clean = str_illness_infile
    else:
        str_illness_clean= str_illness_infile[:-1]
    return str_illness_clean



def get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused):
    str_acupoint_vocabulary_infile=''
    unicode_xue_zi = unicode('穴',"utf8")
    cutResult = jieba.cut(unicode_EachLine)
    resultList = list(cutResult)  
    dict_punctuation = {u'null':0,u',':0,u'.':0,u':':0,u'，':0,u'。':0,u'、':0,u'；':0,u';':0}
    word_one = u'null'
    word_two = u'null'
    word_three = u'null'
    for x_word in resultList:
        word_one=word_two
        word_two=word_three
        word_three = x_word
        if dict_acupoint_vocabulary_summary.has_key(word_two):
            temp_x_word = word_two
            if dict_acupoint_vocabulary_confused.has_key(temp_x_word):
                if (dict_punctuation.has_key(word_one)) or (dict_punctuation.has_key(word_three)):
                    if dict_convert_wrong.has_key(temp_x_word):
                        temp_x_word = dict_convert_wrong[temp_x_word]
                    if dict_convert_combination.has_key(temp_x_word):
                        temp_x_word = dict_convert_combination[temp_x_word]
                    x_word_clean=temp_x_word.replace(unicode_xue_zi, '')
                    str_acupoint_vocabulary_infile += x_word_clean + '、'          
            else:
                if dict_convert_wrong.has_key(temp_x_word):
                    temp_x_word = dict_convert_wrong[temp_x_word]
                if dict_convert_combination.has_key(temp_x_word):
                    temp_x_word = dict_convert_combination[temp_x_word]
                x_word_clean=temp_x_word.replace(unicode_xue_zi, '')
                str_acupoint_vocabulary_infile += x_word_clean + '、' 
    word_one=word_two
    word_two=word_three
    word_three = u'null'
    if dict_acupoint_vocabulary_summary.has_key(word_two):
        temp_x_word = word_two
        if dict_convert_wrong.has_key(temp_x_word):
            temp_x_word = dict_convert_wrong[temp_x_word]
        if dict_convert_combination.has_key(temp_x_word):
            temp_x_word = dict_convert_combination[temp_x_word]
        x_word_clean=temp_x_word.replace(unicode_xue_zi, '')
        str_acupoint_vocabulary_infile += x_word_clean + '、' 
    string_acupoint_judgment=str_acupoint_vocabulary_infile[-1:]   
    if string_acupoint_judgment=='、':
        str_acupoint_vocabulary_infile = str_acupoint_vocabulary_infile[:-1]
    return str_acupoint_vocabulary_infile    



def information_extraction_Controlled_trial(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,dict_data,dict_number):
    result_dict = {}
    List_Agroup = [unicode("A组"),unicode("治疗组"),unicode("针刺组"),unicode("针灸组"),unicode("实验组"),unicode("观察组"),unicode("研究组"),unicode("醒脑组"),unicode("综合组"),unicode("研宄组"),unicode("银针组"),unicode("毫针组"),unicode("针刺治疗"),unicode("电针治疗")]
    List_Bgroup = [unicode("B组"),unicode("对照组"),unicode("药物组"),unicode("参照组"),unicode("常规组"),unicode("中药组"),unicode("对照A组"),unicode("对照B组"),unicode("对照1组"),unicode("对照2组"),unicode("基础治疗"),unicode("推拿疗法"),unicode("推拿法"),unicode("走罐治疗"),unicode("拔罐法"),unicode("走罐法")]
    List_ABgroup = copy.deepcopy(List_Agroup)
    List_ABgroup.extend(List_Bgroup)
    flag_A_show = 0
    flag_B_show = 0
    flag_AB_show = 0 
    flag_begin = 0  
    flag_write_one = 0
    flag_num_str = u''
    string_acupoint_A = ''
    string_acupoint_B = ''
    word_of_Agroup = ''
    word_of_Bgroup = ''
    result_illness_abstract = ''
    result_illness = ''
    list_result_acupoint=[]
    fsock_in = open(inpath, "r")
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_top = ''
        temp_sentence_top_short= ''
        len_sentence=len(unicode_EachLine)
        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
        if temp_illness!= '':
            result_illness += temp_illness + '、' 
        if len_sentence > 5:
            temp_sentence_top_short=unicode_EachLine[:5]
        else:
            temp_sentence_top_short=unicode_EachLine           
        cutResult = jieba.cut(temp_sentence_top_short)
        resultList = list(cutResult) 
        for x_word in resultList:
            if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                if temp_illness!= '':
                    result_illness_abstract += temp_illness + '、' 
        if len_sentence > 15:
            temp_sentence_top=unicode_EachLine[:15]
        else:
            temp_sentence_top=unicode_EachLine
        result_judge_dict = judge_can_read_Controlled_trial(flag_begin,flag_num_str,temp_sentence_top,dict_data,dict_number)
        flag_begin = result_judge_dict['flag_begin']
        flag_num_str = result_judge_dict['flag_num_str']
        if flag_begin==1:
            splitted_sentences = Splitter(unicode_EachLine)
            for sentences in splitted_sentences:
                len_sentence=len(sentences)
                if len_sentence > 15:
                    temp_sentence_top=sentences[:15]
                else:
                    temp_sentence_top=sentences            
                if flag_AB_show == 0:
                   dict_category = get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,temp_sentence_top)
                   type_of_word = dict_category['type_of_word']
                   the_word = dict_category['the_word']
                   if type_of_word=='A':
                       flag_AB_show = 1
                       flag_A_show = 1
                       word_of_Agroup = the_word + '(A组)'
                       temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                       if temp_acupoint_vocabulary !='':
                           string_acupoint_A += temp_acupoint_vocabulary + '、'
                       pprint('flag_AB_show == 0,type_of_word==A,and word_of_Agroup = '+word_of_Agroup+'temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                   elif type_of_word=='B':
                       flag_AB_show = 1
                       flag_B_show = 1  
                       word_of_Bgroup = the_word + '(B组)'
                       temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                       if temp_acupoint_vocabulary !='':
                           string_acupoint_B += temp_acupoint_vocabulary + '、'  
                       pprint('flag_AB_show == 0,type_of_word==A,and word_of_Bgroup = '+word_of_Bgroup+'temp_sentence_top='+temp_sentence_top +'  temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)                 
                elif flag_AB_show == 1:
                   dict_category = get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,temp_sentence_top)
                   type_of_word = dict_category['type_of_word']
                   the_word = dict_category['the_word']
                   if flag_A_show == 1:
                       if type_of_word == 'N' or type_of_word == 'A':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、'
                           pprint('flag_AB_show == 1,type_of_word==N或者A,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'B':
                           flag_AB_show =2
                           flag_A_show =0
                           flag_B_show =1
                           word_of_Bgroup = the_word + '(B组)'
                           if string_acupoint_A != '':
                               string_acupoint_judgment=string_acupoint_A[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_A = string_acupoint_A[:-1]
                               list_result_acupoint.append([word_of_Agroup,string_acupoint_A])
                           else:
                               list_result_acupoint.append([word_of_Agroup,'null'])
                           string_acupoint_A = ''
                           word_of_Agroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'   
                           pprint('flag_AB_show == 1,type_of_word==B,and word_of_Agroup = '+word_of_Agroup+' temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                           
                   elif flag_B_show== 1:
                       if type_of_word == 'N' or type_of_word == 'B':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'  
                           pprint('flag_AB_show == 1,type_of_word==N或者B,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'A':
                           flag_AB_show =2
                           flag_A_show =1
                           flag_B_show =0
                           word_of_Agroup = the_word + '(A组)'
                           if string_acupoint_B != '':
                               string_acupoint_judgment=string_acupoint_B[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_B = string_acupoint_B[:-1]
                               list_result_acupoint.append([word_of_Bgroup,string_acupoint_B])
                           else:
                               list_result_acupoint.append([word_of_Bgroup,'null'])
                           string_acupoint_B = ''
                           word_of_Bgroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           pprint('flag_AB_show == 1,type_of_word==A,and  word_of_Agroup = '+word_of_Agroup+'temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、' 
                elif flag_AB_show == 2:
                   dict_category = get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,temp_sentence_top)
                   type_of_word = dict_category['type_of_word']
                   the_word = dict_category['the_word']
                   if flag_A_show == 1:
                       if type_of_word == 'N' or type_of_word == 'A':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、'
                           pprint('flag_AB_show == 2,type_of_word==N或者A,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'B':
                           flag_AB_show =1
                           flag_A_show =0
                           flag_B_show =1
                           word_of_Bgroup = the_word + '(B组)'
                           if string_acupoint_A != '':
                               string_acupoint_judgment=string_acupoint_A[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_A = string_acupoint_A[:-1]
                               list_result_acupoint.append([word_of_Agroup,string_acupoint_A])
                           else:
                               list_result_acupoint.append([word_of_Agroup,'null'])
                           string_acupoint_A = ''
                           word_of_Agroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'   
                           pprint('flag_AB_show == 2,type_of_word==B,and word_of_Agroup = '+word_of_Agroup+' temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                   elif flag_B_show == 1:
                       if type_of_word == 'N' or type_of_word == 'B':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'  
                           pprint('flag_AB_show == 2,type_of_word==N或者B,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'A':
                           flag_AB_show =1
                           flag_A_show =1
                           flag_B_show =0
                           word_of_Agroup = the_word + '(A组)'
                           if string_acupoint_B != '':
                               string_acupoint_judgment=string_acupoint_B[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_B = string_acupoint_B[:-1]
                               list_result_acupoint.append([word_of_Bgroup,string_acupoint_B])
                           else:
                               list_result_acupoint.append([word_of_Bgroup,'null'])
                           string_acupoint_B = ''
                           word_of_Bgroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、' 
                           pprint('flag_AB_show == 1,type_of_word==A,and  word_of_Agroup = '+word_of_Agroup+'temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
        elif flag_begin==2:  
            flag_write_one +=1
            if flag_write_one==1:
                if flag_A_show == 1:
                    if string_acupoint_A != '':
                        string_acupoint_judgment=string_acupoint_A[-1:]
                        if string_acupoint_judgment=='、':
                            string_acupoint_A = string_acupoint_A[:-1]
                            list_result_acupoint.append([word_of_Agroup,string_acupoint_A])
                    else:
                        list_result_acupoint.append([word_of_Agroup,'null'])
                elif flag_B_show == 1:
                    if string_acupoint_B != '':
                        string_acupoint_judgment=string_acupoint_B[-1:]
                        if string_acupoint_judgment=='、':
                            string_acupoint_B = string_acupoint_B[:-1]
                            list_result_acupoint.append([word_of_Bgroup,string_acupoint_B])
                    else:
                        list_result_acupoint.append([word_of_Bgroup,'null'])                  
    fsock_in.close()
    if result_illness_abstract != '':
        string_acupoint_judgment=result_illness_abstract[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness_abstract[:-1]
    else:
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness[:-1]
    result_dict['list_result_acupoint']=list_result_acupoint
    result_dict['result_illness']=result_illness
    return result_dict



def information_extraction_Controlled_trial_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused):
    result_dict = {}
    List_Agroup = [unicode("A组"),unicode("治疗组"),unicode("针刺组"),unicode("针灸组"),unicode("实验组"),unicode("观察组"),unicode("研究组"),unicode("醒脑组"),unicode("综合组"),unicode("研宄组"),unicode("银针组"),unicode("毫针组"),unicode("针刺治疗"),unicode("电针治疗")]
    List_Bgroup = [unicode("B组"),unicode("对照组"),unicode("药物组"),unicode("参照组"),unicode("常规组"),unicode("中药组"),unicode("对照A组"),unicode("对照B组"),unicode("对照1组"),unicode("对照2组"),unicode("基础治疗"),unicode("推拿疗法"),unicode("推拿法"),unicode("走罐治疗"),unicode("拔罐法"),unicode("走罐法")]
    List_ABgroup = copy.deepcopy(List_Agroup)
    List_ABgroup.extend(List_Bgroup)
    flag_A_show = 0
    flag_B_show = 0
    flag_AB_show = 0 
    flag_begin = 0  
    flag_write_one = 0
    string_acupoint_A = ''
    string_acupoint_B = ''
    word_of_Agroup = ''
    word_of_Bgroup = ''
    result_illness_abstract = ''
    result_illness = ''
    list_result_acupoint=[]
    fsock_in = open(inpath, "r")
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_top = ''
        len_sentence=len(unicode_EachLine)
        if len_sentence > 15:
            temp_sentence_top=unicode_EachLine[:15]
        else:
            temp_sentence_top=unicode_EachLine
        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
        if temp_illness!= '':
            result_illness += temp_illness + '、' 
        if len_sentence > 5:
            temp_sentence_top_short=unicode_EachLine[:5]
        else:
            temp_sentence_top_short=unicode_EachLine           
        cutResult = jieba.cut(temp_sentence_top_short)
        resultList = list(cutResult) 
        for x_word in resultList:
            if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                if temp_illness!= '':
                    result_illness_abstract += temp_illness + '、' 
        result_judge_dict = judge_can_read_Controlled_trial_normal(flag_begin,temp_sentence_top)
        flag_begin = result_judge_dict['flag_begin']
        if flag_begin==1:
            splitted_sentences = Splitter(unicode_EachLine)
            for sentences in splitted_sentences:
                len_sentence=len(sentences)
                if len_sentence > 15:
                    temp_sentence_top=sentences[:15]
                else:
                    temp_sentence_top=sentences            
                if flag_AB_show == 0:
                   dict_category = get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,temp_sentence_top)
                   type_of_word = dict_category['type_of_word']
                   the_word = dict_category['the_word']
                   if type_of_word=='A':
                       flag_AB_show = 1
                       flag_A_show = 1
                       word_of_Agroup = the_word + '(A组)'
                       temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                       if temp_acupoint_vocabulary !='':
                           string_acupoint_A += temp_acupoint_vocabulary + '、'
                       pprint('flag_AB_show == 0,type_of_word==A,and word_of_Agroup = '+word_of_Agroup+'temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                   elif type_of_word=='B':
                       flag_AB_show = 1
                       flag_B_show = 1  
                       word_of_Bgroup = the_word + '(B组)'
                       temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                       if temp_acupoint_vocabulary !='':
                           string_acupoint_B += temp_acupoint_vocabulary + '、'  
                       pprint('flag_AB_show == 0,type_of_word==A,and word_of_Bgroup = '+word_of_Bgroup+'temp_sentence_top='+temp_sentence_top +'  temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)                 
                elif flag_AB_show == 1:
                   dict_category = get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,temp_sentence_top)
                   type_of_word = dict_category['type_of_word']
                   the_word = dict_category['the_word']
                   if flag_A_show == 1:
                       if type_of_word == 'N' or type_of_word == 'A':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、'
                           pprint('flag_AB_show == 1,type_of_word==N或者A,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'B':
                           flag_AB_show =2
                           flag_A_show =0
                           flag_B_show =1
                           word_of_Bgroup = the_word + '(B组)'
                           if string_acupoint_A != '':
                               string_acupoint_judgment=string_acupoint_A[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_A = string_acupoint_A[:-1]
                               list_result_acupoint.append([word_of_Agroup,string_acupoint_A])
                           else:
                               list_result_acupoint.append([word_of_Agroup,'null'])
                           string_acupoint_A = ''
                           word_of_Agroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'   
                           pprint('flag_AB_show == 1,type_of_word==B,and word_of_Agroup = '+word_of_Agroup+' temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                           
                   elif flag_B_show== 1:
                       if type_of_word == 'N' or type_of_word == 'B':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'  
                           pprint('flag_AB_show == 1,type_of_word==N或者B,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'A':
                           flag_AB_show =2
                           flag_A_show =1
                           flag_B_show =0
                           word_of_Agroup = the_word + '(A组)'
                           if string_acupoint_B != '':
                               string_acupoint_judgment=string_acupoint_B[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_B = string_acupoint_B[:-1]
                               list_result_acupoint.append([word_of_Bgroup,string_acupoint_B])
                           else:
                               list_result_acupoint.append([word_of_Bgroup,'null'])
                           string_acupoint_B = ''
                           word_of_Bgroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           pprint('flag_AB_show == 1,type_of_word==A,and  word_of_Agroup = '+word_of_Agroup+'temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、' 
                elif flag_AB_show == 2:
                   dict_category = get_category_and_vocabulary_AB(List_Agroup,List_Bgroup,List_ABgroup,temp_sentence_top)
                   type_of_word = dict_category['type_of_word']
                   the_word = dict_category['the_word']
                   if flag_A_show == 1:
                       if type_of_word == 'N' or type_of_word == 'A':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、'
                           pprint('flag_AB_show == 2,type_of_word==N或者A,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'B':
                           flag_AB_show =1
                           flag_A_show =0
                           flag_B_show =1
                           word_of_Bgroup = the_word + '(B组)'
                           if string_acupoint_A != '':
                               string_acupoint_judgment=string_acupoint_A[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_A = string_acupoint_A[:-1]
                               list_result_acupoint.append([word_of_Agroup,string_acupoint_A])
                           else:
                               list_result_acupoint.append([word_of_Agroup,'null'])
                           string_acupoint_A = ''
                           word_of_Agroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'   
                           pprint('flag_AB_show == 2,type_of_word==B,and word_of_Agroup = '+word_of_Agroup+' temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                   elif flag_B_show == 1:
                       if type_of_word == 'N' or type_of_word == 'B':
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_B += temp_acupoint_vocabulary + '、'  
                           pprint('flag_AB_show == 2,type_of_word==N或者B,and temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
                       elif type_of_word == 'A':
                           flag_AB_show =1
                           flag_A_show =1
                           flag_B_show =0
                           word_of_Agroup = the_word + '(A组)'
                           if string_acupoint_B != '':
                               string_acupoint_judgment=string_acupoint_B[-1:]
                               if string_acupoint_judgment=='、':
                                   string_acupoint_B = string_acupoint_B[:-1]
                               list_result_acupoint.append([word_of_Bgroup,string_acupoint_B])
                           else:
                               list_result_acupoint.append([word_of_Bgroup,'null'])
                           string_acupoint_B = ''
                           word_of_Bgroup = ''
                           temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,sentences,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                           if temp_acupoint_vocabulary !='':
                               string_acupoint_A += temp_acupoint_vocabulary + '、' 
                           pprint('flag_AB_show == 1,type_of_word==A,and  word_of_Agroup = '+word_of_Agroup+'temp_sentence_top='+temp_sentence_top +' temp_acupoint_vocabulary=' + temp_acupoint_vocabulary)
        elif flag_begin==2:
            flag_write_one +=1
            if flag_write_one==1:
                if flag_A_show == 1:
                    if string_acupoint_A != '':
                        string_acupoint_judgment=string_acupoint_A[-1:]
                        if string_acupoint_judgment=='、':
                            string_acupoint_A = string_acupoint_A[:-1]
                            list_result_acupoint.append([word_of_Agroup,string_acupoint_A])
                    else:
                        list_result_acupoint.append([word_of_Agroup,'null'])
                elif flag_B_show == 1:
                    if string_acupoint_B != '':
                        string_acupoint_judgment=string_acupoint_B[-1:]
                        if string_acupoint_judgment=='、':
                            string_acupoint_B = string_acupoint_B[:-1]
                            list_result_acupoint.append([word_of_Bgroup,string_acupoint_B])
                    else:
                        list_result_acupoint.append([word_of_Bgroup,'null'])                  
    fsock_in.close()
    if result_illness_abstract != '':
        string_acupoint_judgment=result_illness_abstract[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness_abstract[:-1]
    else:
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness[:-1]
    result_dict['list_result_acupoint']=list_result_acupoint
    result_dict['result_illness']=result_illness
    return result_dict



def information_extraction_Case_observation(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    string_title = string_title[0:-4]
    flag_title = 0
    flag_begin = 0 
    fsock_in = open(inpath, "r")
    result_acupoint = ''
    result_illness = ''
    result_illness_abstract = ''
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_title = ''
        temp_sentence_top = ''
        temp_sentence_top_short = ''
        len_sentence=len(unicode_EachLine)
        if flag_title == 0:
            if len_sentence > 40:
                temp_sentence_title=unicode_EachLine[:40]
            else:
                temp_sentence_title=unicode_EachLine
            if temp_sentence_title.find(string_title) != -1:
                flag_title = 1          
        elif flag_title == 1:
            if len_sentence > 15:
                temp_sentence_top=unicode_EachLine[:15]
            else:
                temp_sentence_top=unicode_EachLine
            if len_sentence > 5:
                temp_sentence_top_short=unicode_EachLine[:5]
            else:
                temp_sentence_top_short=unicode_EachLine
            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
            if temp_illness !='':
                result_illness += temp_illness + '、'  
            cutResult = jieba.cut(temp_sentence_top_short)
            resultList = list(cutResult) 
            for x_word in resultList:
                if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                    temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                    if temp_illness!= '':
                        result_illness_abstract += temp_illness + '、'        
            result_judge_dict = judge_can_read_Case_observation(flag_begin,temp_sentence_top)
            flag_begin = result_judge_dict['flag_begin']                       
            if flag_begin==1:
                temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                if temp_acupoint_vocabulary !='':
                    result_acupoint += temp_acupoint_vocabulary + '、'  
    fsock_in.close()
    string_acupoint_judgment=result_acupoint[-1:]
    if string_acupoint_judgment=='、':
        result_acupoint = result_acupoint[:-1]
    if result_illness_abstract != '':
        string_acupoint_judgment=result_illness_abstract[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness_abstract[:-1]
    else:
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness[:-1]
    result_dict['result_illness']=result_illness
    result_dict['result_acupoint']=result_acupoint
    return result_dict



def information_extraction_Case_observation_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    flag_title = 0
    flag_begin = 0
    fsock_in = open(inpath, "r")
    result_acupoint = ''
    result_illness = ''
    result_illness_abstract = ''
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_title = ''
        temp_sentence_top = ''
        temp_sentence_top_short = ''
        len_sentence=len(unicode_EachLine)
        if flag_title == 0:
            if len_sentence > 15:
                temp_sentence_title=unicode_EachLine[:15]
            else:
                temp_sentence_title=unicode_EachLine
            if len_sentence > 5:
                temp_sentence_top_short=unicode_EachLine[:5]
            else:
                temp_sentence_top_short=unicode_EachLine
            cutResult = jieba.cut(temp_sentence_title)
            resultList = list(cutResult) 
            for x_word in resultList:
                if (x_word=='收稿日期') or (x_word=='收稿') or (x_word=='编辑') or (x_word=='发稿') or (x_word=='修回日期'):
                    flag_title = 1
        elif flag_title == 1:
            if len_sentence > 15:
                temp_sentence_top=unicode_EachLine[:15]
            else:
                temp_sentence_top=unicode_EachLine
            if len_sentence > 5:
                temp_sentence_top_short=unicode_EachLine[:5]
            else:
                temp_sentence_top_short=unicode_EachLine
            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
            if temp_illness !='':
                result_illness += temp_illness + '、'           
            cutResult = jieba.cut(temp_sentence_top_short)
            resultList = list(cutResult) 
            for x_word in resultList:
                if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                    temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                    if temp_illness!= '':
                        result_illness_abstract += temp_illness + '、'                                        
            result_judge_dict = judge_can_read_Case_observation(flag_begin,temp_sentence_top)
            flag_begin = result_judge_dict['flag_begin']
            if flag_begin==1:
                temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                if temp_acupoint_vocabulary !='':
                    result_acupoint += temp_acupoint_vocabulary + '、'  
    fsock_in.close()
    string_acupoint_judgment=result_acupoint[-1:]
    if string_acupoint_judgment=='、':
        result_acupoint = result_acupoint[:-1]
    if result_illness_abstract != '':
        string_acupoint_judgment=result_illness_abstract[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness_abstract[:-1]
    else:
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness[:-1]
    result_dict['result_illness']=result_illness
    result_dict['result_acupoint']=result_acupoint
    return result_dict



def information_extraction_Case_observation_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    flag_begin = 0 
    fsock_in = open(inpath, "r")
    result_acupoint = ''
    result_illness = ''
    result_illness_abstract = ''
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_top = ''
        temp_sentence_top_short = ''
        len_sentence=len(unicode_EachLine)
        if len_sentence > 15:
            temp_sentence_top=unicode_EachLine[:15]
        else:
            temp_sentence_top=unicode_EachLine
        if len_sentence > 5:
            temp_sentence_top_short=unicode_EachLine[:5]
        else:
            temp_sentence_top_short=unicode_EachLine      
        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
        if temp_illness !='':
            result_illness += temp_illness + '、'  
        cutResult = jieba.cut(temp_sentence_top_short)
        resultList = list(cutResult) 
        for x_word in resultList:
            if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                if temp_illness!= '':
                    result_illness_abstract += temp_illness + '、'                                       
        result_judge_dict = judge_can_read_Case_observation(flag_begin,temp_sentence_top)
        flag_begin = result_judge_dict['flag_begin']
        if flag_begin==1:
            temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
            if temp_acupoint_vocabulary !='':
                result_acupoint += temp_acupoint_vocabulary + '、'                 
    fsock_in.close()
    string_acupoint_judgment=result_acupoint[-1:]
    if string_acupoint_judgment=='、':
        result_acupoint = result_acupoint[:-1]
    if result_illness_abstract != '':
        string_acupoint_judgment=result_illness_abstract[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness_abstract[:-1]
    else:
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
            result_illness = result_illness[:-1]
    result_dict['result_illness']=result_illness
    result_dict['result_acupoint']=result_acupoint
    return result_dict



def information_extraction_Case_report(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    temp_sentence_top = ''
    flag_begin = 0 
    fsock_in = open(inpath, "r")
    result_acupoint_list = []
    result_illness_list = []
    result_acupoint = ''
    result_illness = ''
    result_illness_abstract = ''
    dict_number = {u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    dict_number_chinese = {u'一、':0,u'二、':0,u'三、':0,u'四、':0,u'五、':0}
    dict_an = {u'按：':0,u'按:':0}
    dict_li = {u'一':0,u'二':0,u'三':0,u'四':0,u'五':0,u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    flag_one = 0
    dict_which_feature_exists = judge_which_feature_exists(inpath,string_title)
    flag_number = dict_which_feature_exists['flag_number']
    flag_an = dict_which_feature_exists['flag_an']
    flag_li = dict_which_feature_exists['flag_li']
    flag_fangfa = dict_which_feature_exists['flag_fangfa']
    str_flag_number= '%d' %flag_number
    str_flag_an= '%d' %flag_an
    str_flag_li= '%d' %flag_li
    str_flag_fangfa= '%d' %flag_fangfa
    pprint('-----------------'+inpath+'--------------------\n')
    pprint('-----------------dict_which_feature_exists--------------------\n')
    pprint('flag_number='+ str_flag_number +'\n')
    pprint('flag_an='+ str_flag_an +'\n')
    pprint('flag_li='+ str_flag_li +'\n')
    pprint('flag_fangfa='+ str_flag_fangfa +'\n')
    if flag_number==1:
        for EachLine in fsock_in:
            EachLine= EachLine.strip('\n') 
            unicode_EachLine_dirty=unicode(EachLine,"utf8")
            unicode_EachLine="".join(unicode_EachLine_dirty.split())
            temp_sentence_top = ''
            len_sentence=len(unicode_EachLine)
            if len_sentence > 40:
                temp_sentence_top=unicode_EachLine[:40]
            else:
                temp_sentence_top=unicode_EachLine
            result_judge_dict = judge_can_read_Case_report(flag_begin,temp_sentence_top,string_title)
            flag_begin = result_judge_dict['flag_begin']
            if flag_begin==1:
                len_sentence=len(unicode_EachLine)
                if len_sentence > 10:
                    temp_sentence_top=unicode_EachLine[:10]
                else:
                    temp_sentence_top=unicode_EachLine
                if len(temp_sentence_top)>0:
                    word_first = temp_sentence_top[0]
                    word_two = temp_sentence_top[0:2]
                    if (dict_number.has_key(word_first)) or (dict_number_chinese.has_key(word_two)):
                        if len(unicode_EachLine)<10:
                            flag_one += 1
                            if flag_one == 1:
                                result_acupoint = ''
                                result_illness = ''
                            if flag_one > 1:
                                if result_acupoint != '':
                                    string_acupoint_judgment=result_acupoint[-1:]
                                    if string_acupoint_judgment=='、':
                                        result_acupoint = result_acupoint[:-1]
                                result_acupoint_list.append(result_acupoint)
                                result_acupoint = ''
                                if result_illness != '':
                                    string_acupoint_judgment=result_illness[-1:]
                                    if string_acupoint_judgment=='、':
                                        result_illness = result_illness[:-1]
                                result_illness_list.append(result_illness)
                                result_illness = ''
                temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                if temp_acupoint_vocabulary !='':
                    result_acupoint += temp_acupoint_vocabulary + '、'  
                if temp_illness !='':
                    result_illness += temp_illness + '、'             
        if result_acupoint != '':
            string_acupoint_judgment=result_acupoint[-1:]
            if string_acupoint_judgment=='、':
                result_acupoint = result_acupoint[:-1]
        result_acupoint_list.append(result_acupoint)
        result_acupoint = ''   
        if result_illness != '':
            string_acupoint_judgment=result_illness[-1:]
            if string_acupoint_judgment=='、':
                result_illness = result_illness[:-1]
        result_illness_list.append(result_illness)
        result_illness = ''
    else:
        if flag_an==1:
            for EachLine in fsock_in:
                EachLine= EachLine.strip('\n') 
                unicode_EachLine_dirty=unicode(EachLine,"utf8")
                unicode_EachLine="".join(unicode_EachLine_dirty.split())
                temp_sentence_top = ''
                len_sentence=len(unicode_EachLine)
                if len_sentence > 40:
                    temp_sentence_top=unicode_EachLine[:40]
                else:
                    temp_sentence_top=unicode_EachLine
                result_judge_dict = judge_can_read_Case_report(flag_begin,temp_sentence_top,string_title)
                flag_begin = result_judge_dict['flag_begin']
                if flag_begin==1:
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 10:
                        temp_sentence_top=unicode_EachLine[:10]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len(temp_sentence_top)>0:
                        word_two = temp_sentence_top[0:2]
                        if dict_an.has_key(word_two):
                            if result_acupoint != '':
                                string_acupoint_judgment=result_acupoint[-1:]
                                if string_acupoint_judgment=='、':
                                    result_acupoint = result_acupoint[:-1]
                            result_acupoint_list.append(result_acupoint)
                            result_acupoint = ''
                            
                            if result_illness != '':
                                string_acupoint_judgment=result_illness[-1:]
                                if string_acupoint_judgment=='、':
                                    result_illness = result_illness[:-1]
                            result_illness_list.append(result_illness)
                            result_illness = ''
                    temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                    temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                    if temp_acupoint_vocabulary !='':
                        result_acupoint += temp_acupoint_vocabulary + '、'  
                    if temp_illness !='':
                        result_illness += temp_illness + '、'                       
            if result_acupoint != '':
                string_acupoint_judgment=result_acupoint[-1:]
                if string_acupoint_judgment=='、':
                    result_acupoint = result_acupoint[:-1]
            result_acupoint_list.append(result_acupoint)
            result_acupoint = ''   
            if result_illness != '':
                string_acupoint_judgment=result_illness[-1:]
                if string_acupoint_judgment=='、':
                    result_illness = result_illness[:-1]
            result_illness_list.append(result_illness)
            result_illness = ''
        else:
            if flag_li==1:
                for EachLine in fsock_in:
                    EachLine= EachLine.strip('\n') 
                    unicode_EachLine_dirty=unicode(EachLine,"utf8")
                    unicode_EachLine="".join(unicode_EachLine_dirty.split())
                    temp_sentence_top = ''
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 40:
                        temp_sentence_top=unicode_EachLine[:40]
                    else:
                        temp_sentence_top=unicode_EachLine
                    result_judge_dict = judge_can_read_Case_report(flag_begin,temp_sentence_top,string_title)
                    flag_begin = result_judge_dict['flag_begin']
                    if flag_begin==1:
                        len_sentence=len(unicode_EachLine)
                        if len_sentence > 10:
                            temp_sentence_top=unicode_EachLine[:10]
                        else:
                            temp_sentence_top=unicode_EachLine
                        if len(temp_sentence_top)>0:
                            word_first = temp_sentence_top[0]
                            word_two = temp_sentence_top[0:2]
                            if word_first==u'例':
                                word_to_judge=temp_sentence_top[1:2]
                                if dict_li.has_key(word_to_judge):
                                    flag_one += 1
                                    if flag_one == 1:
                                        result_acupoint = ''
                                        result_illness = ''
                                    if flag_one > 1:
                                        if result_acupoint != '':
                                            string_acupoint_judgment=result_acupoint[-1:]
                                            if string_acupoint_judgment=='、':
                                                result_acupoint = result_acupoint[:-1]
                                        result_acupoint_list.append(result_acupoint)
                                        result_acupoint = ''
                                        if result_illness != '':
                                            string_acupoint_judgment=result_illness[-1:]
                                            if string_acupoint_judgment=='、':
                                                result_illness = result_illness[:-1]
                                        result_illness_list.append(result_illness)
                                        result_illness = ''
                        temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                        if temp_acupoint_vocabulary !='':
                            result_acupoint += temp_acupoint_vocabulary + '、'  
                        if temp_illness !='':
                            result_illness += temp_illness + '、'  
                if result_acupoint != '':
                    string_acupoint_judgment=result_acupoint[-1:]
                    if string_acupoint_judgment=='、':
                        result_acupoint = result_acupoint[:-1]
                result_acupoint_list.append(result_acupoint)
                result_acupoint = ''   
        
                if result_illness != '':
                    string_acupoint_judgment=result_illness[-1:]
                    if string_acupoint_judgment=='、':
                        result_illness = result_illness[:-1]
                result_illness_list.append(result_illness)
                result_illness = ''      
            else:
                for EachLine in fsock_in:
                    unicode_EachLine_dirty=unicode(EachLine,"utf8")
                    unicode_EachLine="".join(unicode_EachLine_dirty.split())
                    temp_sentence_top = ''
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 40:
                        temp_sentence_top=unicode_EachLine[:40]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len_sentence > 5:
                        temp_sentence_top_short=unicode_EachLine[:5]
                    else:
                        temp_sentence_top_short=unicode_EachLine
                    cutResult = jieba.cut(temp_sentence_top_short)
                    resultList = list(cutResult) 
                    for x_word in resultList:
                        if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                            if temp_illness!= '':
                                result_illness_abstract += temp_illness + '、'                    
                    result_judge_dict = judge_can_read_Case_report(flag_begin,temp_sentence_top,string_title)
                    flag_begin = result_judge_dict['flag_begin']
                    if flag_begin==1:
                        temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                        if temp_acupoint_vocabulary !='':
                            result_acupoint += temp_acupoint_vocabulary + '、'  
                        if temp_illness !='':
                            result_illness += temp_illness + '、'  
                if result_acupoint != '':
                    string_acupoint_judgment=result_acupoint[-1:]
                    if string_acupoint_judgment=='、':
                        result_acupoint = result_acupoint[:-1]
                result_acupoint_list.append(result_acupoint)
                result_acupoint = ''   
                
                if result_illness_abstract != '':
                    string_acupoint_judgment=result_illness_abstract[-1:]
                    if string_acupoint_judgment=='、':
                        result_illness = result_illness_abstract[:-1]
                else:
                    if result_illness != '':
                        string_acupoint_judgment=result_illness[-1:]
                        if string_acupoint_judgment=='、':
                            result_illness = result_illness[:-1]             
                result_illness_list.append(result_illness)
               
    result_dict['result_illness_list']=result_illness_list
    result_dict['result_acupoint_list']=result_acupoint_list
    string_debug = ''
    for string in result_acupoint_list:
        string_debug += string + '、'
    pprint('string_debug='+ string_debug +'\n')
    pprint('string_debug=')
    pprint(result_acupoint_list)
    pprint('\n')
    fsock_in.close()
    return result_dict



def information_extraction_Case_report_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title,string_illness):
    result_dict = {}
    temp_sentence_top = ''
    flag_begin = 0 
    fsock_in = open(inpath, "r")
    result_acupoint_list = []
    result_illness_list = []
    result_acupoint = ''
    result_illness = ''
    result_illness_abstract = ''
    dict_number = {u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    dict_number_chinese = {u'一、':0,u'二、':0,u'三、':0,u'四、':0,u'五、':0}
    dict_an = {u'按：':0,u'按:':0}
    dict_li = {u'一':0,u'二':0,u'三':0,u'四':0,u'五':0,u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    flag_one = 0
    dict_which_feature_exists = judge_which_feature_exists(inpath,string_title)
    flag_number = dict_which_feature_exists['flag_number']
    flag_an = dict_which_feature_exists['flag_an']
    flag_li = dict_which_feature_exists['flag_li']
    flag_fangfa = dict_which_feature_exists['flag_fangfa']
    str_flag_number= '%d' %flag_number
    str_flag_an= '%d' %flag_an
    str_flag_li= '%d' %flag_li
    str_flag_fangfa= '%d' %flag_fangfa
    pprint('-----------------'+inpath+'--------------------\n')
    pprint('-----------------dict_which_feature_exists--------------------\n')
    pprint('flag_number='+ str_flag_number +'\n')
    pprint('flag_an='+ str_flag_an +'\n')
    pprint('flag_li='+ str_flag_li +'\n')
    pprint('flag_fangfa='+ str_flag_fangfa +'\n')
    if flag_number==1:
        for EachLine in fsock_in:
            EachLine= EachLine.strip('\n') 
            unicode_EachLine_dirty=unicode(EachLine,"utf8")
            unicode_EachLine="".join(unicode_EachLine_dirty.split())
            temp_sentence_top = ''
            len_sentence=len(unicode_EachLine)
            if len_sentence > 40:
                temp_sentence_top=unicode_EachLine[:40]
            else:
                temp_sentence_top=unicode_EachLine
            result_judge_dict = judge_can_read_Case_report_normal(flag_begin,temp_sentence_top,string_title,string_illness)
            flag_begin = result_judge_dict['flag_begin']
            if flag_begin==1:
                len_sentence=len(unicode_EachLine)
                if len_sentence > 10:
                    temp_sentence_top=unicode_EachLine[:10]
                else:
                    temp_sentence_top=unicode_EachLine
                if len(temp_sentence_top)>0:
                    word_first = temp_sentence_top[0]
                    word_two = temp_sentence_top[0:2]
                    if (dict_number.has_key(word_first)) or (dict_number_chinese.has_key(word_two)):
                        if len(unicode_EachLine)<10:
                            flag_one += 1
                            if flag_one == 1:
                                result_acupoint = ''
                                result_illness = ''
                            if flag_one > 1:
                                if result_acupoint != '':
                                    string_acupoint_judgment=result_acupoint[-1:]
                                    if string_acupoint_judgment=='、':
                                        result_acupoint = result_acupoint[:-1]
                                result_acupoint_list.append(result_acupoint)
                                result_acupoint = ''
                                if result_illness != '':
                                    string_acupoint_judgment=result_illness[-1:]
                                    if string_acupoint_judgment=='、':
                                        result_illness = result_illness[:-1]
                                result_illness_list.append(result_illness)
                                result_illness = ''
                temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                if temp_acupoint_vocabulary !='':
                    result_acupoint += temp_acupoint_vocabulary + '、'  
                if temp_illness !='':
                    result_illness += temp_illness + '、'  
        if result_acupoint != '':
            string_acupoint_judgment=result_acupoint[-1:]
            if string_acupoint_judgment=='、':
                result_acupoint = result_acupoint[:-1]
        result_acupoint_list.append(result_acupoint)
        result_acupoint = ''   
        if result_illness != '':
            string_acupoint_judgment=result_illness[-1:]
            if string_acupoint_judgment=='、':
                result_illness = result_illness[:-1]
        result_illness_list.append(result_illness)
        result_illness = ''
    else:
        if flag_an==1:
            for EachLine in fsock_in:
                EachLine= EachLine.strip('\n') 
                unicode_EachLine_dirty=unicode(EachLine,"utf8")
                unicode_EachLine="".join(unicode_EachLine_dirty.split())
                temp_sentence_top = ''
                len_sentence=len(unicode_EachLine)
                if len_sentence > 40:
                    temp_sentence_top=unicode_EachLine[:40]
                else:
                    temp_sentence_top=unicode_EachLine
                result_judge_dict = judge_can_read_Case_report_normal(flag_begin,temp_sentence_top,string_title,string_illness)
                flag_begin = result_judge_dict['flag_begin']
                if flag_begin==1:
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 10:
                        temp_sentence_top=unicode_EachLine[:10]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len(temp_sentence_top)>0:
                        word_two = temp_sentence_top[0:2]
                        if dict_an.has_key(word_two):
                            if result_acupoint != '':
                                string_acupoint_judgment=result_acupoint[-1:]
                                if string_acupoint_judgment=='、':
                                    result_acupoint = result_acupoint[:-1]
                            result_acupoint_list.append(result_acupoint)
                            result_acupoint = ''
                            
                            if result_illness != '':
                                string_acupoint_judgment=result_illness[-1:]
                                if string_acupoint_judgment=='、':
                                    result_illness = result_illness[:-1]
                            result_illness_list.append(result_illness)
                            result_illness = ''
                    temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                    temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                    if temp_acupoint_vocabulary !='':
                        result_acupoint += temp_acupoint_vocabulary + '、'  
                    if temp_illness !='':
                        result_illness += temp_illness + '、'   
            if result_acupoint != '':
                string_acupoint_judgment=result_acupoint[-1:]
                if string_acupoint_judgment=='、':
                    result_acupoint = result_acupoint[:-1]
            result_acupoint_list.append(result_acupoint)
            result_acupoint = ''   
            if result_illness != '':
                string_acupoint_judgment=result_illness[-1:]
                if string_acupoint_judgment=='、':
                    result_illness = result_illness[:-1]
            result_illness_list.append(result_illness)
            result_illness = ''  
        else:
            if flag_li==1:
                for EachLine in fsock_in:
                    EachLine= EachLine.strip('\n') 
                    unicode_EachLine_dirty=unicode(EachLine,"utf8")
                    unicode_EachLine="".join(unicode_EachLine_dirty.split())
                    temp_sentence_top = ''
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 40:
                        temp_sentence_top=unicode_EachLine[:40]
                    else:
                        temp_sentence_top=unicode_EachLine
                    result_judge_dict = judge_can_read_Case_report_normal(flag_begin,temp_sentence_top,string_title,string_illness)
                    flag_begin = result_judge_dict['flag_begin']
                    if flag_begin==1:
                        len_sentence=len(unicode_EachLine)
                        if len_sentence > 10:
                            temp_sentence_top=unicode_EachLine[:10]
                        else:
                            temp_sentence_top=unicode_EachLine
                        if len(temp_sentence_top)>0:
                            word_first = temp_sentence_top[0]
                            word_two = temp_sentence_top[0:2]
                            if word_first==u'例':
                                word_to_judge=temp_sentence_top[1:2]
                                if dict_li.has_key(word_to_judge):
                                    flag_one += 1
                                    if flag_one == 1:
                                        result_acupoint = ''
                                        result_illness = ''
                                    if flag_one > 1:
                                        if result_acupoint != '':
                                            string_acupoint_judgment=result_acupoint[-1:]
                                            if string_acupoint_judgment=='、':
                                                result_acupoint = result_acupoint[:-1]
                                        result_acupoint_list.append(result_acupoint)
                                        result_acupoint = ''
                                        if result_illness != '':
                                            string_acupoint_judgment=result_illness[-1:]
                                            if string_acupoint_judgment=='、':
                                                result_illness = result_illness[:-1]
                                        result_illness_list.append(result_illness)
                                        result_illness = ''
                        temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                        if temp_acupoint_vocabulary !='':
                            result_acupoint += temp_acupoint_vocabulary + '、'  
                        if temp_illness !='':
                            result_illness += temp_illness + '、'      
                if result_acupoint != '':
                    string_acupoint_judgment=result_acupoint[-1:]
                    if string_acupoint_judgment=='、':
                        result_acupoint = result_acupoint[:-1]
                result_acupoint_list.append(result_acupoint)
                result_acupoint = ''           
                if result_illness != '':
                    string_acupoint_judgment=result_illness[-1:]
                    if string_acupoint_judgment=='、':
                        result_illness = result_illness[:-1]
                result_illness_list.append(result_illness)
                result_illness = ''  
            else:
                for EachLine in fsock_in:
                    unicode_EachLine_dirty=unicode(EachLine,"utf8")
                    unicode_EachLine="".join(unicode_EachLine_dirty.split())
                    temp_sentence_top = ''
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 40:
                        temp_sentence_top=unicode_EachLine[:40]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len_sentence > 5:
                        temp_sentence_top_short=unicode_EachLine[:5]
                    else:
                        temp_sentence_top_short=unicode_EachLine
                    cutResult = jieba.cut(temp_sentence_top_short)
                    resultList = list(cutResult) 
                    for x_word in resultList:
                        if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                            if temp_illness!= '':
                                result_illness_abstract += temp_illness + '、' 
                    result_judge_dict = judge_can_read_Case_report_normal(flag_begin,temp_sentence_top,string_title,string_illness)
                    flag_begin = result_judge_dict['flag_begin']
                    if flag_begin==1:
                        temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                        if temp_acupoint_vocabulary !='':
                            result_acupoint += temp_acupoint_vocabulary + '、'  
                        if temp_illness !='':
                            result_illness += temp_illness + '、'  
                if result_acupoint != '':
                    string_acupoint_judgment=result_acupoint[-1:]
                    if string_acupoint_judgment=='、':
                        result_acupoint = result_acupoint[:-1]
                result_acupoint_list.append(result_acupoint)
                result_acupoint = ''   
                
                if result_illness_abstract != '':
                    string_acupoint_judgment=result_illness_abstract[-1:]
                    if string_acupoint_judgment=='、':
                        result_illness = result_illness_abstract[:-1]
                else:
                    if result_illness != '':
                        string_acupoint_judgment=result_illness[-1:]
                        if string_acupoint_judgment=='、':
                            result_illness = result_illness[:-1]               
                result_illness_list.append(result_illness)
    result_dict['result_illness_list']=result_illness_list
    result_dict['result_acupoint_list']=result_acupoint_list
    string_debug = ''
    for string in result_acupoint_list:
        string_debug += string + '、'
    pprint('string_debug='+ string_debug +'\n')
    pprint('string_debug=')
    pprint(result_acupoint_list)
    pprint('\n')
    fsock_in.close()
    return result_dict



def information_extraction_Case_report_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    temp_sentence_top = ''
    flag_begin = 0  
    fsock_in = open(inpath, "r")
    result_acupoint_list = []
    result_illness_list = []
    result_acupoint = ''
    result_illness = ''
    result_illness_abstract = ''
    dict_number = {u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    dict_number_chinese = {u'一、':0,u'二、':0,u'三、':0,u'四、':0,u'五、':0}
    dict_an = {u'按：':0,u'按:':0}
    dict_li = {u'一':0,u'二':0,u'三':0,u'四':0,u'五':0,u'1':0,u'2':0,u'3':0,u'4':0,u'5':0}
    flag_one = 0
    dict_which_feature_exists = judge_which_feature_exists(inpath,string_title)
    flag_number = dict_which_feature_exists['flag_number']
    flag_an = dict_which_feature_exists['flag_an']
    flag_li = dict_which_feature_exists['flag_li']
    flag_fangfa = dict_which_feature_exists['flag_fangfa']
    str_flag_number= '%d' %flag_number
    str_flag_an= '%d' %flag_an
    str_flag_li= '%d' %flag_li
    str_flag_fangfa= '%d' %flag_fangfa
    pprint('-----------------'+inpath+'--------------------\n')
    pprint('-----------------dict_which_feature_exists--------------------\n')
    pprint('flag_number='+ str_flag_number +'\n')
    pprint('flag_an='+ str_flag_an +'\n')
    pprint('flag_li='+ str_flag_li +'\n')
    pprint('flag_fangfa='+ str_flag_fangfa +'\n')
    if flag_number==1:
        for EachLine in fsock_in:
            EachLine= EachLine.strip('\n') 
            unicode_EachLine_dirty=unicode(EachLine,"utf8")
            unicode_EachLine="".join(unicode_EachLine_dirty.split())
            temp_sentence_top = ''
            len_sentence=len(unicode_EachLine)
            if len_sentence > 40:
                temp_sentence_top=unicode_EachLine[:40]
            else:
                temp_sentence_top=unicode_EachLine
            result_judge_dict = judge_can_read_Case_report_normal2(flag_begin,temp_sentence_top,string_title)
            flag_begin = result_judge_dict['flag_begin']
            if flag_begin==1:
                len_sentence=len(unicode_EachLine)
                if len_sentence > 10:
                    temp_sentence_top=unicode_EachLine[:10]
                else:
                    temp_sentence_top=unicode_EachLine
                if len(temp_sentence_top)>0:
                    word_first = temp_sentence_top[0]
                    word_two = temp_sentence_top[0:2]
                    if (dict_number.has_key(word_first)) or (dict_number_chinese.has_key(word_two)):
                        if len(unicode_EachLine)<10:
                            flag_one += 1
                            if flag_one == 1:
                                result_acupoint = ''
                                result_illness = ''
                            if flag_one > 1:
                                if result_acupoint != '':
                                    string_acupoint_judgment=result_acupoint[-1:]
                                    if string_acupoint_judgment=='、':
                                        result_acupoint = result_acupoint[:-1]
                                result_acupoint_list.append(result_acupoint)
                                result_acupoint = ''
                                if result_illness != '':
                                    string_acupoint_judgment=result_illness[-1:]
                                    if string_acupoint_judgment=='、':
                                        result_illness = result_illness[:-1]
                                result_illness_list.append(result_illness)
                                result_illness = ''
                temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                if temp_acupoint_vocabulary !='':
                    result_acupoint += temp_acupoint_vocabulary + '、'  
                if temp_illness !='':
                    result_illness += temp_illness + '、'                       
        if result_acupoint != '':
            string_acupoint_judgment=result_acupoint[-1:]
            if string_acupoint_judgment=='、':
                result_acupoint = result_acupoint[:-1]
        result_acupoint_list.append(result_acupoint)
        result_acupoint = ''   
        if result_illness != '':
            string_acupoint_judgment=result_illness[-1:]
            if string_acupoint_judgment=='、':
                result_illness = result_illness[:-1]
        result_illness_list.append(result_illness)
        result_illness = ''
    else:
        if flag_an==1:
            for EachLine in fsock_in:
                EachLine= EachLine.strip('\n') 
                unicode_EachLine_dirty=unicode(EachLine,"utf8")
                unicode_EachLine="".join(unicode_EachLine_dirty.split())
                temp_sentence_top = ''
                len_sentence=len(unicode_EachLine)
                if len_sentence > 40:
                    temp_sentence_top=unicode_EachLine[:40]
                else:
                    temp_sentence_top=unicode_EachLine
                result_judge_dict = judge_can_read_Case_report_normal2(flag_begin,temp_sentence_top,string_title)
                flag_begin = result_judge_dict['flag_begin']
                if flag_begin==1:
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 10:
                        temp_sentence_top=unicode_EachLine[:10]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len(temp_sentence_top)>0:
                        word_two = temp_sentence_top[0:2]
                        if dict_an.has_key(word_two):
                            if result_acupoint != '':
                                string_acupoint_judgment=result_acupoint[-1:]
                                if string_acupoint_judgment=='、':
                                    result_acupoint = result_acupoint[:-1]
                            result_acupoint_list.append(result_acupoint)
                            result_acupoint = ''                            
                            if result_illness != '':
                                string_acupoint_judgment=result_illness[-1:]
                                if string_acupoint_judgment=='、':
                                    result_illness = result_illness[:-1]
                            result_illness_list.append(result_illness)
                            result_illness = ''
                    temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                    temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                    if temp_acupoint_vocabulary !='':
                        result_acupoint += temp_acupoint_vocabulary + '、'  
                    if temp_illness !='':
                        result_illness += temp_illness + '、'  
            if result_acupoint != '':
                string_acupoint_judgment=result_acupoint[-1:]
                if string_acupoint_judgment=='、':
                    result_acupoint = result_acupoint[:-1]
            result_acupoint_list.append(result_acupoint)
            result_acupoint = ''   
            if result_illness != '':
                string_acupoint_judgment=result_illness[-1:]
                if string_acupoint_judgment=='、':
                    result_illness = result_illness[:-1]
            result_illness_list.append(result_illness)
            result_illness = ''
        else:
            if flag_li==1:
                for EachLine in fsock_in:
                    EachLine= EachLine.strip('\n') 
                    unicode_EachLine_dirty=unicode(EachLine,"utf8")
                    unicode_EachLine="".join(unicode_EachLine_dirty.split())
                    temp_sentence_top = ''
                    temp_sentence_top_short = ''
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 40:
                        temp_sentence_top=unicode_EachLine[:40]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len_sentence > 5:
                        temp_sentence_top_short=unicode_EachLine[:5]
                    else:
                        temp_sentence_top_short=unicode_EachLine
                    cutResult = jieba.cut(temp_sentence_top_short)
                    resultList = list(cutResult) 
                    for x_word in resultList:
                        if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                            if temp_illness!= '':
                                result_illness_abstract += temp_illness + '、'  
                    result_judge_dict = judge_can_read_Case_report_normal2(flag_begin,temp_sentence_top,string_title)
                    flag_begin = result_judge_dict['flag_begin']
                    if flag_begin==1:
                        len_sentence=len(unicode_EachLine)
                        if len_sentence > 10:
                            temp_sentence_top=unicode_EachLine[:10]
                        else:
                            temp_sentence_top=unicode_EachLine
                        if len(temp_sentence_top)>0:
                            word_first = temp_sentence_top[0]
                            word_two = temp_sentence_top[0:2]
                            if word_first==u'例':
                                word_to_judge=temp_sentence_top[1:2]
                                if dict_li.has_key(word_to_judge):
                                    flag_one += 1
                                    if flag_one == 1:
                                        result_acupoint = ''
                                        result_illness = ''
                                    if flag_one > 1:
                                        if result_acupoint != '':
                                            string_acupoint_judgment=result_acupoint[-1:]
                                            if string_acupoint_judgment=='、':
                                                result_acupoint = result_acupoint[:-1]
                                        result_acupoint_list.append(result_acupoint)
                                        result_acupoint = ''
                                        if result_illness != '':
                                            string_acupoint_judgment=result_illness[-1:]
                                            if string_acupoint_judgment=='、':
                                                result_illness = result_illness[:-1]
                                        result_illness_list.append(result_illness)
                                        result_illness = ''
                        temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                        if temp_acupoint_vocabulary !='':
                            result_acupoint += temp_acupoint_vocabulary + '、'  
                        if temp_illness !='':
                            result_illness += temp_illness + '、'   
                if result_acupoint != '':
                    string_acupoint_judgment=result_acupoint[-1:]
                    if string_acupoint_judgment=='、':
                        result_acupoint = result_acupoint[:-1]
                result_acupoint_list.append(result_acupoint)
                result_acupoint = ''   
        
                if result_illness != '':
                    string_acupoint_judgment=result_illness[-1:]
                    if string_acupoint_judgment=='、':
                        result_illness = result_illness[:-1]
                result_illness_list.append(result_illness)
                result_illness = ''
            else:
                for EachLine in fsock_in:
                    unicode_EachLine_dirty=unicode(EachLine,"utf8")
                    unicode_EachLine="".join(unicode_EachLine_dirty.split())
                    temp_sentence_top = ''
                    temp_sentence_top_short = ''
                    len_sentence=len(unicode_EachLine)
                    if len_sentence > 40:
                        temp_sentence_top=unicode_EachLine[:40]
                    else:
                        temp_sentence_top=unicode_EachLine
                    if len_sentence > 5:
                        temp_sentence_top_short=unicode_EachLine[:5]
                    else:
                        temp_sentence_top_short=unicode_EachLine
                    cutResult = jieba.cut(temp_sentence_top_short)
                    resultList = list(cutResult) 
                    for x_word in resultList:
                        if (x_word=='摘要') or (x_word=='关键词') or (x_word=='主题词'):
                            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                            if temp_illness!= '':
                                result_illness_abstract += temp_illness + '、'                  
                    result_judge_dict = judge_can_read_Case_report_normal2(flag_begin,temp_sentence_top,string_title)
                    flag_begin = result_judge_dict['flag_begin']
                    if flag_begin==1:
                        temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                        temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
                        if temp_acupoint_vocabulary !='':
                            result_acupoint += temp_acupoint_vocabulary + '、'  
                        if temp_illness !='':
                            result_illness += temp_illness + '、'  
                if result_acupoint != '':
                    string_acupoint_judgment=result_acupoint[-1:]
                    if string_acupoint_judgment=='、':
                        result_acupoint = result_acupoint[:-1]
                result_acupoint_list.append(result_acupoint)
                result_acupoint = ''   
                
                if result_illness_abstract != '':
                    string_acupoint_judgment=result_illness_abstract[-1:]
                    if string_acupoint_judgment=='、':
                        result_illness = result_illness_abstract[:-1]
                else:
                    if result_illness != '':
                        string_acupoint_judgment=result_illness[-1:]
                        if string_acupoint_judgment=='、':
                            result_illness = result_illness[:-1]                  
                result_illness_list.append(result_illness)
    result_dict['result_illness_list']=result_illness_list
    result_dict['result_acupoint_list']=result_acupoint_list
    string_debug = ''
    for string in result_acupoint_list:
        string_debug += string + '、'
    pprint('string_debug='+ string_debug +'\n')
    pprint('string_debug=')
    pprint(result_acupoint_list)
    pprint('\n')
    fsock_in.close()
    return result_dict   



def information_extraction_Case_report_OneResult(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    temp_sentence_top = ''
    flag_begin = 0  
    result_acupoint = ''
    result_illness = ''
    fsock_in = open(inpath, "r")
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_top = ''
        len_sentence=len(unicode_EachLine)
        if len_sentence > 40:
            temp_sentence_top=unicode_EachLine[:40]
        else:
            temp_sentence_top=unicode_EachLine
        result_judge_dict = judge_can_read_Case_report(flag_begin,temp_sentence_top,string_title)
        flag_begin = result_judge_dict['flag_begin']
        if flag_begin==1:
            temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
            if temp_acupoint_vocabulary !='':
                result_acupoint += temp_acupoint_vocabulary + '、'  
            if temp_illness !='':
                result_illness += temp_illness + '、'  
    fsock_in.close()
    if (result_acupoint != ''):
        string_acupoint_judgment=result_acupoint[-1:]
        if string_acupoint_judgment=='、':
    	     result_acupoint = result_acupoint[:-1]
    if (result_illness != ''):
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
    	     result_illness = result_illness[:-1]                      
    result_dict['result_illness']=result_illness
    result_dict['result_acupoint']=result_acupoint
    return result_dict   



def information_extraction_Case_report_OneResult_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title,string_illness):
    result_dict = {}
    temp_sentence_top = ''
    flag_begin = 0 
    result_acupoint = ''
    result_illness = ''
    fsock_in = open(inpath, "r")
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_top = ''
        len_sentence=len(unicode_EachLine)
        if len_sentence > 40:
            temp_sentence_top=unicode_EachLine[:40]
        else:
            temp_sentence_top=unicode_EachLine
        result_judge_dict = judge_can_read_Case_report_normal(flag_begin,temp_sentence_top,string_title,string_illness)
        flag_begin = result_judge_dict['flag_begin']
        if flag_begin==1:
            temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
            if temp_acupoint_vocabulary !='':
                result_acupoint += temp_acupoint_vocabulary + '、'  
            if temp_illness !='':
                result_illness += temp_illness + '、'  
    fsock_in.close()
    if (result_acupoint != ''):
        string_acupoint_judgment=result_acupoint[-1:]
        if string_acupoint_judgment=='、':
             result_acupoint = result_acupoint[:-1]
    if (result_illness != ''):
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
             result_illness = result_illness[:-1]         
    result_dict['result_illness']=result_illness
    result_dict['result_acupoint']=result_acupoint
    return result_dict   



def information_extraction_Case_report_OneResult_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title):
    result_dict = {}
    temp_sentence_top = ''
    flag_begin = 0  
    result_acupoint = ''
    result_illness = ''
    fsock_in = open(inpath, "r")
    for EachLine in fsock_in:
        EachLine= EachLine.strip('\n') 
        unicode_EachLine_dirty=unicode(EachLine,"utf8")
        unicode_EachLine="".join(unicode_EachLine_dirty.split())
        temp_sentence_top = ''
        len_sentence=len(unicode_EachLine)
        if len_sentence > 40:
            temp_sentence_top=unicode_EachLine[:40]
        else:
            temp_sentence_top=unicode_EachLine
        result_judge_dict = judge_can_read_Case_report_normal2(flag_begin,temp_sentence_top,string_title)
        flag_begin = result_judge_dict['flag_begin']
        if flag_begin==1:
            temp_acupoint_vocabulary = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,unicode_EachLine,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
            temp_illness = get_illness_in_sentence(illness_dict,unicode_EachLine)
            if temp_acupoint_vocabulary !='':
                result_acupoint += temp_acupoint_vocabulary + '、'  
            if temp_illness !='':
                result_illness += temp_illness + '、'  
    fsock_in.close()
    if (result_acupoint != ''):
        string_acupoint_judgment=result_acupoint[-1:]
        if string_acupoint_judgment=='、':
             result_acupoint = result_acupoint[:-1]
    if (result_illness != ''):
        string_acupoint_judgment=result_illness[-1:]
        if string_acupoint_judgment=='、':
             result_illness = result_illness[:-1]                     
    result_dict['result_illness']=result_illness
    result_dict['result_acupoint']=result_acupoint
    return result_dict   



def To_Get_information_extraction_files(list_filename_all,filepath,file_out_name):
    fsock_out = open(file_out_name, "w")
    fsock_out.write("编号,文献题目名称,所属的种类码,所属的文献种类,抽取的病名汇总,抽取的穴位汇总\n")
    target_names = ['病例观察','对照试验','个案报道']
    illness_dict = get_illness_dict()
    dict_acupoint_vocabulary_confused=get_acupoint_vocabulary_confused_dict()
    dict_acupoint_vocabulary_summary = get_acupoint_vocabulary_dict()
    dict_convert_acupoint_vocabulary=get_convert_acupoint_vocabulary_dict() 
    dict_convert_wrong = get_convert_wrong_dict()  
    dict_convert_combination = get_convert_combination_dict() 
    dict_data=get_dict_data(1970,2017)
    dict_number=get_dict_data(1,4)
    dict_acupoint_vocabulary_Duplicate_removal = get_acupoint_vocabulary_Duplicate_removal_dict()
    for list_file in list_filename_all:
        string_number=list_file[0] 
        print('string_number:'+string_number+'\n')
        string_title=list_file[1] 
        string_classification_num = list_file[2] 
        string_classification_name = ''
        string_acupoint = ''
        string_illness = ''
        string_illness = get_illness_in_sentence(illness_dict,string_title)
        inpath = filepath + string_title + '.txt'
        if string_classification_num == u'1':
            string_classification_name = target_names[0]
            result_dict = information_extraction_Case_observation(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
            result_illness = result_dict['result_illness']
            result_acupoint = result_dict['result_acupoint']
            string_acupoint = result_acupoint
            if string_acupoint =='':
                result_dict = information_extraction_Case_observation_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                result_illness = result_dict['result_illness']
                result_acupoint = result_dict['result_acupoint']
                string_acupoint = result_acupoint
                if string_acupoint =='':
                    result_dict = information_extraction_Case_observation_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                    result_illness = result_dict['result_illness']
                    result_acupoint = result_dict['result_acupoint']
                    string_acupoint = result_acupoint
            if len(string_acupoint)>0:
                string_acupoint = to_Duplicate_removal_acupoint_vocabulary(string_acupoint,dict_convert_acupoint_vocabulary)
            if len(string_illness)>0:
                string_illness = to_Duplicate_removal_illness(string_illness)
            else:
                if len(result_illness)>0:
                    string_illness = to_Duplicate_removal_illness(result_illness)
            if (string_acupoint==''):
                string_acupoint = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,string_title,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                if len(string_acupoint)>0:
                    string_acupoint = to_Duplicate_removal_acupoint_vocabulary(string_acupoint,dict_convert_acupoint_vocabulary)     
            fsock_out.write(string_number +','+ string_title +','+ string_classification_num +','+ string_classification_name +','+ string_illness +','+ string_acupoint + "\n")
        elif string_classification_num == u'2':
            string_classification_name = target_names[1]
            result_dict = information_extraction_Controlled_trial(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,dict_data,dict_number)
            list_result_acupoint = result_dict['list_result_acupoint']
            result_illness = result_dict['result_illness']
            flag_not_null=0
            if len(list_result_acupoint) >0:
                for list_reslut_acupoint_part in list_result_acupoint:
                    if list_reslut_acupoint_part[1] != 'null':
                        flag_not_null=1
            if flag_not_null == 1:
                for list_reslut_acupoint_part in list_result_acupoint:
                    if list_reslut_acupoint_part[1] !='null':
                        string_reslut_acupoint_part = list_reslut_acupoint_part[1]
                        string_reslut_acupoint_part = to_Duplicate_removal_acupoint_vocabulary(string_reslut_acupoint_part,dict_convert_acupoint_vocabulary)
                        string_acupoint += list_reslut_acupoint_part[0] + ':' + string_reslut_acupoint_part + ';'
            else :
                result_dict = information_extraction_Controlled_trial_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                list_result_acupoint = result_dict['list_result_acupoint']
                result_illness = result_dict['result_illness']
                flag_not_null_2 = 0
                if len(list_result_acupoint) >0:
                    for list_reslut_acupoint_part in list_result_acupoint:
                        if list_reslut_acupoint_part[1] != 'null':
                            flag_not_null_2 = 1
                if flag_not_null_2 == 1:
                    for list_reslut_acupoint_part in list_result_acupoint:
                        if list_reslut_acupoint_part[1] !='null':
                            string_reslut_acupoint_part = list_reslut_acupoint_part[1]
                            string_reslut_acupoint_part = to_Duplicate_removal_acupoint_vocabulary(string_reslut_acupoint_part,dict_convert_acupoint_vocabulary)
                            string_acupoint += list_reslut_acupoint_part[0] + ':' + string_reslut_acupoint_part + ';'
                else :
                    result_dict = information_extraction_Case_observation(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                    result_illness = result_dict['result_illness']
                    result_acupoint = result_dict['result_acupoint']
                    string_acupoint = result_acupoint
                    if string_acupoint =='':
                        result_dict = information_extraction_Case_observation_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                        result_illness = result_dict['result_illness']
                        result_acupoint = result_dict['result_acupoint']
                        string_acupoint = result_acupoint              
                        if string_acupoint =='':
                            result_dict = information_extraction_Case_observation_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                            result_illness = result_dict['result_illness']
                            result_acupoint = result_dict['result_acupoint']
                            string_acupoint = result_acupoint                                    
                    if len(string_acupoint)>0:
                        string_acupoint = to_Duplicate_removal_acupoint_vocabulary(string_acupoint,dict_convert_acupoint_vocabulary)
            if len(string_illness)>0:
                string_illness = to_Duplicate_removal_illness(string_illness)
            else:
                if len(result_illness)>0:
                    string_illness = to_Duplicate_removal_illness(result_illness)
            if (string_acupoint==''):
                string_acupoint = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,string_title,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                if len(string_acupoint)>0:
                    string_acupoint = to_Duplicate_removal_acupoint_vocabulary(string_acupoint,dict_convert_acupoint_vocabulary)       
            fsock_out.write(string_number +','+ string_title +','+ string_classification_num +','+ string_classification_name +','+ string_illness +','+ string_acupoint + "\n") 
        elif string_classification_num == u'3':
            string_classification_name = target_names[2]
            if (string_illness == ''):
                flag_acupoint_vocabulary_null=0
                result_dict_list = information_extraction_Case_report(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                result_illness_list = result_dict_list['result_illness_list']
                result_acupoint_list = result_dict_list['result_acupoint_list']
                temp_string_acupoint = ''
                for temp_word in result_acupoint_list:
                    if temp_word !='':
                       temp_string_acupoint += temp_word
                if temp_string_acupoint=='':
                    for word in result_illness_list:
                        if word !='':
                            string_illness += word
                    list_string_illness= string_illness.split('、') 
                    string_illness_part =list_string_illness[0]
                    if string_illness_part=='':
                        string_illness = ''
                        result_dict_list = information_extraction_Case_report_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                        result_illness_list = result_dict_list['result_illness_list']
                        result_acupoint_list = result_dict_list['result_acupoint_list']
                        temp_string_acupoint=''
                        for temp_word in result_acupoint_list:
                            if temp_word !='':
                                temp_string_acupoint += temp_word
                        if temp_string_acupoint=='':
                            string_acupoint = ''
                            for i in range(len(result_illness_list)):
                                temp_string_illness = result_illness_list[i]
                                if temp_string_illness !='':
                                    string_illness += temp_string_illness + '、'
                            if string_illness != '':
                                string_acupoint_judgment=string_illness[-1:]
                                if string_acupoint_judgment=='、':
                                    string_illness = string_illness[:-1]
                                string_illness = to_Duplicate_removal_illness(string_illness)
                        else:
                            for i in range(len(result_acupoint_list)):
                                temp_string_acupoint = to_Duplicate_removal_acupoint_vocabulary(result_acupoint_list[i],dict_convert_acupoint_vocabulary)
                                temp_string_illness = to_Duplicate_removal_illness(result_illness_list[i])
                                if temp_string_acupoint != '':
                                    string_acupoint += '（'+ temp_string_illness +'穴位处方）' + temp_string_acupoint + ';' 
                                    if temp_string_illness != '':
                                        string_illness += temp_string_illness + '、'                                      
                            if string_illness != '':
                                string_acupoint_judgment=string_illness[-1:]
                                if string_acupoint_judgment=='、':
                                    string_illness = string_illness[:-1]
                                string_illness = to_Duplicate_removal_illness(string_illness)
                    else:
                        string_illness = ''
                        result_dict_list = information_extraction_Case_report_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title,string_illness_part)
                        result_illness_list = result_dict_list['result_illness_list']
                        result_acupoint_list = result_dict_list['result_acupoint_list']
                        temp_string_acupoint=''
                        for temp_word in result_acupoint_list:
                            if temp_word !='':
                                temp_string_acupoint += temp_word
                        if temp_string_acupoint=='':
                            string_illness = ''                            
                            result_dict_list = information_extraction_Case_report_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                            result_illness_list = result_dict_list['result_illness_list']
                            result_acupoint_list = result_dict_list['result_acupoint_list']
                            temp_string_acupoint=''
                            for temp_word in result_acupoint_list:
                                if temp_word !='':
                                    temp_string_acupoint += temp_word
                            if temp_string_acupoint=='':
                                string_acupoint = ''
                                for i in range(len(result_illness_list)):
                                    temp_string_illness = result_illness_list[i]
                                    if temp_string_illness !='':
                                        string_illness += temp_string_illness + '、'
                                if string_illness != '':
                                    string_acupoint_judgment=string_illness[-1:]
                                    if string_acupoint_judgment=='、':
                                        string_illness = string_illness[:-1]
                                    string_illness = to_Duplicate_removal_illness(string_illness)
                            else:
                                for i in range(len(result_acupoint_list)):
                                    temp_string_acupoint = to_Duplicate_removal_acupoint_vocabulary(result_acupoint_list[i],dict_convert_acupoint_vocabulary)
                                    temp_string_illness = to_Duplicate_removal_illness(result_illness_list[i])
                                    if temp_string_acupoint != '':
                                        string_acupoint += '（'+ temp_string_illness +'穴位处方）' + temp_string_acupoint + ';' 
                                        if temp_string_illness != '':
                                            string_illness += temp_string_illness + '、'                                         
                                if string_illness != '':
                                    string_acupoint_judgment=string_illness[-1:]
                                    if string_acupoint_judgment=='、':
                                        string_illness = string_illness[:-1]
                                    string_illness = to_Duplicate_removal_illness(string_illness)                       
                        else:
                            for i in range(len(result_acupoint_list)):
                                temp_string_acupoint = to_Duplicate_removal_acupoint_vocabulary(result_acupoint_list[i],dict_convert_acupoint_vocabulary)
                                temp_string_illness = to_Duplicate_removal_illness(result_illness_list[i])
                                if temp_string_acupoint != '':
                                    string_acupoint += '（'+ temp_string_illness +'穴位处方）' + temp_string_acupoint + ';' 
                                    if temp_string_illness != '':
                                        string_illness += temp_string_illness + '、'                                         
                                    
                            if string_illness != '':
                                string_acupoint_judgment=string_illness[-1:]
                                if string_acupoint_judgment=='、':
                                    string_illness = string_illness[:-1]
                                string_illness = to_Duplicate_removal_illness(string_illness)
                else:
                    for i in range(len(result_acupoint_list)):
                        temp_string_acupoint = to_Duplicate_removal_acupoint_vocabulary(result_acupoint_list[i],dict_convert_acupoint_vocabulary)
                        temp_string_illness = to_Duplicate_removal_illness(result_illness_list[i])
                        if temp_string_acupoint != '':
                            string_acupoint += '（'+ temp_string_illness +'穴位处方）' + temp_string_acupoint + ';' 
                            if temp_string_illness != '':
                                string_illness += temp_string_illness + '、'                            
                    if string_illness != '':
                        string_acupoint_judgment=string_illness[-1:]
                        if string_acupoint_judgment=='、':
                            string_illness = string_illness[:-1]
                    string_illness = to_Duplicate_removal_illness(string_illness)
            else:
                string_illness = to_Duplicate_removal_illness(string_illness)
                result_dict = information_extraction_Case_report_OneResult(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                temp_acupoint = result_dict['result_acupoint']
                if temp_acupoint=='':
                    string_acupoint = temp_acupoint
                else:
                    string_acupoint = to_Duplicate_removal_acupoint_vocabulary(temp_acupoint,dict_convert_acupoint_vocabulary)
                if string_acupoint =='':
                    list_string_illness= string_illness.split('、')  
                    string_illness_part =list_string_illness[0]
                    result_dict = information_extraction_Case_report_OneResult_normal(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title,string_illness_part)
                    temp_acupoint = result_dict['result_acupoint']
                    if temp_acupoint=='':
                        string_acupoint = temp_acupoint
                    else:
                        string_acupoint = to_Duplicate_removal_acupoint_vocabulary(temp_acupoint,dict_convert_acupoint_vocabulary)
                    if string_acupoint =='':
                        result_dict = information_extraction_Case_report_OneResult_normal2(inpath,dict_acupoint_vocabulary_summary,illness_dict,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused,string_title)
                        temp_acupoint = result_dict['result_acupoint']
                        if temp_acupoint=='':
                            string_acupoint = temp_acupoint
                        else:
                            string_acupoint = to_Duplicate_removal_acupoint_vocabulary(temp_acupoint,dict_convert_acupoint_vocabulary)
                if string_acupoint!='':
                    string_acupoint = '（'+ string_illness +'穴位处方）' + string_acupoint + ';' 
            pprint('主函数正文string_acupoint='+ string_acupoint +'\n')
            if (string_acupoint==''):
                string_acupoint = get_acupoint_vocabulary_in_sentence(dict_acupoint_vocabulary_summary,string_title,dict_convert_wrong,dict_convert_combination,dict_acupoint_vocabulary_confused)
                if len(string_acupoint)>0:
                    string_acupoint = to_Duplicate_removal_acupoint_vocabulary(string_acupoint,dict_convert_acupoint_vocabulary) 
                    string_acupoint = '（'+ string_illness +'穴位处方）' + string_acupoint + ';' 
            fsock_out.write(string_number +','+ string_title +','+ string_classification_num +','+ string_classification_name +','+ string_illness +','+ string_acupoint + "\n")
    fsock_out.close()
    
    

if __name__ == "__main__":
    #---------------------------------------------------------------------
    #Classification
    #---------------------------------------------------------------------
    filepath = u'E:/tools/Classification/'
    outname_train = u'train_data_cut.txt'
    outname_listname_train = u'train_data_listname.txt'
    outname_test = u'test_data_cut.txt'
    outname_listname_test = u'test_data_listname.txt'
    filepath_sub_PAPER = u'E:/tools/InformationExtraction/All_TXT_PAPER/'
    To_process_classification_text(filepath,outname_train,outname_listname_train,outname_test,outname_listname_test,filepath_sub_PAPER)

    stopwords_path="Dict/Dict_StopWords.txt"
    filename_test_data_listname="test_data_listname.txt"
    filename_test_data_cut="test_data_cut.txt"
    filename_train_data_cut="train_data_cut.txt"
    file_out = u'E:/tools/Classification/classification_out.csv'
    To_Get_Classified_files(stopwords_path,filename_test_data_listname,filename_test_data_cut,filename_train_data_cut,file_out)
    

    #---------------------------------------------------------------------
    #Information Extraction
    #---------------------------------------------------------------------
    list_filename_all=get_file_name_and_classificated(u'E:/tools/Classification/classification_out.csv')
    filepath =  u'E:/tools/InformationExtraction/All_TXT_PAPER/'
    file_out_name = u'E:/tools/InformationExtraction/IE_out.csv'
    To_Get_information_extraction_files(list_filename_all,filepath,file_out_name)
    
    

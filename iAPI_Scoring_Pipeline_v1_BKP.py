#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from string import punctuation
from rake_nltk import Rake
import joblib

#import snow_ticket as snow_ticket
#import snowflake_connection as sc
import dateutil.parser
#import Logz_connection as log

## Declare file paths & variables :
input_files_path = "input_files"
output_files_path = "output_files"
TFIDF_VECTORIZER_FILE_PATH_Impact_iAPI = '/tfidf_impact_iAPI.pkl'
TFIDF_VECTORIZER_FILE_PATH_Urgency_iAPI = '/tfidf_urgency_iAPI.pkl'
IMPACT_MODEL_iAPI_PATH = '/Impact_model_iAPI.pkl'
URGENCY_MODEL_iAPI_PATH = '/Urgency_model_iAPI.pkl'
iAPI_new_logs_file_name = '/iAPI_Experience Layer_19thFeb_to_25thFeb_Prod_Error_Warn_Exception.csv'              
existing_patterns_impact_urgency_labels_file_name = "/iAPI_Patterns_Impact_Urgency_Pritority_labels.csv"  
Impact_Urgency_Priority_Mapping = "/Impact_Urgency_Priority_mapping.csv"

N_WORDS = 6
NUM_WORDS = N_WORDS
LOG_MESSAGE= 'message'
PATTERN = 'PATTERN'
TIMESTAMP = '@timestamp'
# MESSAGE_ONLY = 'message_only'
MESSAGE_ONLY = 'message_only_1'
CLEAN_MSG = 'clean_msg'
IMPACT = 'IMPACT'
URGENCY = 'URGENCY'
PRIORITY = 'PRIORITY'
LOG_LEVEL = 'logLevel'
THREAD = 'thread'
CLASS_NAME_METHOD_NAME = 'class_name_method_name'

tags_list = ['status', 'statusMessage', 'resultCode', 'resultText']

LOGZIO_PROJECT_NAME = 'Wireless Revenue Management Prod'
APPLICATION_NAME = 'iAPI'
ENV_NAME = 'PRODUCTION'
LOGZIO_PROJECT = 'LOGZIO_PROJECT'
APPLICATION = 'APPLICATION'
STATUS = 'STATUS'
PENDING_REVIEW = 'PENDING FOR REVIEW'
ENVIRONMENT = 'ENVIRONMENT'
LOG_MESSAGE_TIMESTAMP= 'LOG_MESSAGE_TIMESTAMP'
NEW = 'NEW'
EXISTING = 'EXISTING'
APP_NAME = 'iAPI Prod'
TARGET_SYSTEM_NAME = 'iAPI'
#TARGET_URI_NAME = CLASS_NAME_METHOD_NAME
TARGET_SYSTEM = 'TARGET_SYSTEM'
TARGET_URI = 'TARGET_URI'
#PENDING_REVIEW = 'PENDING_REVIEW'


### Identify Parseable & Non-Parseable log messages:
def identify_parseable_unparseable_logs(df_latest_iAPI_new_logs):
#     print("Inside identify_parseable_unparseable_logs()")
    df_latest_iAPI_segmented = pd.DataFrame()
    excluded_messages_df = pd.DataFrame(columns=df_latest_iAPI_new_logs.columns)
    regex_for_msg = re.compile(r'(?P<time>[\d\-:\s\.]+)\s(?P<LEVEL>\w+).*\-+\s+\[(?P<thread>[\s\w\-]+)\]\s+(?P<class>[\.\w\$]+)\s*:\s*(?P<message>.*)')
#     regex_for_msg = re.compile(r'(?P<time>[\d\-:\s\.]+)\s(?P<LEVEL>\w+).*\-+\s+\[(?P<thread>[\s\w\-]+)\]\s+(?P<class>[\.\w]+)\s*:\s*(?P<message_1>[\w\s\&]*):*(?P<message_2>.*)')
    logLevel_all = []
    timestamp_all = []
    thread_all = []
    class_name_method_name_all = []
    message_only_all = []
    excluded_message_all = []
    j=0
    k=0
    for i in range(df_latest_iAPI_new_logs.shape[0]):
#         regex_for_msg = re.compile(r'(?P<time>[\d\-:\s\.]+)\s(?P<LEVEL>\w+).*\-+\s+\[(?P<thread>[\s\w\-]+)\]\s+(?P<class>[\.\w]+)\s*:\s*(?P<message>.*)')    
        msg = df_latest_iAPI_new_logs[LOG_MESSAGE][i]
        msg_segments = regex_for_msg.search(msg)
        try:
            timestamp, logLevel, thread, class_name_method_name, message_only = msg_segments.groups()
            logLevel_all.append(logLevel)
            timestamp_all.append(timestamp)
            thread_all.append(thread)
            class_name_method_name_all.append(class_name_method_name)
            message_only_all.append(message_only)
            results_temp = pd.DataFrame()
            results_temp.loc[k, LOG_MESSAGE] = df_latest_iAPI_new_logs.iloc[i][LOG_MESSAGE]
            results_temp.loc[k, TIMESTAMP] = df_latest_iAPI_new_logs.iloc[i][TIMESTAMP]            
            results_temp.loc[k,MESSAGE_ONLY] = message_only
            results_temp.loc[k,LOG_LEVEL] = logLevel
            results_temp.loc[k,'timestamp'] = timestamp
            results_temp.loc[k,THREAD] = thread
            results_temp.loc[k,CLASS_NAME_METHOD_NAME] = class_name_method_name
            df_latest_iAPI_segmented = pd.concat([df_latest_iAPI_segmented, results_temp])
            k= k+1
        except AttributeError:
#             print("can't make a group for msg : ", msg)
#             excluded_message_all.append(msg)
            excluded_messages_df.loc[j]  = df_latest_iAPI_new_logs.iloc[i]
            j=j+1
    
    print("Number of records not following the standard format (Unparseable logs) : ",excluded_messages_df.shape[0])
    print("Number of records following the standard format & hence being processed ahead (Parseable logs) " ,df_latest_iAPI_segmented.shape[0])

    return df_latest_iAPI_segmented,excluded_messages_df

# df_latest_iAPI_segmented,excluded_messages_df = identify_parseable_unparseable_logs(df_latest_iAPI_new_logs) 
# df_latest_iAPI_segmented.to_excel("df_latest_iAPI_segmented_v5.xlsx")


# ### Step 1: Get the well defined Errors & Exceptions :
def extract_well_defined_errors_exceptions(df_latest_iAPI_segmented): 
    """
    Extracting the well-defined erros & exceptions from the log messages

    Generates:
        results (pandas.DataFrame) : Returns a dataframe with the well-defined errors/exceptions
        results_others (pandas.DataFrame) : Returns a dataframe without the well-defined errors/exceptions
        unique_patterns_from_step_1 : Returns a list with the well-defined errors/exceptions
    """  
    print("\n Extracting well-defined errors & exceptions from iAPI log messages...")
    results = pd.DataFrame()
    results_others = pd.DataFrame(columns=df_latest_iAPI_segmented.columns)
    j=0
    k=0
    errors_list = []

    for i in range(len(df_latest_iAPI_segmented[LOG_MESSAGE].tolist())):
        errors = re.findall("""(?x) # Verbose mode
            (?:\w+)             # Match one or more word characters
            (?:Error|Exception) # Match 'Error' or 'Exception':  
            """, df_latest_iAPI_segmented[LOG_MESSAGE].iloc[i])
        errors_list.append(errors)
        if errors == []:
            results_others.loc[j]  = df_latest_iAPI_segmented.iloc[i]
            j = j+1
        else:
            results_temp = pd.DataFrame()
            results_temp.loc[k, PATTERN] = errors[0]
            results_temp.loc[k, LOG_MESSAGE] = df_latest_iAPI_segmented.iloc[i][LOG_MESSAGE]
            results_temp.loc[k, TIMESTAMP] = df_latest_iAPI_segmented.iloc[i][TIMESTAMP]
            results_temp.loc[k,MESSAGE_ONLY] = df_latest_iAPI_segmented.iloc[i][MESSAGE_ONLY]
            results_temp.loc[k,LOG_LEVEL] = df_latest_iAPI_segmented.iloc[i][LOG_LEVEL]
            results_temp.loc[k,THREAD] = df_latest_iAPI_segmented.iloc[i][THREAD]
            results_temp.loc[k,CLASS_NAME_METHOD_NAME] = df_latest_iAPI_segmented.iloc[i][CLASS_NAME_METHOD_NAME]
            
            k=k+1
            results = pd.concat([results, results_temp])
      
    errors_list_all = [item for sublist in errors_list for item in sublist]
    unique_errors_pattern = set(errors_list_all)
    unique_patterns_from_step_1 = list(unique_errors_pattern)
    patterns_from_step_1_df = results.copy()
    patterns_from_step_1_df[PATTERN] = patterns_from_step_1_df[PATTERN].str.rstrip()
    print("No. of well-defined errors : ", len(unique_patterns_from_step_1))
    print("Total No. of records processed in Step 1: {}".format(df_latest_iAPI_segmented.shape[0]))
    print("No. of records with well-defined error name : {}".format(results.shape[0]))
    #print("Results shape -> {}".format(results.shape))
    print("No. of records going to Step-2 -> {}".format(results_others.shape[0]))
    
    return unique_patterns_from_step_1, results_others,patterns_from_step_1_df

# unique_patterns_from_step_1, results_others,patterns_from_step_1_df =  extract_well_defined_errors_exceptions(df_latest_iAPI_segmented)


# ### message splitting for further steps :
# #### Step (1) :
# Split the text at '%c{10} [%M] -  class name followed by method name' .
# Get the text following %c{10} [%M]

def get_msgs_after_split_at_class_method(results_others):
#     print("Inside get_msgs_after_split_at_class_method()")
    class_method_list = results_others['class_name_method_name'].tolist()    
    df_for_step_2 = pd.DataFrame()
    excluded_logs_without_class_method_df = pd.DataFrame(columns=results_others.columns)
    log_msg_to_process = []
    excluded_logs_without_class_method = []
    j=0
    k=0
    for i in range(results_others.shape[0]):    
        test_str = results_others[LOG_MESSAGE][i]
#         class_method_test_str = results_others['class_name_method_name'][i]    
        temp = re.split(rf"({'|'.join(class_method_list)})", test_str)
        
#         class_method_test_str = results_others[CLASS_NAME_METHOD_NAME][i]        
#         temp = re.split(rf"({'|'.join(class_method_list)})", test_str)
        res = [ele for ele in temp if ele]
        if len(res)==3:
            log_msg_to_process.append(res[2])
            results_temp = pd.DataFrame()
            results_temp.loc[j,LOG_MESSAGE] = results_others.iloc[i][LOG_MESSAGE]
            results_temp.loc[j,TIMESTAMP] = results_others.iloc[i][TIMESTAMP] 
            results_temp.loc[j,MESSAGE_ONLY] = results_others.iloc[i][MESSAGE_ONLY]
            results_temp.loc[j,LOG_LEVEL] = results_others.iloc[i][LOG_LEVEL]
            results_temp.loc[j,THREAD] = results_others.iloc[i][THREAD]
            results_temp.loc[j,CLASS_NAME_METHOD_NAME] = results_others.iloc[i][CLASS_NAME_METHOD_NAME]            
            df_for_step_2 = pd.concat([df_for_step_2, results_temp]) 
            j = j+1
        else:
            excluded_logs_without_class_method.append(test_str)
            excluded_logs_without_class_method_df.loc[k] = results_others.iloc[i]
            k = k+1
            
    print("Logs to process in further Steps- 2,3,4 : ",df_for_step_2.shape[0])
    print("Logs not having className & methodName in log structure for split, hence excluded : ",excluded_logs_without_class_method_df.shape[0])
            
    return df_for_step_2,excluded_logs_without_class_method_df
    
# df_for_step_2,excluded_logs_without_class_method_df = get_msgs_after_split_at_class_method(results_others)
# print(df_for_step_2.shape)
# df_for_step_2.head(2)


# ### Step 2 :
# Process log_msg_to_process (which have any tags ) to get patterns from msg
# tags_list = ['status', 'statusMessage', 'resultCode', 'resultText']
def extract_patterns_from_msgs_with_tags(df_for_step_2, tags_list):
    excluded_logs_without_tags = []
    patterns_from_step_2 = []
    excluded_logs_without_tags_df = pd.DataFrame(columns=df_for_step_2.columns)
    patterns_from_step_2_df = pd.DataFrame()
    j=0
    k=0    
    
    for i in range(df_for_step_2.shape[0]):     
        test_str = df_for_step_2[MESSAGE_ONLY][i]
        temp = re.split(rf"({'|'.join(tags_list)})", test_str)
        res = [ele for ele in temp if ele]
        if len(res)>=3:
            pattern = res[0]
            pattern = re.sub(r'[^\w\s]', lambda m: "&" if m.group(0) == "&" else "", pattern)
            patterns_from_step_2.append(pattern) 
            results_temp = pd.DataFrame()
            results_temp.loc[j,PATTERN] = pattern
            results_temp.loc[j,LOG_MESSAGE] = df_for_step_2.iloc[i][LOG_MESSAGE]
            results_temp.loc[j,TIMESTAMP] = df_for_step_2.iloc[i][TIMESTAMP]
            results_temp.loc[j,MESSAGE_ONLY] = df_for_step_2.iloc[i][MESSAGE_ONLY]
            results_temp.loc[j,LOG_LEVEL] = df_for_step_2.iloc[i][LOG_LEVEL]
            results_temp.loc[j,THREAD] = df_for_step_2.iloc[i][THREAD]
            results_temp.loc[j,CLASS_NAME_METHOD_NAME] = df_for_step_2.iloc[i][CLASS_NAME_METHOD_NAME]             
            patterns_from_step_2_df = pd.concat([patterns_from_step_2_df, results_temp])  
            patterns_from_step_2_df[PATTERN] = patterns_from_step_2_df[PATTERN].str.rstrip()            
            j = j+1            
        else:
            excluded_logs_without_tags.append(test_str) 
            excluded_logs_without_tags_df.loc[k] = df_for_step_2.iloc[i]
            k=k+1

    patterns_from_step_2_new = list(set(patterns_from_step_2))
#     patterns_from_step_2_modified = [re.sub(r'[^\w\s]', '', test_str) for test_str in patterns_from_step_2_new]

    # To keep '&' in the pattern and remove rest of the punctiona marks
#     patterns_from_step_2_modified = [re.sub(r'[^\w\s]', lambda m: "&" if m.group(0) == "&" else "", test_str) for test_str in patterns_from_step_2_new]
    
    unique_patterns_from_step_2 = list(set(patterns_from_step_2_new))

    print("No. of patterns from step-2: ", len(unique_patterns_from_step_2))
    print("Total No. of records processed in Step 2: {}".format(df_for_step_2.shape[0]))
    print("No. of records with patterns in step-2 : {}".format(patterns_from_step_2_df.shape[0]))
    print("No. of records going to Step-3 -> {}".format(excluded_logs_without_tags_df.shape[0]))
    
    return patterns_from_step_2_df,unique_patterns_from_step_2,excluded_logs_without_tags,excluded_logs_without_tags_df
            
# df_for_step_2[MESSAGE_ONLY] = df_for_step_2[LOG_MESSAGE].apply(lambda x : extract_msg_only(x))
# patterns_from_step_2_df,unique_patterns_from_step_2,excluded_logs_without_tags,excluded_logs_without_tags_df = extract_patterns_from_msgs_with_tags(df_for_step_2, tags_list)


# ### Step 3 :
# - Process excluded_logs_without_tags (which have any tags ) to extract patterns from short msgs:
# - msgs with length<=6 itself becomes a pattern
def extract_patterns_from_short_msgs(excluded_logs_without_tags_df,NUM_WORDS):
    
    df_for_step_3 = excluded_logs_without_tags_df.copy()
    patterns_from_step_3 = []
    df_for_step_4 = pd.DataFrame(columns=df_for_step_3.columns)
    patterns_from_step_3_df = pd.DataFrame()
    long_msgs = []
    j=0
    k=0        
    for i in range(df_for_step_3.shape[0]):  # log in df_for_step_3[MESSAGE_ONLY].tolist():
        log = df_for_step_3[MESSAGE_ONLY][i]     
        if len(log.split())<=NUM_WORDS:
            pattern = log
            patterns_from_step_3.append(pattern) 
            results_temp = pd.DataFrame()
            results_temp.loc[j,PATTERN] = pattern
            results_temp.loc[j,LOG_MESSAGE] = df_for_step_3.iloc[i][LOG_MESSAGE]
            results_temp.loc[j,TIMESTAMP] = df_for_step_3.iloc[i][TIMESTAMP]
            results_temp.loc[j,MESSAGE_ONLY] = df_for_step_3.iloc[i][MESSAGE_ONLY]
            results_temp.loc[j,LOG_LEVEL] = df_for_step_3.iloc[i][LOG_LEVEL]
            results_temp.loc[j,THREAD] = df_for_step_3.iloc[i][THREAD]
            results_temp.loc[j,CLASS_NAME_METHOD_NAME] = df_for_step_3.iloc[i][CLASS_NAME_METHOD_NAME]             
            patterns_from_step_3_df = pd.concat([patterns_from_step_3_df, results_temp])      
            patterns_from_step_3_df[PATTERN] = patterns_from_step_3_df[PATTERN].str.rstrip()
            j = j+1            
        else:
            long_msgs.append(log) 
            df_for_step_4.loc[k] = df_for_step_3.iloc[i]
            k=k+1 
            
    unique_patterns_from_short_msgs = list(set(patterns_from_step_3))
    unique_patterns_from_step_3 = unique_patterns_from_short_msgs
    unique_patterns_from_step_3 = [t.replace('\\n"',"") for t in unique_patterns_from_step_3]
    print("Number of records processed in Step 3 : ", df_for_step_3.shape[0])
    print("Number of patterns in step 3 from short msgs : ",len(unique_patterns_from_step_3))
    print("Number of records with patterns from short msgs : ", patterns_from_step_3_df.shape[0])
    print("Number of long msgs going in Step 4 : ", df_for_step_4.shape[0])    
    
    return unique_patterns_from_step_3, patterns_from_step_3_df, df_for_step_4


# unique_patterns_from_step_3, patterns_from_step_3_df, df_for_step_4 = extract_patterns_from_short_msgs(excluded_logs_without_tags_df,NUM_WORDS)


# ### Step 4 : Process long_msgs to extract patterns using RAKE
stop_words = set(stopwords.words('english')+list(punctuation))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))   # to remove url from the text
    text = re.sub('[^a-zA-Z]', ' ', str(text))   # remove anything except a-z & A-Z
    text = re.sub('[^a-zA-Z]', ' ', str(text))   # remove anything except a-z & A-Z (basically to remove numbers)
    text = re.sub(r'[^\w\s]', '', str(text))   # to remove punctuations from text
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word = w) for w in text]
    text = [item for item in text if len(item) >= 2]    # to consider >=2 alphabet words
    text = ' '.join(text)
    return text

def extract_patterns_from_long_msgs_using_rake(df_for_step_4,NUM_WORDS):
    df_for_step_4[CLEAN_MSG] = [clean_text(text) for text in df_for_step_4[MESSAGE_ONLY]]        
    patterns_from_step_4 = []
    df_for_step_5 = pd.DataFrame(columns=df_for_step_4.columns)
    patterns_from_step_4_df = pd.DataFrame()
    j=0       
    for i in range(df_for_step_4.shape[0]):  
        r=Rake()
        r.extract_keywords_from_text(df_for_step_4[CLEAN_MSG][i])
        pattern = r.get_ranked_phrases()[:1][0]
        pattern_split = pattern.split(" ")[:NUM_WORDS]
        pattern = (" ".join(pattern_split[:]))
        patterns_from_step_4.append(pattern)   
        results_temp = pd.DataFrame()
        results_temp.loc[j,PATTERN] = pattern
        results_temp.loc[j,LOG_MESSAGE] = df_for_step_4.iloc[i][LOG_MESSAGE]
        results_temp.loc[j,TIMESTAMP] = df_for_step_4.iloc[i][TIMESTAMP]
        results_temp.loc[j,MESSAGE_ONLY] = df_for_step_4.iloc[i][MESSAGE_ONLY]
        results_temp.loc[j,LOG_LEVEL] = df_for_step_4.iloc[i][LOG_LEVEL]
        results_temp.loc[j,THREAD] = df_for_step_4.iloc[i][THREAD]
        results_temp.loc[j,CLASS_NAME_METHOD_NAME] = df_for_step_4.iloc[i][CLASS_NAME_METHOD_NAME]        
        patterns_from_step_4_df = pd.concat([patterns_from_step_4_df, results_temp])      
        patterns_from_step_4_df[PATTERN] = patterns_from_step_4_df[PATTERN].str.rstrip()
        j = j+1            

    unique_patterns_from_step_4_set = set(patterns_from_step_4)
    unique_patterns_from_step_4 = list(unique_patterns_from_step_4_set)

    print("Number of records processed in Step 4 : ", df_for_step_4.shape[0])
    print("Number of patterns in step 4 from long msgs : ",len(unique_patterns_from_step_4))
    print("Number of records with patterns from short msgs : ", patterns_from_step_4_df.shape[0])   

    return patterns_from_step_4_df, unique_patterns_from_step_4
    
# patterns_from_step_4_df, unique_patterns_from_step_4 = extract_patterns_from_long_msgs_using_rake(df_for_step_4,NUM_WORDS)


# ### Call all the 4 pattern extraction functions in a single function:
def extract_patterns(df_latest_iAPI_segmented, NUM_WORDS): 

    """
    Combines the output from extract_well_defined_errors_exceptions() , extract_pattern_with_words_errors_exceptions_around_it() and extract_patterns_using_n_words()

    Generates:
        output_df : Returns a dataframe of all the patterns extracted from all the 4 functions mentioned above

    """    
    unique_patterns_from_step_1, results_others,patterns_from_step_1_df =  extract_well_defined_errors_exceptions(df_latest_iAPI_segmented)
    df_for_step_2,excluded_logs_without_class_method_df = get_msgs_after_split_at_class_method(results_others)
#     df_for_step_2[MESSAGE_ONLY] = df_for_step_2[LOG_MESSAGE].apply(lambda x : extract_msg_only(x))
    patterns_from_step_2_df,unique_patterns_from_step_2,excluded_logs_without_tags,excluded_logs_without_tags_df = extract_patterns_from_msgs_with_tags(df_for_step_2, tags_list)
    unique_patterns_from_step_3, patterns_from_step_3_df, df_for_step_4 = extract_patterns_from_short_msgs(excluded_logs_without_tags_df,NUM_WORDS)
    patterns_from_step_4_df, unique_patterns_from_step_4 = extract_patterns_from_long_msgs_using_rake(df_for_step_4,NUM_WORDS)
    
    patterns_step_1 = unique_patterns_from_step_1
    patterns_step_2 = unique_patterns_from_step_2        
    patterns_step_3 = unique_patterns_from_step_3
    patterns_step_4 = unique_patterns_from_step_4
    print("\n Number of patterns in each step:",len(patterns_step_1),len(patterns_step_2),len(patterns_step_3),len(patterns_step_4))
    required_data = list(zip(patterns_step_1,patterns_step_2,patterns_step_3))
    output_df = pd.DataFrame({'Well_defined_Errors_Exceptions': pd.Series(patterns_step_1),
                                   'Patterns_from_msgs_with_tags': pd.Series(patterns_step_2),
                                   'Patterns_from_short_msgs_without_tags': pd.Series(patterns_step_3),
                                   'Patterns_from_long_msgs_without_tags_using_RAKE': pd.Series(patterns_step_4)})

#     output_df.to_csv(output_files_path+"/Output_patterns_from_iAPI_new_logs.csv", index=False)
    output_df.to_csv(output_files_path+"/Output_patterns_from_iAPI_Prod.csv", index=False)
    
    return patterns_from_step_1_df, patterns_from_step_2_df,patterns_from_step_3_df,patterns_from_step_4_df



def identify_logs_with_existing_patterns(df_latest_iAPI_segmented,patterns_from_step_1_df, patterns_from_step_2_df,patterns_from_step_3_df,patterns_from_step_4_df):
    """
    Fetches new logs for iAPI Prod & finds the pattern which it contains, from the existing patterns that we have identified in iteration-4.
    It maps the Impact of the matched pattern to the new log, hence having Impact assigned to the new log msg based on the patterns already identified.

    Generates:
        df_existing_patterns_new_logs :Returns a dataframe having the new logMsgs, its matched pattern & mapped Impact ; for which there is a match with the patterns that we already identified.
        df_keywords_new_logs : Returns a dataframe having the new logMsgs, its matched pattern & mapped Impact ; for which there is a match with the Keywords that we already identified.
        df_logs_with_no_existing_pattern : Returns a dataframe having the new logMsgs ; for which there is NO match with the patterns & the Keywords that we already identified. These are the log Messages which needs to handled separately
        df_consolidated_existing_patterns : Returns a consolidated dataframe having the new logMsgs, its matched pattern & mapped Impact ; for which there is a match with the patterns & the Keywords that we already identified. Basically consolidated dataframe for df_existing_patterns_new_logs & df_keywords_new_logs
    """
    print("patterns_from_step_1_df " ,patterns_from_step_1_df.shape)
    print("patterns_from_step_2_df ",patterns_from_step_2_df.shape)
    print("patterns_from_step_3_df ",patterns_from_step_3_df.shape)
    print("patterns_from_step_4_df ",patterns_from_step_4_df.shape)
    df_new_logs = df_latest_iAPI_segmented.copy()
    print("Number of logs processing : ",df_new_logs.shape[0])
    #df_new_logs.dropna(inplace=True)

    df_existing_impact_urgency_labels = pd.read_csv(input_files_path+existing_patterns_impact_urgency_labels_file_name)   
    #df_existing_impact_urgency_labels = sc.read_from_snowflake(cs, sc.FETCH_ReadyForUse_Patterns) 
    #df_existing_impact_urgency_labels.to_csv(output_files_path+"/df_existing_impact_urgency_labels_from_DB.csv")
    df_existing_impact_urgency_labels=df_existing_impact_urgency_labels[[PATTERN,IMPACT,URGENCY,PRIORITY]]
    #df_existing_impact_urgency_labels[IMPACT] =  pd.to_numeric(df_existing_impact_urgency_labels[IMPACT])
    #df_existing_impact_urgency_labels[URGENCY] =  pd.to_numeric(df_existing_impact_urgency_labels[URGENCY])
    #df_existing_impact_urgency_labels[PRIORITY] =  pd.to_numeric(df_existing_impact_urgency_labels[PRIORITY])
    
    #print("df_existing_impact_urgency_labels.dtpyes : ", df_existing_impact_urgency_labels.dtypes)
    
    #print("df_existing_impact_urgency_labels : ", df_existing_impact_urgency_labels[PATTERN])
    results_step1_step2_step3_step4_df = pd.concat([patterns_from_step_1_df,patterns_from_step_2_df,patterns_from_step_3_df,patterns_from_step_4_df], ignore_index=True)
    print("results_step1_step2_step3_step4_df.shape() : ",results_step1_step2_step3_step4_df.shape)
    #results_step1_step2_step3_step4_df.to_csv(r"output/results_step1_step2_step3_step4_df.csv")

    ## Step 2: find df_existing_patterns_new_logs i.e. logs with existing pattern:
    pattern_impact_dict = {df_existing_impact_urgency_labels.iloc[i][PATTERN]: df_existing_impact_urgency_labels.iloc[i][IMPACT] for i in range(df_existing_impact_urgency_labels.shape[0])}
    #print("pattern_impact_dict : ",pattern_impact_dict)
    pattern_urgency_dict = {df_existing_impact_urgency_labels.iloc[i][PATTERN]: df_existing_impact_urgency_labels.iloc[i][URGENCY] for i in range(df_existing_impact_urgency_labels.shape[0])}
    pattern_priority_dict = {df_existing_impact_urgency_labels.iloc[i][PATTERN]: df_existing_impact_urgency_labels.iloc[i][PRIORITY] for i in range(df_existing_impact_urgency_labels.shape[0])}

    count=0
    patterns_found_list = []
    for new_pattern in results_step1_step2_step3_step4_df[PATTERN].tolist():
        #print("new_pattern : ",new_pattern)
        for pattern in df_existing_impact_urgency_labels[PATTERN].squeeze().tolist():
            if pattern in new_pattern:
                patterns_found_list.append(new_pattern)
                count= count+1

    output_step_2_df = results_step1_step2_step3_step4_df[results_step1_step2_step3_step4_df[PATTERN].isin(patterns_found_list)]
    df_existing_patterns_new_logs = output_step_2_df.copy()
    df_existing_patterns_new_logs.to_csv(output_files_path+"/df_existing_patterns_new_logs.csv")
    df_existing_patterns_new_logs = df_existing_patterns_new_logs.assign(IMPACT = df_existing_patterns_new_logs[PATTERN].map(pattern_impact_dict))
    df_existing_patterns_new_logs = df_existing_patterns_new_logs.assign(URGENCY = df_existing_patterns_new_logs[PATTERN].map(pattern_urgency_dict))
    df_existing_patterns_new_logs = df_existing_patterns_new_logs.assign(PRIORITY = df_existing_patterns_new_logs[PATTERN].map(pattern_priority_dict))
       
    print("Number of Log Messages with existing pattern : ",df_existing_patterns_new_logs.shape[0])
    ## Step 3: Find log messages (newly fetched logs i.e. df_new_logs) which 
    #(a) Are already excluded from excluded keywords 
    #(b) Do not have existing patterns (from iteration 4)
    # New logs with New patterns :
    logs_with_no_existing_pattern = list(set(df_new_logs[LOG_MESSAGE]) - set(df_existing_patterns_new_logs[LOG_MESSAGE]))
      
    new_patterns_list = results_step1_step2_step3_step4_df[results_step1_step2_step3_step4_df[LOG_MESSAGE].isin(logs_with_no_existing_pattern)][PATTERN]
    df_logs_with_no_existing_pattern = results_step1_step2_step3_step4_df[results_step1_step2_step3_step4_df[LOG_MESSAGE].isin(logs_with_no_existing_pattern)]
    df_logs_with_no_existing_pattern[LOGZIO_PROJECT] = LOGZIO_PROJECT_NAME
    df_logs_with_no_existing_pattern[APPLICATION] = APPLICATION_NAME
    df_logs_with_no_existing_pattern[ENVIRONMENT] = ENV_NAME
    df_logs_with_no_existing_pattern[STATUS] = NEW
    df_logs_with_no_existing_pattern[TARGET_SYSTEM] = TARGET_SYSTEM_NAME
    df_logs_with_no_existing_pattern[TARGET_URI] = df_logs_with_no_existing_pattern[CLASS_NAME_METHOD_NAME]
    
    print("Number of Log messages with new patterns : ", df_logs_with_no_existing_pattern.shape[0])    
    df_logs_with_no_existing_pattern.to_csv(output_files_path+"/iAPI_new_logs_with_new_patterns.csv",index=False)  

    # To avoid duplicate entry of already any pattern which already exist in the ATC_PATTERNS table either 'Ready for Use'/'Pending for Review'/
    ATC_STG_PATTERNS_output_iAPI_new_patterns_df = pd.DataFrame(columns = df_logs_with_no_existing_pattern.columns)
    df_all_patterns_iAPI = df_existing_impact_urgency_labels
    for i in range(df_logs_with_no_existing_pattern.shape[0]):
        if not df_logs_with_no_existing_pattern.iloc[i][PATTERN] in (df_all_patterns_iAPI[PATTERN].tolist()):
            ATC_STG_PATTERNS_output_iAPI_new_patterns_df.loc[ATC_STG_PATTERNS_output_iAPI_new_patterns_df.shape[0]] = df_logs_with_no_existing_pattern.iloc[i]

    ATC_STG_PATTERNS_output_iAPI_new_patterns_df = ATC_STG_PATTERNS_output_iAPI_new_patterns_df[[APPLICATION, LOGZIO_PROJECT, ENVIRONMENT, PATTERN, STATUS]].drop_duplicates([PATTERN])
    
    # Step 5: Consolidate results:
    df_consolidated_existing_patterns = pd.concat([ df_existing_patterns_new_logs], ignore_index=True)

    df_consolidated_existing_patterns = df_consolidated_existing_patterns[[TIMESTAMP,LOG_MESSAGE,PATTERN,IMPACT,URGENCY,PRIORITY]]
        
    print("Total log messages with Existing patterns : " ,df_consolidated_existing_patterns.shape[0])
    df_consolidated_existing_patterns[LOGZIO_PROJECT] = LOGZIO_PROJECT_NAME
    df_consolidated_existing_patterns[APPLICATION] = APPLICATION_NAME
    df_consolidated_existing_patterns[ENVIRONMENT] = ENV_NAME
    df_consolidated_existing_patterns[STATUS] = EXISTING
    df_consolidated_existing_patterns[TARGET_SYSTEM] = TARGET_SYSTEM_NAME
    df_consolidated_existing_patterns[TARGET_URI] = df_existing_patterns_new_logs[CLASS_NAME_METHOD_NAME]
            
    df_consolidated_existing_patterns.to_csv(output_files_path+"/iAPI_mapped_impact_urgency_priority_to_new_logs_from_existing_patterns.csv", index=False)
    
    df_consolidated_existing_patterns_insertion = df_consolidated_existing_patterns[[APPLICATION,LOGZIO_PROJECT, ENVIRONMENT,LOG_MESSAGE,PATTERN, TIMESTAMP,TARGET_SYSTEM,TARGET_URI]]  
    df_new_errors_exceptions_insertion = df_logs_with_no_existing_pattern[[APPLICATION,LOGZIO_PROJECT, ENVIRONMENT,LOG_MESSAGE,PATTERN, TIMESTAMP,TARGET_SYSTEM,TARGET_URI]]
 
    #df_consolidated_logs_final = pd.concat([df_consolidated_existing_patterns, df_logs_with_no_existing_pattern], ignore_index=True)    
    #df_consolidated_logs_final = df_consolidated_logs_final[[LOG_MESSAGE, PATTERN, TIMESTAMP]]
    #df_consolidated_logs_final.to_csv(output_files_path+"/df_consolidated_logs_final_iAPI.csv",index=False)

    df_consolidated_logs_final = pd.concat([df_consolidated_existing_patterns_insertion, df_new_errors_exceptions_insertion], ignore_index=True)  
    df_consolidated_logs_final = df_consolidated_logs_final[[APPLICATION, LOGZIO_PROJECT, ENVIRONMENT, LOG_MESSAGE, PATTERN, TIMESTAMP,TARGET_SYSTEM,TARGET_URI]]
    # Insert all new log msgs in ATCSTGLOGMSG :
    df_consolidated_logs_final.rename(columns = {TIMESTAMP:LOG_MESSAGE_TIMESTAMP}, inplace = True)
    
    #sc.write_into_snowflake(cs,df_consolidated_logs_final,sc.INSERT_LOG_MSGS)  
    df_consolidated_logs_final.to_csv(output_files_path+"/df_consolidated_logs_final.csv",index=False)
    
    return df_consolidated_existing_patterns, df_logs_with_no_existing_pattern,ATC_STG_PATTERNS_output_iAPI_new_patterns_df
    

def scoring_new_patterns_for_urgency(df_logs_with_no_existing_pattern): 
    if not df_logs_with_no_existing_pattern.empty:     
        print("Scoring new patterns for Urgency")
        #create_model_for_urgency()
        new_patterns_list = df_logs_with_no_existing_pattern[PATTERN]        
        model2 = joblib.load(output_files_path+URGENCY_MODEL_iAPI_PATH)
        #tfidf_urgency = pickle.load(open(TFIDF_VECTORIZER_FILE_PATH_Urgency_CatServ, 'rb'))
        tfidf_urgency = joblib.load(output_files_path+TFIDF_VECTORIZER_FILE_PATH_Urgency_iAPI)          

        new_patterns_list_tfidf=tfidf_urgency.transform(new_patterns_list)
        new_pattern_tfidf=pd.DataFrame(new_patterns_list_tfidf.toarray(), columns=tfidf_urgency.get_feature_names())        
        predictions = model2.predict(new_pattern_tfidf)
        urgency_list = [1,2,3]
        numbers_list = [1,2,0]
        df_urgency_mapping = pd.DataFrame({'Urgency_label' : urgency_list ,'Urgency_number' :numbers_list })

        Urgency_for_new_patterns = []
        for i in range(len(new_patterns_list)):
            pred_urgency = df_urgency_mapping.loc[df_urgency_mapping['Urgency_number']==np.argmax(predictions[i]),'Urgency_label'].values[0]
            Urgency_for_new_patterns.append(pred_urgency)

        print("Predicted Urgency for new logs : ", Urgency_for_new_patterns)  
        
        return Urgency_for_new_patterns

def scoring_new_patterns_for_impact(df_logs_with_no_existing_pattern): 
    if not df_logs_with_no_existing_pattern.empty:     
        print("Scoring new patterns for Impact")
        #create_model_for_impact()
        new_patterns_list = df_logs_with_no_existing_pattern[PATTERN]        
        model2 = joblib.load(output_files_path+IMPACT_MODEL_iAPI_PATH)
        #tfidf_impact = pickle.load(open(TFIDF_VECTORIZER_FILE_PATH_Impact_CatServ, 'rb'))
        tfidf_impact = joblib.load(output_files_path+TFIDF_VECTORIZER_FILE_PATH_Impact_iAPI)          

        new_patterns_list_tfidf=tfidf_impact.transform(new_patterns_list)
        new_pattern_tfidf=pd.DataFrame(new_patterns_list_tfidf.toarray(), columns=tfidf_impact.get_feature_names())        
        predictions = model2.predict(new_pattern_tfidf)
        impact_list = [1,2,3]
        numbers_list = [1,2,0]
        df_impact_mapping = pd.DataFrame({'Impact_label' : impact_list ,'Impact_number' :numbers_list })

        Impact_for_new_patterns = []
        for i in range(len(new_patterns_list)):
            pred_impact = df_impact_mapping.loc[df_impact_mapping['Impact_number']==np.argmax(predictions[i]),'Impact_label'].values[0]
            Impact_for_new_patterns.append(pred_impact)

        print("Predicted Impact for new logs : ", Impact_for_new_patterns)  
        
        return Impact_for_new_patterns

def assign_priority(df_logs_with_no_existing_pattern, ATC_STG_PATTERNS_output_iAPI_new_patterns_df):
    if not df_logs_with_no_existing_pattern.empty:     
        impact_urgency_priority_mapping_df = pd.read_csv(input_files_path+Impact_Urgency_Priority_Mapping)
        #print(impact_urgency_priority_mapping_df)
        new_logs_impact = Impact_for_new_patterns
        new_logs_urgency = Urgency_for_new_patterns
        new_patterns = df_logs_with_no_existing_pattern[PATTERN].values
        new_patterns_logs = df_logs_with_no_existing_pattern[LOG_MESSAGE].values
        new_patterns_timestamp = df_logs_with_no_existing_pattern[TIMESTAMP].values

        Priority_for_new_patterns = []
        for i in range(len(new_patterns)):
            pred_priority = impact_urgency_priority_mapping_df.loc[(impact_urgency_priority_mapping_df[IMPACT]==new_logs_impact[i]) & (impact_urgency_priority_mapping_df[URGENCY]==new_logs_urgency[i]),PRIORITY].values[0]
            Priority_for_new_patterns.append(pred_priority)

        new_patterns_mapped_impact_urgency_priority_df = pd.DataFrame({TIMESTAMP:new_patterns_timestamp ,LOG_MESSAGE: new_patterns_logs, PATTERN: new_patterns ,IMPACT: new_logs_impact, URGENCY: new_logs_urgency, PRIORITY :Priority_for_new_patterns})
        print("Mapped Impact, Urgency & Priority to new patterns : ", new_patterns_mapped_impact_urgency_priority_df)

        new_patterns_mapped_impact_urgency_priority_df[LOGZIO_PROJECT] = LOGZIO_PROJECT_NAME
        new_patterns_mapped_impact_urgency_priority_df[APPLICATION] = APPLICATION_NAME
        new_patterns_mapped_impact_urgency_priority_df[ENVIRONMENT] = ENV_NAME
        new_patterns_mapped_impact_urgency_priority_df[STATUS] = PENDING_REVIEW
        new_patterns_mapped_impact_urgency_priority_df_copy = new_patterns_mapped_impact_urgency_priority_df.copy()  #Required for Business output
        new_patterns_mapped_impact_urgency_priority_df = new_patterns_mapped_impact_urgency_priority_df[[PATTERN, IMPACT, URGENCY, PRIORITY, STATUS, LOGZIO_PROJECT, APPLICATION, ENVIRONMENT, LOG_MESSAGE, TIMESTAMP]]

        #new_patterns_mapped_impact_urgency_priority_df.to_csv("output/new_patterns_mapped_impact_urgency_priority_df_copy.csv", index=False)

        # Insert New patterns to ATC_STG_PATTERNS tbl having mapped impact, urgency & priority      
        ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df  = new_patterns_mapped_impact_urgency_priority_df.drop_duplicates([PATTERN])  # this will have DISTINCT new patterns 
        #ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df.drop([TIMESTAMP,LOG_MESSAGE],axis=1,inplace=True)
        ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df = ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df[[APPLICATION,LOGZIO_PROJECT, ENVIRONMENT, PATTERN, STATUS, IMPACT, URGENCY, PRIORITY]]

        #To avoid duplicate entry of same pattern in ATC_STG_PATTERNS table:
        ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df = ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df.loc[ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df[PATTERN].isin(ATC_STG_PATTERNS_output_iAPI_new_patterns_df[PATTERN])]
        # Insert new errors/exceptions/patterns in ATCSTGPATTERNS with mapped impact, urgency, priority & status
        #sc.write_into_snowflake(cs,ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df,sc.INSERT_NEW_PATTERNS)
        #snowflakeData.insert_to_ATCSTGPATTERNS_tbl_iAPI(ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df)                
        ATC_STG_PATTERNS_output_iAPI_new_patterns_scored_df.to_csv(output_files_path+"/ATC_STG_PATTERNS_output_iAPI_new_patterns_scored.csv",index=False)

    else :
        new_patterns_mapped_impact_urgency_priority_df = pd.DataFrame(columns=[PATTERN, IMPACT, URGENCY, PRIORITY, STATUS, LOGZIO_PROJECT, APPLICATION, ENVIRONMENT, TIMESTAMP, LOG_MESSAGE])
        
    return new_patterns_mapped_impact_urgency_priority_df
    
def create_ticket(new_patterns_mapped_impact_urgency_priority_df,df_consolidated_existing_patterns):
    #all_logs_priority_df =pd.concat([new_patterns_mapped_impact_urgency_priority_df, df_consolidated_existing_patterns])    
    all_logs_priority_df = pd.concat([new_patterns_mapped_impact_urgency_priority_df,df_consolidated_existing_patterns]) 
    final_ticket_logs_df = all_logs_priority_df.loc[all_logs_priority_df[PRIORITY].isin(["1","2"])]
    final_ticket_logs_df.to_csv("output_files/final_ticket_logs_df.csv",index=False)

    final_ticket_logs_df_unique_logs = final_ticket_logs_df.drop_duplicates([LOG_MESSAGE])  # for unique log msgs to go as tickets in SNOW
    #print("Number of records in final_ticket_logs_df_unique_logs : ",final_ticket_logs_df_unique_logs.shape[0])
    final_ticket_logs_df_unique_logs.to_csv("output_files/final_ticket_logs_df_unique_logs.csv",index=False)    
    
    return final_ticket_logs_df_unique_logs
    

if __name__ == '__main__':

    #ctx = sc.establish_connection()
    #cs = sc.get_cursor(ctx) 
    print("Fetching the latest iAPI logs from csv file...")
    df_latest_iAPI_new_logs = pd.read_csv(input_files_path+iAPI_new_logs_file_name)
    #print("Fetching the latest iAPI logs from LogZ...")
#     message = []
#     timestamp = []
#     message = []
#     timestamp = []
#     app_name = APP_NAME
#     logs_list, first_fetch_count, n_iter, data_scroll = log.get_first_set_logs_from_logz(app_name)
#     df_latest_iAPI_new_logs = pd.DataFrame(columns = [LOG_MESSAGE])
#     #    df_latest_iAPI_logs = pd.DataFrame(columns  = [iAPI_LOG_MESSAGE, iAPI_TIMESTAMP]) 
#     #print("logs_list : ",logs_list)      
#     #df_latest_iAPI_logs[LOG_MESSAGE] = logs_list
#     #df_latest_iAPI_logs.to_csv("latest_iAPI_logs_28thFeb.csv")
#     #print("length :", len(logs_list))      
      
#     second_fetch_count = 0
#     logs_list1 = []
#     for _ in range(n_iter):
#         logs_list1 = log.get_next_set_logs_from_logz(app_name, data_scroll)
#         second_fetch_count += len(logs_list1)
#         #print("first_fetch_count : ",first_fetch_count)
#     print("total logs fetched -> {}".format(first_fetch_count+second_fetch_count))
#     df_latest_iAPI_new_logs[LOG_MESSAGE] = logs_list + logs_list1
#     for i in range(len(df_latest_iAPI_new_logs)):
#         message.append(df_latest_iAPI_new_logs[LOG_MESSAGE].values[i]['_source']['message'])
#         timestamp.append(df_latest_iAPI_new_logs[LOG_MESSAGE].values[i]['_source']['@timestamp'])
#     df_latest_iAPI_new_logs[LOG_MESSAGE] = message
#     df_latest_iAPI_new_logs[TIMESTAMP] = timestamp
#     df_latest_iAPI_new_logs = df_latest_iAPI_new_logs[[TIMESTAMP,LOG_MESSAGE]] 

    df_latest_iAPI_new_logs[TIMESTAMP] = [(dateutil.parser.parse(time_str[1:-5])).strftime('%Y-%m-%dT%H:%M:%S.%f') for time_str in df_latest_iAPI_new_logs[TIMESTAMP].to_list() ]
    df_latest_iAPI_new_logs.rename(columns = {LOG_MESSAGE:LOG_MESSAGE}, inplace = True)
    #df_latest_iAPI_new_logs = df_latest_iAPI_new_logs.copy()    

    # Identify logs which follows (Parseable)/ not follows (Unparseable) the standard log structure in iAPI :
    df_latest_iAPI_segmented,excluded_messages_df = identify_parseable_unparseable_logs(df_latest_iAPI_new_logs)
    
    # Extracting well-defined errors/exceptions and patterns from the new log messages.
    patterns_from_step_1_df, patterns_from_step_2_df,patterns_from_step_3_df,patterns_from_step_4_df = extract_patterns(df_latest_iAPI_segmented, NUM_WORDS)
    print("Extracted Patterns Successfully.\n=====================================")   

    # Extracting log msgs with existing patterns
    df_consolidated_existing_patterns, df_logs_with_no_existing_pattern, ATC_STG_PATTERNS_output_iAPI_new_patterns_df = identify_logs_with_existing_patterns(df_latest_iAPI_segmented, patterns_from_step_1_df, patterns_from_step_2_df,patterns_from_step_3_df,patterns_from_step_4_df)
    print("Extracted log msgs with existing patterns & new patterns from new log msgs.\n===========================")  
    
    # Scoring new patterns for Urgency :
    Urgency_for_new_patterns = scoring_new_patterns_for_urgency(df_logs_with_no_existing_pattern)
    Impact_for_new_patterns = scoring_new_patterns_for_impact(df_logs_with_no_existing_pattern)
    new_patterns_mapped_impact_urgency_priority_df = assign_priority(df_logs_with_no_existing_pattern, ATC_STG_PATTERNS_output_iAPI_new_patterns_df)
    
    final_ticket_logs_df_unique_logs = create_ticket(new_patterns_mapped_impact_urgency_priority_df,df_consolidated_existing_patterns)
    #snow_ticket.create_snow_incidents(final_ticket_logs_df_unique_logs)
    
    #ctx.close()

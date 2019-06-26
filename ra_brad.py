#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:44:24 2019

@author: MOOSE
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,recall_score,precision_score, average_precision_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from xgboost import plot_importance
from termcolor import colored
import random
import time
import datetime as dt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from operator import itemgetter


# Name of your working Folder which contains all the .csvs, .py necessary
dir_Folder= 'IM Residency Predictive Analytics'

# Set to true to remove step scores from analyses - Also removes step 2 scores
remove_step_scores = False

# Set to integer to set cutoff for step 1 score - Also removes step 2 scores
step_score_predictor_cutoff = 220

# Set test_size as fraction of the total # of samples in 2017 data, and nsample from 2018 data
test_size=0.2
validation_size=0.2

included_variables = {'AAMC ID': 'ID',
                      'Medical Education or Training Interrupted': 'education_interruption',
                      'ACLS': 'ACLS',
                      'BLS': 'BLS',
                      'Board Certification': 'board_certified',
                      'Malpractice Cases Pending': 'malpractice_pending',
                      'Medical Licensure Problem': 'licensure_problem',
                      'PALS': 'PALS',
                      'Misdemeanor Conviction': 'misdemeanor',
                      # 'Alpha Omega Alpha': 'aoa_school',
                      'Alpha Omega Alpha (Yes/No)': 'aoa_recipient',
                      'Citizenship': 'citizenship',
                      # 'Contact City': 'contact_city',
                      #'Contact Country': 'contact_country',
                      #'Contact State': 'contact_state',
                      # 'Contact Zip': 'contact_zip',
                      'Date of Birth': 'dob',
                      'Gender': 'gender',
                      # 'Gold Humanism Honor Society': 'gold_school',
                      'Gold Humanism Honor Society (Yes/No)': 'gold_recipient',
                      'Military Service Obligation': 'military_obligation',
                      'Participating as a Couple in NRMP': 'couples_matching',
                      # 'Permanent City': 'permanent_city',
                      #'Permanent Country': 'permanent_country',
                      #'Permanent State': 'permanent_state',
                      # 'Permanent Zip': 'permanent_zip',
                      'Self Identify': 'race',
                      # 'Sigma Sigma Phi': 'sigma_school',
                      'Sigma Sigma Phi (Yes/No)': 'sigma_recipient',
                      'US or Canadian Applicant': 'us_or_canadian',
                      'Visa Sponsorship Needed': 'visa_need',
                      # 'Application Reviewed': 'app_reviewed',
                      #'Withdrawn by Applicant': 'app_withdrawn_stud',
                      # 'Withdrawn by Program': 'app_withdrawn_prog',
                      # 'On Hold': 'on_hold',
                      # 'Average Document Score': 'avg_doc_score',
                      'ECFMG Certification Received': 'ecfmg_cert_received',
                      'CSA Exam Status': 'csa_received',
                      'ECFMG Certified': 'ecfmg_cert',
                      'Medical School Transcript Received': 'transcript_received',
                      'MSPE Received': 'mspe_received',
                      'Personal Statement Received': 'ps_received',
                      'Photo Received': 'photo_received',
                      'Medical School Country': 'ms_country',
                      'Medical School State/Province': 'ms_state',
                      'Medical School of Graduation': 'ms_name',
                      # 'COMLEX-USA Level 1 Score': 'comlex_score_1',
                      # 'COMLEX-USA Level 2 CE Score': 'comlex_score_2',
                      # 'COMLEX-USA Level 2 PE Score': 'complex_pass_pe',
                       'USMLE Step 1 Score': 'step_1_score',
                       'USMLE Step 2 CK Score': 'step_2ck_score',
                       'USMLE Step 2 CS Score': 'step_2cs_score',
                       'USMLE Step 3 Score': 'step_3_score',
                      'Tracks Applied by Applicant': 'app_track_1',
                      'Tracks Applied by Applicant_1': 'app_track_2',
                      'Tracks Applied by Applicant_2': 'app_track_3',
                      'Tracks Applied by Applicant_3': 'app_track_4',
                      'Count of Non Peer Reviewed Online Publication': 'count_nonpeer_online',
                      'Count of Oral Presentation': 'count_oral_present',
                      'Count of Other Articles': 'count_other_articles',
                      'Count of Peer Reviewed Book Chapter': 'count_book_chapters',
                      'Count of Peer Reviewed Journal Articles/Abstracts': 'count_peer_journal',
                      'Count of Peer Reviewed Journal Articles/Abstracts(Other than Published)': 'count_nonpeer_journal',
                      'Count of Peer Reviewed Online Publication': 'count_peer_online',
                      'Count of Poster Presentation': 'count_poster_present',
                      'Count of Scientific Monograph': 'count_science_monograph'}

"""Importing files"""
# df and df2 are the 2017 2018 datasets, respectively
df = pd.read_csv('ERAS 2017.csv', low_memory=False, index_col=False)\
    [list(included_variables)]
df2 = pd.read_csv('ERAS 2018.csv', low_memory=False, index_col=False)\
   [list(included_variables)]
# Converting column names to preferred names
df.rename(columns=included_variables, inplace=True)
df2.rename(columns=included_variables, inplace=True)

# Converting DOB to age
df['age_years'] = (pd.to_datetime('09/15/2018')-pd.to_datetime(df['dob'])).dt.days.div(365).round(4)
df.drop(columns='dob', inplace=True)

df2['age_years'] = (pd.to_datetime('09/15/2018')-pd.to_datetime(df2['dob'])).dt.days.div(365).round(4)
df2.drop(columns='dob', inplace=True)

# Filling in non-median/non-mean "nan"
df['BLS'].fillna('No', inplace=True)
df['visa_need'].fillna('No', inplace=True)
#df['permanent_state'].fillna('None Given', inplace=True)
#df['contact_state'].fillna('None Given', inplace=True)
df['ms_state'].fillna('None Given', inplace=True)

df['age_years'].fillna(df['age_years'].mean(), inplace=True)

df['step_1_complete'] = df['step_1_score'].notna()
df['step_2ck_complete'] = df['step_2ck_score'].notna()
df['step_2cs_score'].fillna('Not Taken', inplace=True)
df['step_3_complete'] = df['step_3_score'].notna()
df.drop(columns='step_3_score', inplace=True)

# Filling in non-median/non-mean "nan"
df2['BLS'].fillna('No', inplace=True)
df2['visa_need'].fillna('No', inplace=True)
#df2['permanent_state'].fillna('None Given', inplace=True)
#df2['contact_state'].fillna('None Given', inplace=True)
df2['ms_state'].fillna('None Given', inplace=True)

df2['age_years'].fillna(df2['age_years'].mean(), inplace=True)

df2['step_1_complete'] = df2['step_1_score'].notna()
df2['step_2ck_complete'] = df2['step_2ck_score'].notna()
df2['step_2cs_score'].fillna('Not Taken', inplace=True)
df2['step_3_complete'] = df2['step_3_score'].notna()
df2.drop(columns='step_3_score', inplace=True)


# fill NA race
df['race'].fillna('None Given', inplace=True)
df2['race'].fillna('None Given', inplace=True)

# Drop ms state
df.drop(columns='ms_state', inplace=True)
df2.drop(columns='ms_state', inplace=True)

# strings to bool so that onehotencoder can factorize
df['step_2cs_score']=df['step_2cs_score'].replace('Fail', False)
df['step_2cs_score']=df['step_2cs_score'].replace('Not Taken', False)
df['step_2cs_score']=df['step_2cs_score'].replace('Pass',True)

df2['step_2cs_score']=df2['step_2cs_score'].replace('Fail', False)
df2['step_2cs_score']=df2['step_2cs_score'].replace('Not Taken',False)
df2['step_2cs_score']=df2['step_2cs_score'].replace('Pass',True)

#condense 3 responses
df['aoa_recipient'] = df['aoa_recipient'].str.replace('No Response','No')
df2['aoa_recipient'] = df['aoa_recipient'].str.replace('No Response','No')

df['gold_recipient'] = df['gold_recipient'].str.replace('No Response','No')
df2['gold_recipient'] = df2['gold_recipient'].str.replace('No Response','No')

df['sigma_recipient'] = df['sigma_recipient'].str.replace('No Response','No')
df2['sigma_recipient'] = df2['sigma_recipient'].str.replace('No Response','No')

df['malpractice_pending']= df['malpractice_pending'].replace(['N', 'Y'], ['No', 'Yes'])
df2['malpractice_pending']= df2['malpractice_pending'].replace(['N', 'Y'], ['No', 'Yes'])
df['licensure_problem']=df['licensure_problem'].replace(['N', 'Y'], ['No', 'Yes'])
df2['licensure_problem']=df2['licensure_problem'].replace(['N', 'Y'], ['No', 'Yes'])


# Graph Step 1 Scores
df.dropna(subset=['step_1_score'])['step_1_score'].plot.hist()

plt.xlabel('2017 Step 1 Score')
plt.ylabel('2017 Number of Applicants')
plt.show()

df2.dropna(subset=['step_1_score'])['step_1_score'].plot.hist()
plt.xlabel('2018 Step 1 Score')
plt.ylabel('2018 Number of Applicants')
plt.show()

if remove_step_scores:
    df.drop(columns=['step_1_score', 'step_2ck_score'], inplace=True)
    df2.drop(columns=['step_1_score', 'step_2ck_score'], inplace=True)
elif type(step_score_predictor_cutoff) == int:
    df.dropna(subset=['step_2cs_score'], inplace=True)
    df2.dropna(subset=['step_2cs_score'], inplace=True)
    df['step_1_cutoff'] = df['step_1_score'] >= step_score_predictor_cutoff
    df2['step_1_cutoff'] = df2['step_1_score'] >= step_score_predictor_cutoff
    df.drop(columns=['step_2ck_score'], inplace=True)
    df2.drop(columns=['step_2ck_score'], inplace=True)
    df['step_1_score'].fillna(180, inplace=True)
    df2['step_1_score'].fillna(180, inplace=True)

else:
    df['step_1_score'].fillna(df['step_1_score'].mean(), inplace=True)
    df2['step_1_score'].fillna(df2['step_1_score'].mean(), inplace=True)
    df['step_2ck_score'].fillna(df['step_1_score'], inplace=True)
    df2['step_2ck_score'].fillna(df2['step_1_score'], inplace=True)



####################Unifying categorical variables
################# RACE
"""Formatting Data"""
"""One hot encoding medical school country"""

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

x = list(df['race'])
y = list(df2['race'])
allvalues = unique(intersect(x,y))  # superfluous but I think it adds clarity

def matcher(x):
    for i in allvalues:
        if i.lower() in x.lower():
            return i
    else:
        return np.nan
    
df['race'] = df['race'].apply(matcher)
df2['race'] = df2['race'].apply(matcher)

del x,y, allvalues


##################### MS Country

x = list(df['ms_country'])
y = list(df2['ms_country'])
allvalues = unique(intersect(x,y))  # superfluous but I think it adds clarity

def matcher(x):
    for i in allvalues:
        if i.lower() in x.lower():
            return i
    else:
        return np.nan
    
df['ms_country'] = df['ms_country'].apply(matcher)
df2['ms_country'] = df2['ms_country'].apply(matcher)




df['race'].fillna('None Given', inplace=True)
df2['race'].fillna('None Given', inplace=True)
df['ms_country'].fillna('None Given', inplace=True)
df2['ms_country'].fillna('None Given', inplace=True)

#############################  Integrate school rankings

df_schools = pd.read_csv('2017_school_ranks.csv', low_memory=False, 
                         usecols=['INSTITUTION_ID', 'LONG_DESCRIPTION', 'SHORT_DESCRIPTION','RESEARCH_RANK'],
                         nrows=65)
school_list_df=df['ms_name']
school_list_df2=df2['ms_name']



school_list_df=school_list_df.replace('New York','New York University',regex=True)
school_list_df2=school_list_df2.replace('New York','New York University',regex=True)
#school_list_df=school_list_df.replace('New York','New York Univeristy',regex=True)

#import difflib
#tmp = []
#for i, a in enumerate(school_list_df_schools):
#    new_l = [difflib.SequenceMatcher(None, a, b).ratio() for b in school_list_df]
#    ind = new_l.index(max(new_l))
#    eltMove = school_list_df.pop(ind)
#    tmp.append(eltMove)

searchfor = ['New York',
             'Harvard',
             'Johns Hopkins',
             'Stanford',
             'San Francisco',
             'San Diego',
             'Duke',
             'University of Washington',
             'Michigan',
             'Columbia',
             'Penn',
             'Washington University',
             'Yale',
             'David Geffen',
             'Vanderbilt',
             'Pritzker',
             'Pittsburgh',
             'Northwestern University',
             'Feinberg',
             'Weill',
             'Cornell',
             'California',
             'Mayo Medical',
             'Baylor',
             'Mount Sinai',
             'Emory',
             'Case Western',
             'University of Texas Southwestern',
             'University of Virginia',
             'University of Wisconsin',
             'Oregon',
             'Boston University',
             'Warren Alpert',
             'Northeastern Ohio',
             'University of Rochester',
             'Keck',
             'Southern California',
             'Dartmouth',
             'University of Alabama',
             'University of Colorado',
             'University of Iowa',
             'Albert Einstein',
             'University of Cincinnati',
             'University of Florida',
             'University of Maryland',
             'University of Utah',
             'University of Minnesota',
             'University of California',
             'Indiana University',
             'Georgetown',
             'University of Miami',
             'Tufts',
             'University of Massachusetts',
             'University of Illinois',
             'Thomas Jefferson University',
             'Sidney Kimmel',
             'Wake Forest University',
             'Temple University',
             'University of Connecticut',
             'Morsani',
             'USF',
             'University of Vermont',
             'George Washington University',
             'Rush',
             'Stony Brook University',
             'University of Texas',
             'University of Kentucky',
             'University of North Carolina',

             ]

school_list_regex =school_list_df[school_list_df.str.contains('|'.join(searchfor))]
school_list_regex_2 =school_list_df2[school_list_df2.str.contains('|'.join(searchfor))]

rejects = ['Michigan State',
             'Upstate Medical University',
             'West Virginia',
             'Eastern Virginia',
             'Central Michigan ',
             'Edward Via',
             'Ladoke Akintola',
             'Loyola University',
             'Medical College of Wisconsin'
             ]

school_list_regex2 = school_list_regex[~school_list_regex.str.contains('|'.join(rejects))]
school_list_regex2_2 = school_list_regex_2[~school_list_regex_2.str.contains('|'.join(rejects))]

not_in_ranking_list=school_list_df[~school_list_df.isin(school_list_regex2)]
not_in_ranking_list2=school_list_df2[~school_list_df2.isin(school_list_regex2_2)]

df_school_2017=pd.concat([school_list_df,school_list_regex2],axis=1)
df_school_2018=pd.concat([school_list_df2,school_list_regex2_2],axis=1)


df_school_2017.columns = ['Ranking school names', 'school names']
df_school_2018.columns = ['Ranking school names', 'school names']
df_school_2017['school names'].fillna('Other', inplace=True)
df_school_2018['school names'].fillna('Other', inplace=True)

df['ms_name']=df_school_2017['school names']
df2['ms_name']=df_school_2018['school names']

del df_school_2017, 
df_school_2018, 
not_in_ranking_list, 
not_in_ranking_list2, 
rejects, 
school_list_regex_2, 
school_list_regex2_2,
school_list_regex,
searchfor,
df_schools,
school_list_df,
school_list_df2

# Create the ranking dictionary 
school_list_df_schools=df_schools[['LONG_DESCRIPTION','RESEARCH_RANK']]

df=df.merge(school_list_df_schools, left_on='ms_name', right_on='LONG_DESCRIPTION')
df2=df2.merge(school_list_df_schools, left_on='ms_name', right_on='LONG_DESCRIPTION')

df.drop(columns=['ms_name','LONG_DESCRIPTION'], inplace=True)
df2.drop(columns=['ms_name','LONG_DESCRIPTION'], inplace=True)


"""One Hot Encoding Tracks Applied For"""
# CIT is "Clinical Investigation Track"
# PC is "Primary Care Track"
# CAT is the "Traditional Track"
# PST is probably "Physician Scientist Track"

# Beginning One Hot Encoding
df['med_prelim'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                        'app_track_4']] == 'Medicine-Preliminary|2978140P0 (Preliminary)').any(axis=1)
df['im_trad'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Internal Medicine - NYU Traditional|2978140C0 (Categorical)').any(axis=1)
df['im_clin_invest'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Internal Medicine - Clin Invest Track|2978140C2 (Categorical)').any(axis=1)
df['im_prim'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                            'app_track_4']] == 'Medicine-Primary|2978140M0 (Primary Care)').any(axis=1)
df['im_prelim_anes'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Med-Prelim/Anesthesiology|2978140P2 (Preliminary)').any(axis=1)
df['im_research'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Int Med/Research Pathway|2978140C1 (Categorical)').any(axis=1)
df['im_tisch'] = (df[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Internal Medicine - NYU Tisch-Kimmel|2978140C3 (Categorical)').any(axis=1)
#df2
df2['med_prelim'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                        'app_track_4']] == 'Medicine-Preliminary|2978140P0 (Preliminary)').any(axis=1)
df2['im_trad'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Internal Medicine - NYU Traditional|2978140C0 (Categorical)').any(axis=1)
df2['im_clin_invest'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Internal Medicine - Clin Invest Track|2978140C2 (Categorical)').any(axis=1)
df2['im_prim'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                            'app_track_4']] == 'Medicine-Primary|2978140M0 (Primary Care)').any(axis=1)
df2['im_prelim_anes'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Med-Prelim/Anesthesiology|2978140P2 (Preliminary)').any(axis=1)
df2['im_research'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Int Med/Research Pathway|2978140C1 (Categorical)').any(axis=1)
df2['im_tisch'] = (df2[['app_track_1', 'app_track_2', 'app_track_3',
                     'app_track_4']] == 'Internal Medicine - NYU Tisch-Kimmel|2978140C3 (Categorical)').any(axis=1)
# Removing applied old applied tracks column
df.drop(columns=['app_track_1', 'app_track_2', 'app_track_3', 'app_track_4'], inplace=True)
# Removing applied old applied tracks column
df2.drop(columns=['app_track_1', 'app_track_2', 'app_track_3', 'app_track_4'], inplace=True)

"""Begin app selective prediction"""
# Importing 2017 Traditional Invited Applicants
trad_apps_invited = pd.read_csv('Applicants 2017 CAT.csv', low_memory=False,
                                index_col=False)[['AAMC ID', 'Interview Date']]
trad_apps_invited['invited'] = True
trad_apps_invited.rename(columns={'AAMC ID': 'ID',
                                  'Interview Date': 'interview_date'}, inplace=True)
trad_apps_invited.drop(columns='interview_date', inplace=True)

# Create list of traditional applicants
trad_apps = df[df['im_trad'] == 1].merge(trad_apps_invited, on='ID', how='left')    #TODO: Mention 3 that didn't apply
trad_apps['invited'].fillna(value=False, inplace=True)








"""2018 on test data"""
# df and df2 are the 2017 2018 datasets, respectively
################################################################
trad_apps_invited2 = pd.read_csv('Applicants 2018 CAT.csv', low_memory=False,
                                index_col=False)[['AAMC ID', 'Interview Date']]
trad_apps_invited2['invited'] = True
trad_apps_invited2.rename(columns={'AAMC ID': 'ID',
                                  'Interview Date': 'interview_date'}, inplace=True)
trad_apps_invited2.drop(columns='interview_date', inplace=True)

# Create list of traditional applicants
trad_apps2 = df2[df2['im_trad'] == 1].merge(trad_apps_invited2, on='ID', how='left')    #TODO: Mention 3 that didn't apply
trad_apps2['invited'].fillna(value=False, inplace=True)

# Move invited (labels) to front 
cols = trad_apps.columns.tolist()
cols2 = trad_apps2.columns.tolist()

cols.insert(0, cols.pop(cols.index('invited')))
cols.insert(0, cols.pop(cols.index('invited')))



# Invited columns (labels) as Class now.
trad_apps.rename(columns={'invited':'Class'}, inplace=True)
trad_apps2.rename(columns={'invited':'Class'}, inplace=True)

# Remove a lot of variables
trad_apps.drop(columns=['citizenship', 'step_1_complete','step_1_cutoff', 'med_prelim','im_clin_invest','im_prim','im_research','im_prelim_anes', 'im_tisch'],inplace=True)
trad_apps2.drop(columns=['citizenship','step_1_complete','step_1_cutoff', 'med_prelim','im_clin_invest','im_prim','im_research','im_prelim_anes', 'im_tisch'],inplace=True)


#####################################################################################
#trad_apps['train']=1
#trad_apps2['test']=0

#combined =pd.concat([trad_apps, trad_apps2])

#dfcombined=pd.get_dummies(combined['ID'])
#combined=pd.concat([combined,dfcombined], axis=1)

#train_df=combined[combined["train"]==1]
#test_df=combined[combined["train"]==0]

#train_df.drop(['train'],axis=1,inplace=True)
#train_df.drop(['train'],axis=1,inplace=True)

############################# match dttypes 
for x in trad_apps.columns:
    trad_apps2[x]=trad_apps2[x].astype(trad_apps[x].dtypes.name)
dtypes1=trad_apps.dtypes
dtypes2=trad_apps2.dtypes

for col in trad_apps.columns[trad_apps.dtypes == 'bool']:
    trad_apps[col] = trad_apps[col].map({True: 1, False: 0})
    
for col in trad_apps2.columns[trad_apps2.dtypes == 'bool']:
    trad_apps2[col] = trad_apps2[col].map({True: 1, False: 0})

trad_apps.replace("No", 0, inplace=True)
trad_apps.replace("Yes", 1, inplace=True)
trad_apps2.replace("No", 0, inplace=True)
trad_apps2.replace("Yes", 1, inplace=True)




# Test_split. Note that we can't use sklearns test_train_split cuz the train and test data are from 2 different sources.
# Why? I'm guesssing indicies issues. NOTE that RF, LR, and other classifers may not have this issue, but this xgb needs to use
# df.sample(n=test_split) for now.
test_split=round(len(trad_apps)*test_size)
val_split=round(len(trad_apps)*validation_size)

#trad_apps2=trad_apps2.sample(n=test_split)
trad_appst, trad_appsv=train_test_split(trad_apps, test_size = validation_size)
# _, trad_apps2=train_test_split(trad_apps2, test_size = test_size)

trad_appsv.insert(1,'source','val')
trad_appst.insert(1,'source', 'train')
trad_apps2.insert(1,'source', 'test')
# I've saved the train and test data (Including ID and source for later csv result processing later in the pipeline)
trad_appsv.to_csv('Valid_2017_modified.csv',index=False)
trad_appst.to_csv('Train_2017_modified.csv',index=False)
trad_apps2.to_csv('Test_2018_modified.csv',index=False)






#################### Anything before this line can be titled data_processing.py #############################
#####################################################################################################################
##################### Anything after this line can be titled models.py, or classifier.py + results.py, or whatever you want.

#Load data:

train = pd.read_csv('Train_2017_modified.csv')
test = pd.read_csv('Test_2018_modified.csv')
val= pd.read_csv('Valid_2017_modified.csv')

print('Train original:', train.shape, 'Test original:', test.shape, 'Valid original:', val.shape)

print('Train dtypes:',train.dtypes)



data=pd.concat([train, test, val], axis=0)
data.shape



"""
# ROC curve threshold
threshhold= 0.05

test_size = 0.33
test_split=round(len(train)*test_size)


X_train=data.loc[data['source']=='train'] 
X_test=data.loc[data['source']=='test'] 
X_test=X_test.sample(n=test_split)

y_train=data['Class']
y_test=data['Class']
y_test=y_test.sample(n=test_split)


X_train.drop(columns=['source','ID','im_trad', 'Class'],inplace=True)
X_test.drop(columns=['source','ID','im_trad','Class'],inplace=True)
y_train.drop(columns=['source', 'ID','im_trad'],inplace=True)
y_test.drop(columns=['source', 'ID','im_trad'],inplace=True)

# Combine into one data

train=pd.concat([X_train,y_train],axis=1)
test=pd.concat([X_test,y_test],axis=1)

data=pd.concat([train, test],ignore_index=False)
data.shape
"""


################################## Label Encoders #####################################
le = LabelEncoder()
var_to_encode = ['gender', 'race','ms_country']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
    


    
# One-Hot Coding
data = pd.get_dummies(data, columns=var_to_encode)
print(data.columns)

train=data.loc[data['source'] == 'train']
test=data.loc[data['source'] == 'test']
val=data.loc[data['source'] == 'val']

train_pre=train.copy()
test_pre=test.copy()
val_pre=val.copy()


train.drop(columns=['ID', 'source'],inplace=True)
test.drop(columns=['ID', 'source'],inplace=True)
val.drop(columns=['ID', 'source'],inplace=True)


###########################################################################
# a lot of thanks to this guy https://www.kaggle.com/jeremy123w/xgboost-with-roc-curve/data
# Input data files are available in the same directory as the py file.
# UTF-8 decoder
# NOTE : You absolutely NEED to decode utf8 for xgb. Especially when using Label Encoders.
# xgb doesn't like UTF-8 encoding and will interpret as a boolean. 
from subprocess import check_output
print(check_output(["cd", '..', 'ls'+str(dir_Folder)]).decode("utf8"))
print(check_output(["cd", '..','ls'+str(dir_Folder)]).decode("utf8"))


# Use this part if you want to use an experimental csv file.
"""
data = pd.read_csv('creditcard.csv')
"""


# Feature maps. I've decided to use a .fmap that contains all the column names AND the indices of train and test set.
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

# Custom importance generated from feature map/. Can use plot_implotance instead also.
def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

# this is where we get the names of the columns 
def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)
# single_run
# return test_prediction, imp, gbm.best_iteration+1, y_valid, y_pred, X_valid
# later we save as preds, imp, num_boost_rounds, y_valid, y_pred, X_valid
def run_single(train, test, features, target, random_state=0):
    eta = 0.1 
    max_depth= 6 
    subsample = 1
    colsample_bytree = 1
    min_chil_weight=1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight":min_chil_weight,
        "seed": random_state,
        #"num_class" : 22,
    }
    
    # we boost for 500 rounds, and will train until eval-auc hasn't improved in 20 rounds.
    num_boost_round = 500
    early_stopping_rounds = 20


   
    ###################### Uncomment me to use train_test_split instead.
    # X_train, X_valid = train_test_split(train, test_size=test_size, random_state=seed)
    
    # A few notes on XGB.
    # 1. Optimally, xgb likes DMatrix or .value formats instead of pd.Dataframe.
    # 2. xgb DOES hates two things: UTF-8 encoding and booleans Make sure your train and test variables contain none of those
    # 3. 
    
    X_train=train
    X_test=test
    X_valid=val
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], label=y_train, missing=-99)
    dvalid = xgb.DMatrix(X_valid[features], label=y_valid, missing =-99)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    
    # getting y_pred
    model = XGBClassifier(max_depth=12, learning_rate=0.1, n_estimators=1000, n_jobs=8, booster='gbtree')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    
    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    
    #area under the precision-recall curve
    score = average_precision_score(X_valid[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    
    check2=check.round()
    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))
    
    # TODO: need to fix this cross-validation confusion matrix
    """
    # Making the Confusion Matrix
    cm=confusion_matrix(X_valid[features], check)
    cmf = pd.DataFrame(np.array(confusion_matrix(X_valid[features], check))).rename_axis('Predictions', axis=1)
    cmf.index.name = 'Truth'
    print(colored('The Confusion Matrix after REAL cross-validation is: ', 'red'),'\n', cmf)
    # Calculate the accuracy on test set
    predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
    print(colored('The Accuracy on Validation Set is: ', 'magenta'), colored(predict_accuracy_on_test_set, 'magenta'))
    
    print(colored('Total Run time: ', 'grey'), colored(dt.datetime.now()-start_time, 'grey'))
    
    # classification_report
    target_names = ['0','1']
    print(classification_report(y_valid, y_pred, target_names=target_names))
    """
    
    
    
    imp = get_importance(gbm, features)
    #print('Importance array: ', imp)
    
    # First Feature Importance graph 
    x_labels = [val[0] for val in imp]
    y_labels = [val[1] for val in imp]
    plt.figure(figsize=(12, 6))
    ax = pd.Series(y_labels).plot(kind='bar', title='Feature Importance after validation and early stopping')
    ax.set_xticklabels(x_labels)
    
    rects = ax.patches
    
    for rect, label in zip(rects, y_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    # test_prediction is synonimous to y_pred. score is the average precision score after Early Stopping. 
    # score here is NOT like sklearns precision_score
    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing = -99), ntree_limit=gbm.best_iteration+1)
    score = average_precision_score(test[target].values, test_prediction)

    print('area under the precision-recall curve test set: {:.6f}'.format(score))
    
    
    ############################################ ROC Curve ####################
 
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    roc_auc = auc(fpr, tpr)
    # Second feature importance map using plot_importance
    xgb.plot_importance(gbm, title='Feature Importance after validation and early stopping')
    plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve after valiation and ES')
    plt.legend(loc="lower right")
    plt.show()
    ###########################################################################
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, imp, gbm.best_iteration+1, y_valid, y_pred, X_valid


########################## Differences between the ROC AUC and PR AUC ########################################
# Since PR does not account for true negatives (as TN is not a component of either Precision or Recall), 
# or there are many more negatives than positives (a characteristic of class imbalance problem), 
# we would use PR. If not, use ROC.
# PR is much better in illustrating the differences of the algorithms in the case where there are a lot more negative examples than the positive examples.
# We should be using PR AUC for cases where the class imbalance problem occurs like in this example, and not use ROC AUC?
# 

# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ",start_time)

#train, test = train_test_split(data, test_size=.1, random_state=random.seed(2016))

features = list(train.columns.values)
features.remove('Class')
print(features)


print("Building model.. ",dt.datetime.now()-start_time)

# run_single parameters
# Input train, test, features, target (column with labels, and random seed)
# 
preds, imp, num_boost_rounds, y_test, y_pred, X_valid = run_single(train, test, features, 'Class',10)



# Making the Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
cmf = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred))).rename_axis('Predictions', axis=1)
cmf.index.name = 'Truth'
print(colored('The Confusion Matrix after cross-validation is: ', 'red'),'\n', cmf)
# Calculate the accuracy on test set
predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
print(colored('The Accuracy on Validation Set is: ', 'magenta'), colored(predict_accuracy_on_test_set, 'magenta'))

print(colored('Total Run time: ', 'grey'), colored(dt.datetime.now()-start_time, 'grey'))

# classification_report
target_names = ['0','1']
print(classification_report(y_test, y_pred, target_names=target_names))




# Learning Cruve
# Monitering Training Performance
# 
X_train=train.drop('Class', axis=1)
y_train=train['Class']
X_test=test.drop('Class', axis=1)
y_test=test['Class']


model = XGBClassifier()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = metrics.accuracy_score(y_test, predictions)
print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))

y_test = np.asarray(y_test)
misclassified = y_test != model.predict(X_test)



# retrieve performance metrics
results = model.evals_result()
#print("model.evals_result()", results)
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()





# Making the Confusion Matrix for ONLY test predictions
cm=confusion_matrix(y_test, y_pred)
cmf = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred))).rename_axis('Predictions', axis=1)
cmf.index.name = 'Truth'
print(colored('The Confusion Matrix for test set is: ', 'red'),'\n', cmf)
# Calculate the accuracy on test set
predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])
print(colored('The Accuracy on Test Set is: ', 'magenta'), colored(predict_accuracy_on_test_set, 'magenta'))

print(colored('Total Run time: ', 'grey'), colored(dt.datetime.now()-start_time, 'grey'))
# classification_report
target_names = ['0','1']
print(classification_report(y_test, y_pred, target_names=target_names))



# CFeature Importance using Gradient Boosting
y_pred_proba = model.predict_proba(X_test)[:, 1]

xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_pred_proba)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
xgb.plot_importance(model, title='Feature Importance using Gradient Boosting')
plt.show()
plt.figure()
lw = 2
plt.plot(xgb_fpr, xgb_tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % xgb_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (no valid or ES. Only on Test set)')
plt.legend(loc="lower right")
plt.show()


"""Using sklearn Multiclass Logistic Regression"""
regressor = LogisticRegression(solver='newton-cg', multi_class='multinomial')
# Fitting model on test data
regressor.fit(X_train, y_train)
# Predicting outcome based on test data set
y_pred_lr = regressor.predict(X_test)

y_pred_lr_proba = regressor.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, threshhold = roc_curve(y_test, y_pred_lr_proba)
lr_roc_auc = auc(lr_fpr, lr_tpr)

print('AUC of multi-class logistic regression:')
print(lr_roc_auc)

cnf_matrix = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred_lr))).rename_axis('Predictions', axis=1)
cnf_matrix.index.name = 'Truth'
print(cnf_matrix)

# classification_report
target_names = ['0','1']
print(classification_report(y_test, y_pred, target_names=target_names))

"""Using Random Forest"""
clf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=8)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred_rf = clf.predict(X_test)

y_pred_rf_proba = clf.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, threshhold = roc_curve(y_test, y_pred_rf_proba)
rf_roc_auc = auc(rf_fpr, rf_tpr)

print('AUC of Random Forest:')
print(rf_roc_auc)

cnf_matrix = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred_rf))).rename_axis('Predictions', axis=1)
cnf_matrix.index.name = 'Truth'
print(cnf_matrix)

# classification_report
target_names = ['0','1']
print(classification_report(y_test, y_pred, target_names=target_names))






""" Plotting all classifiers"""
plt.figure()
plt.plot(lr_fpr, lr_tpr, color='darkgreen',
         lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot(xgb_fpr, xgb_tpr, color='darkorange',
         lw=2, label='XGBoost (area = %0.2f)' % xgb_roc_auc)
plt.plot(rf_fpr, rf_tpr, color='darkred',
         lw=2, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve with other classifiers')
plt.legend(loc="lower right")
plt.show()









### actual code for roc + threshold charts start here 
# compute fpr, tpr, thresholds and roc_auc
# Comparing with other classifiers

xgb_fpr, xgb_tpr, thresholds = roc_curve(y_test, y_pred_proba)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)

 
plt.figure()
plt.plot(xgb_fpr, xgb_tpr, label='ROC curve (area = %0.2f)' % (xgb_roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
 
# create the axis of thresholds (scores)
ax2 = plt.gca().twinx()
ax2.plot(xgb_fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
ax2.set_ylabel('Threshold',color='r')
ax2.set_ylim([thresholds[-1],thresholds[0]])
ax2.set_xlim([xgb_fpr[0],xgb_fpr[-1]])
 
plt.savefig('roc_and_threshold.png')

####################### Predictions

sub = pd.DataFrame(data=y_pred)
# sub.rename(index=str, columns={0:"GroundTruth"}, inplace=True)
sub['XGB_predicted probability'] = y_pred_proba
sub['LR_predicted probability'] = y_pred_lr_proba
sub['RF_predicted probability'] = y_pred_rf_proba
sub['misclassified']=misclassified
sub['y_test']=y_test
sub['ID']= test_pre['ID']
sub.head()

sub.rename(index=str, columns={0:"Predictions"}, inplace=True)
sub = sub.sort_values('y_test', ascending=False)
sub.to_csv('Predictions_ID_Whole.csv',index=False)








##################### Parellel Coordinate Plots

sub = pd.read_csv('Predictions_ID.csv')

import plotly.plotly as py
import plotly.graph_objs as go



datal = [
    go.Parcoords(
        line = dict(color = sub['ID'],
                   colorscale = [[0,'#D7C16B'],[0.5,'#23D8C3'],[1,'#F3F10F']]),
        dimensions = list([
            dict(range = [0,1],
                constraintrange = [0.5,1],
                label = 'XGB_predicted probability', values = sub['XGB_predicted probability']),
            dict(range = [0,1],
                label = 'LR_predicted probability', values = sub['LR_predicted probability']),
            dict(range = [0,1],
                label = 'RF_predicted probability', values = sub['RF_predicted probability'])
            
        ])
    )
]

layout = go.Layout(
    plot_bgcolor = '#E5E5E5',
    paper_bgcolor = '#E5E5E5',
    font=dict(size=6)
    
)

fig = go.Figure(data = datal, layout = layout, )
py.plot(fig, filename = 'parcoords-basic')

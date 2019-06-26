data_dir = '/Users/MOOSE/Desktop/Practicum_Thesis/IM_Residency_Predictive Analytics'
# data_dir = 'D:/Data/Residency'
# data_dir = '/media/data/Data/Residency'

drop_step_scores = False
step_analysis_cutoff = None

scale_vars = ['age_years', 'step_1_score', 'step_2ck_score']

# TODO: Make a better way to select variables

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
                      # 'Contact Country': 'contact_country',
                      'Contact State': 'contact_state',
                      # 'Contact Zip': 'contact_zip',
                      'Date of Birth': 'dob',
                      'Gender': 'gender',
                      # 'Gold Humanism Honor Society': 'gold_school',
                      'Gold Humanism Honor Society (Yes/No)': 'gold_recipient',
                      'Military Service Obligation': 'military_obligation',
                      'Participating as a Couple in NRMP': 'couples_matching',
                      # 'Permanent City': 'permanent_city',
                      'Permanent Country': 'permanent_country',
                      'Permanent State': 'permanent_state',
                      # 'Permanent Zip': 'permanent_zip',
                      'Self Identify': 'race',
                      # 'Sigma Sigma Phi': 'sigma_school',
                      'Sigma Sigma Phi (Yes/No)': 'sigma_recipient',
                      'US or Canadian Applicant': 'us_or_canadian',
                      'Visa Sponsorship Needed': 'visa_need',
                      # 'Application Reviewed': 'app_reviewed',
                      # 'Withdrawn by Applicant': 'app_withdrawn_stud',
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
                      'Count of Non Peer Reviewed Online Publication': 'count_nonpeer_online',
                      'Count of Oral Presentation': 'count_oral_present',
                      'Count of Other Articles': 'count_other_articles',
                      'Count of Peer Reviewed Book Chapter': 'count_book_chapters',
                      'Count of Peer Reviewed Journal Articles/Abstracts': 'count_peer_journal',
                      'Count of Peer Reviewed Journal Articles/Abstracts(Other than Published)': 'count_nonpeer_journal',
                      'Count of Peer Reviewed Online Publication': 'count_peer_online',
                      'Count of Poster Presentation': 'count_poster_present',
                      'Count of Scientific Monograph': 'count_science_monograph'}
countries = ['AFG', 'AGO', 'AIA', 'ALA', 'ALB', 'AND', 'ANT', 'ARE', 'ARG',
                 'ARM', 'ASM', 'ATA', 'ATF', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI',
                 'BEL', 'BEN', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLM',
                 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BVT',
                 'BWA', 'CAF', 'CAN', 'CCK', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR',
                 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CXR',
                 'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA',
                 'ECU', 'EGY', 'ERI', 'ESH', 'ESP', 'EST', 'ETH', 'FIN', 'FJI',
                 'FLK', 'FRA', 'FRO', 'FSM', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA',
                 'GIB', 'GIN', 'GLP', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL',
                 'GTM', 'GUF', 'GUM', 'GUY', 'HKG', 'HMD', 'HND', 'HRV', 'HTI',
                 'HUN', 'IDN', 'IMN', 'IND', 'IOT', 'IRL', 'IRN', 'IRQ', 'ISL',
                 'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ',
                 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY',
                 'LCA', 'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAC', 'MAF',
                 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI',
                 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT', 'MSR', 'MTQ',
                 'MUS', 'MWI', 'MYS', 'MYT', 'NAM', 'NCL', 'NER', 'NFK', 'NGA',
                 'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK',
                 'PAN', 'PCN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK',
                 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'REU', 'ROU', 'RUS', 'RWA',
                 'SAU', 'SDN', 'SEN', 'SGP', 'SGS', 'SHN', 'SJM', 'SLB', 'SLE',
                 'SLV', 'SMR', 'SOM', 'SPM', 'SRB', 'STP', 'SUR', 'SVK', 'SVN',
                 'SWE', 'SWZ', 'SYC', 'SYR', 'TCA', 'TCD', 'TGO', 'THA', 'TJK',
                 'TKL', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN',
                 'TZA', 'UGA', 'UKR', 'UMI', 'URY', 'USA', 'UZB', 'VAT', 'VCT',
                 'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'YEM', 'ZAF',
                 'ZMB', 'ZWE']

applicant_file = data_dir + '/ERAS 2017.csv'
invite_file = data_dir + '/Applicants 2017 Consolidated.csv'
applicant_file_test = data_dir + '/ERAS 2018.csv'
invite_file_test = data_dir + '/Applicants 2018 Consolidated.csv'
usnwr_file = data_dir + '/usnwr.csv'

# Pre-processing data
from ResidencyTools import ResidencyPreprocessing

X, y = ResidencyPreprocessing(applicant_file=applicant_file, invite_file=invite_file,
                              usnwr_file=usnwr_file, included_variables=included_variables, year=2017,
                              drop_step_scores=drop_step_scores, step_analysis_cutoff=step_analysis_cutoff)

X_test, y_test = ResidencyPreprocessing(applicant_file=applicant_file_test, invite_file=invite_file_test,
                                        usnwr_file=usnwr_file, included_variables=included_variables, year=2018,
                                        drop_step_scores=drop_step_scores, step_analysis_cutoff=step_analysis_cutoff)

# Dropping AAMC ID from training set; Remove test set ID later (Need copy)
X.drop(columns='ID', inplace=True)

# Removing AAMC ID and making a copy for identification later
df_2018 = X_test[:]
X_test.drop(columns='ID', inplace=True)

# Encoding Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ResidencyTools import encoding_columns

"""Encoding and Scaling Data"""
# Creating temporary concatenated tables
df_enc = pd.concat([X, X_test[X_test.columns.difference(['ID'])]])

# TODO Create a function to Scale features, or implement a better preprocessng function
scaler_age = StandardScaler()
df_enc['age_years'] = scaler_age.fit(df_enc['age_years'].values.reshape(-1, 1))
X['age_years'] = scaler_age.transform(X['age_years'].values.reshape(-1, 1))
X_test['age_years'] = scaler_age.transform(X_test['age_years'].values.reshape(-1, 1))

if not drop_step_scores and step_analysis_cutoff is None:
    scaler_step_1 = StandardScaler()
    df_enc['step_1_score'] = scaler_step_1.fit(df_enc['step_1_score'].values.reshape(-1, 1))
    X['step_1_score'] = scaler_step_1.transform(X['step_1_score'].values.reshape(-1, 1))
    X_test['step_1_score'] = scaler_step_1.transform(X_test['step_1_score'].values.reshape(-1, 1))

    scaler_step_2ck = StandardScaler()
    df_enc['step_2ck_score'] = scaler_step_2ck.fit(df_enc['step_2ck_score'].values.reshape(-1, 1))
    X['step_2ck_score'] = scaler_step_2ck.transform(X['step_2ck_score'].values.reshape(-1, 1))
    X_test['step_2ck_score'] = scaler_step_2ck.transform(X_test['step_2ck_score'].values.reshape(-1, 1))


# Encoding data
le_vars, he_vars = encoding_columns(df_enc)

# One hot encoding training data
from sklearn.compose import make_column_transformer

preprocess = make_column_transformer((OneHotEncoder(), list(he_vars)+list(le_vars)),
                                     sparse_threshold=0,
                                     remainder='passthrough')
preprocess.fit(df_enc)
X = preprocess.transform(X)
X_test = preprocess.transform(X_test)

"""Splitting train and validation"""
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

"""Begin Testing"""
from ResidencyTools import classifier_training, classifier_testing

# Logistic Regression Testing
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='newton-cg', multi_class='multinomial')
print('Validation AUC of multi-class logistic regression:')
lr_classifier, lr_fpr, lr_tpr, lr_roc_auc_value =\
    classifier_training(classifier, X_train, y_train, X_val, y_val)
print(lr_roc_auc_value)

print('2018 Test AUC of multi-class logistic regression:')
lr_fpr_test, lr_tpr_test, lr_roc_auc_test, lr_pred_prob_test, lr_precision_test, lr_recall_test =\
    classifier_testing(classifier, X_test, y_test)
print(lr_roc_auc_test)

# XGBoost Testing
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=12, learning_rate=0.1, n_estimators=1000, n_jobs=8, booster='gbtree')
print('Validation AUC of XGBoost:')
xg_classifier, xg_fpr, xg_tpr, xg_roc_auc_value =\
    classifier_training(classifier, X_train, y_train, X_val, y_val)
print(xg_roc_auc_value)

print('2018 Test AUC of XGBoost:')
xg_fpr_test, xg_tpr_test, xg_roc_auc_test, xg_pred_prob_test, xg_precision_test, xg_recall_test =\
    classifier_testing(classifier, X_test, y_test)
print(xg_roc_auc_test)

# Random Forest Testing
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=8)
print('Validation AUC of Random Forest:')
rf_classifier, rf_fpr, rf_tpr, rf_roc_auc_value =\
    classifier_training(classifier, X_train, y_train, X_val, y_val)
print(rf_roc_auc_value)

print('2018 Test AUC of Random Forest:')
rf_fpr_test, rf_tpr_test, rf_roc_auc_test, rf_pred_prob_test, rf_precision_test, rf_recall_test =\
    classifier_testing(classifier, X_test, y_test)
print(rf_roc_auc_test)

"""Plot ROC and AUC"""
import matplotlib.pyplot as plt

# 2017 Test Data
plt.figure()
plt.plot(lr_fpr, lr_tpr, color='darkgreen',
         lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc_value)
plt.plot(xg_fpr, xg_tpr, color='darkorange',
         lw=2, label='XGBoost (area = %0.2f)' % xg_roc_auc_value)
plt.plot(rf_fpr, rf_tpr, color='darkred',
         lw=2, label='Random Forest (area = %0.2f)' % rf_roc_auc_value)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curve')
plt.legend(loc="lower right")
plt.show()

# 2018 Test Data
plt.figure()
plt.plot(lr_fpr_test, lr_tpr_test, color='darkgreen',
         lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc_test)
plt.plot(xg_fpr_test, xg_tpr_test, color='darkorange',
         lw=2, label='XGBoost (area = %0.2f)' % xg_roc_auc_test)
plt.plot(rf_fpr_test, rf_tpr_test, color='darkred',
         lw=2, label='Random Forest (area = %0.2f)' % rf_roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('2018 Test ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting Precision Recall Curve for 2018 data
from funcsigs import signature

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(lr_recall_test, lr_precision_test, color='darkgreen', where='post', lw=2, label='Logistic Regression')
plt.step(xg_recall_test, xg_precision_test, color='darkorange', where='post', lw=2, label='XGBoost')
plt.step(rf_recall_test, rf_precision_test, color='darkred', where='post', lw=2, label='Random Forest')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2018 Test Precision-Recall Curve')
plt.legend(loc="lower right")

plt.show()

# Plotting Parallel Coordinate Plot
# Creating threshold cutoff of logistic regression predictions
threshold = 0.35
lr_pred_thresholded = pd.DataFrame(np.where(lr_pred_prob_test > threshold, 'Predict Invite', 'Predict No Invite'),
                                   columns=['Prediction'])

df_2018 = pd.concat([df_2018, y_test, pd.DataFrame(lr_pred_prob_test, columns=['LR_PredictedProbability']),
                     pd.DataFrame(xg_pred_prob_test, columns=['XR_PredictedProbability']),
                     pd.DataFrame(rf_pred_prob_test, columns=['RF_PredictedProbability'])], axis=1)

if not drop_step_scores and step_analysis_cutoff is None:
    df_2018.to_csv("2018 - Interview Predictions.csv", index=None, header=True)

# # Finding parallel plot code http://benalexkeen.com/parallel-coordinates-in-matplotlib/
# lr_2018 = pd.concat([df_2018, lr_pred_thresholded], axis=1)
# lr_2018 = lr_2018.applymap(str)
# par_plot = pd.plotting.parallel_coordinates(lr_2018, 'Prediction')
# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 20
# plt.show()

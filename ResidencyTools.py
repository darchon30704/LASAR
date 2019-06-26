# ResidencyPreprocessing function is analogous to a "dataloader" in pytorch.
# This dataloader would need applicant_file, invite_file, usnwr_file
# step_analysis_cutoff and drop_step_scores are part of the step-score experiments

def ResidencyPreprocessing(applicant_file, invite_file, usnwr_file, included_variables, year,
                           drop_step_scores=False, step_analysis_cutoff=None):
    import pandas as pd
    import numpy as np

    """Importing files"""
    # Importing Applications
    df_applicants = pd.read_csv(applicant_file, low_memory=False, index_col=False)

    # Finding the tracks
    tracks = {}
    counter = 1
    for var in df_applicants.columns:
        if var.find('Tracks Applied by Applicant') == 0:
            tracks[var] = 'app_track_' + str(counter)
            counter += 1

    # Adding to dictionary of included variables dictionary
    included_variables = dict(included_variables, **tracks)

    # Abbreviating DataFrame
    df_applicants = df_applicants[list(included_variables)]

    # Converting column names to preferred names
    df_applicants.rename(columns=included_variables, inplace=True)

    # Importing Traditional Invited Applicants
    df_invite = pd.read_csv(invite_file, low_memory=False, index_col=False)[['AAMC ID', 'Interview Date']]

    # Importing USNWR Top Schools
    df_usnwr = pd.read_csv(usnwr_file, low_memory=False, index_col=False)

    """Data Pre-processing"""
    # Converting DOB to age
    df_applicants['age_years'] = (pd.to_datetime('09/15/'+str(year)) -
                                  pd.to_datetime(df_applicants['dob'])).dt.days.div(365).round(4)
    df_applicants.drop(columns='dob', inplace=True)


    # Filling in non-median/non-mean "nan"
    df_applicants['BLS'].fillna('No', inplace=True)
    df_applicants['visa_need'].fillna('No', inplace=True)
    df_applicants['permanent_state'].fillna('None Given', inplace=True)
    df_applicants['contact_state'].fillna('None Given', inplace=True)
    df_applicants['ms_state'].fillna('None Given', inplace=True)
    df_applicants['race'].fillna('Not Given', inplace=True)

    df_applicants['age_years'].fillna(df_applicants['age_years'].mean(), inplace=True)

    df_applicants['step_1_complete'] = df_applicants['step_1_score'].notna()
    df_applicants['step_2ck_complete'] = df_applicants['step_2ck_score'].notna()
    df_applicants['step_2cs_score'].fillna('Not Taken', inplace=True)
    df_applicants['step_3_complete'] = df_applicants['step_3_score'].notna()
    df_applicants.drop(columns='step_3_score', inplace=True)

    if drop_step_scores:
        df_applicants.drop(columns=['step_1_score', 'step_2ck_score'], inplace=True)
    elif type(step_analysis_cutoff) == int:
        df_applicants.dropna(subset=['step_1_score', 'step_2ck_score'], inplace=True)
        df_applicants['step_1_cutoff'] = df_applicants['step_1_score'] >= step_analysis_cutoff
        df_applicants.drop(columns=['step_1_score', 'step_2ck_score'], inplace=True)
    else:
        df_applicants['step_1_score'].fillna(df_applicants['step_1_score'].mean(), inplace=True)
        df_applicants['step_2ck_score'].fillna(df_applicants['step_1_score']+np.mean(df_applicants.step_2ck_score) -\
                                               np.mean(df_applicants.step_1_score), inplace=True)
#Since there are 245 factors under medical school countries, we need to create a dictionary that is one-hot encoded.
    """Formatting Data"""
    """One hot encoding medical school country"""
    ms_df = df_applicants.ms_country.str.split(pat=',', expand=True)
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

    # Beginning One Hot Encoding for Countries
    for country in countries:
        df_applicants[country] = (ms_df == country).any(axis=1)

    # Dropping old column and cleaning memory
    df_applicants.drop(columns='ms_country', inplace=True)
    del ms_df

    # Beginning USNWR Top Rankings
    df_applicants['usnwr'] = False
    for i in df_usnwr.usnwr_eras:
        df_applicants.loc[df_applicants.ms_name == i, 'usnwr'] = True
    #df_applicants = pd.concat([df_applicants, pd.DataFrame(usnwr, columns=['usnwr'])], axis=1, ignore_index=True)



# 7 factors under race. Originally, there are more than 100 different strings that the applicants put under race. 
# Again, we are one-hot encoding these 100 strings into only 7 factors for simplicity.
    """One hot encoding race"""
    # Separating data into individual columns
    race_df = df_applicants.race.str.split(pat='|', expand=True)
    races = ['American Indian or Alaskan Native', 'Asian', 'Black or African American',
             'Hispanic, Latino, or of Spanish origin',
             'Native Hawaiian or Pacific Islander', 'Other', 'White']

    # Beginning One Hot Encoding Process
    for race in races:
        df_applicants[race] = (race_df == race).any(axis=1)

    # Dropping old column and cleaning memory
    df_applicants.drop(columns='race', inplace=True)
    del race_df, races

    """One Hot Encoding Tracks Applied For"""
    # CIT is "Clinical Investigation Track"
    # PC is "Primary Care Track"
    # CAT is the "Traditional Tack"
    # PST is probably "Physician Scientist Track"

    # Beginning One Hot Encoding
    df_applicants['med_prelim'] = (df_applicants[list(tracks.values())]
                                   == 'Medicine-Preliminary|2978140P0 (Preliminary)').any(axis=1)
    df_applicants['im_trad'] = (((df_applicants[list(tracks.values())]
                                  == 'Internal Medicine - NYU Traditional|2978140C0 (Categorical)') |
                                 (df_applicants[list(tracks.values())]
                                  == 'Int Med- NYU Traditional|2978140C0 (Categorical)'))).any(axis=1)
    df_applicants['im_clin_invest'] = (df_applicants[list(tracks.values())]
                                       == 'Internal Medicine - Clin Invest Track|2978140C2 (Categorical)').any(axis=1)
    df_applicants['im_prim'] = (df_applicants[list(tracks.values())]
                                == 'Medicine-Primary|2978140M0 (Primary Care)').any(axis=1)
    df_applicants['im_prelim_anes'] = (df_applicants[list(tracks.values())]
                                       == 'Med-Prelim/Anesthesiology|2978140P2 (Preliminary)').any(axis=1)
    df_applicants['im_research'] = (df_applicants[list(tracks.values())]
                                    == 'Int Med/Research Pathway|2978140C1 (Categorical)').any(axis=1)
    df_applicants['im_tisch'] = (((df_applicants[list(tracks.values())]
                                   == 'Internal Medicine - NYU Tisch-Kimmel|2978140C3 (Categorical)') |
                                  (df_applicants[list(tracks.values())]
                                   == 'Int Med- NYU Tisch-Kimmel|2978140C3 (Categorical)'))).any(axis=1)

    # Removing applied old applied tracks column
    df_applicants.drop(columns=tracks.values(), inplace=True)

    """Combining Traditional Applicant Data"""
    df_invite['invited'] = True
    df_invite.rename(columns={'AAMC ID': 'ID', 'Interview Date': 'interview_date'}, inplace=True)
    df_invite.drop(columns='interview_date', inplace=True)

    # Create list of traditional and clinical investigation applicants
    # TODO CHECK IF CIT IS INCLUDED
    apps = df_applicants[(df_applicants['im_trad'] == 1) |
                         (df_applicants['im_clin_invest'] == 1) |
                         (df_applicants['im_prim'] == 1) |
                         (df_applicants['im_research'] == 1)
                         ].merge(df_invite, on='ID', how='left')
    apps['invited'].fillna(value=False, inplace=True)

    if type(step_analysis_cutoff) == int:
        X = apps[apps.columns.difference(['invited', 'step_1_cutoff'])]
        y = apps['step_1_cutoff']
    else:
        X = apps[apps.columns.difference(['invited'])]
        y = apps['invited']

    return X, y


def encoding_columns(df):
    """ Encoding categorical data """
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
    # Label Encoder
    le_vars = (df[df.columns.difference(countries + ['age_years', 'step_1_score', 'step_2ck_score'])].
               nunique() <= 2)
    le_vars = le_vars[le_vars].index

    he_vars = ((df.nunique() >= 3) &
               (df.columns != 'ID') &
               (df.columns != 'step_1_score') &
               (df.columns != 'step_2ck_score') &
               (df.columns != 'age_years') &
               (df.columns != 'count_nonpeer_online') &
               (df.columns != 'count_oral_present') &
               (df.columns != 'count_other_articles') &
               (df.columns != 'count_book_chapters') &
               (df.columns != 'count_peer_journal') &
               (df.columns != 'count_peer_online') &
               (df.columns != 'count_poster_present') &
               (df.columns != 'count_science_monograph'))
    he_vars = he_vars.index[he_vars]

    return le_vars, he_vars


def classifier_training(classifier, X_train, y_train, X_test, y_test):
    from sklearn.metrics import roc_curve, auc

    # Fitting model on test data
    classifier.fit(X_train, y_train)
    # Predicting outcome based on test data set
    y_pred = classifier.predict(X_test)

    y_pred_prob = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_value = auc(fpr, tpr)
    return classifier, fpr, tpr, roc_auc_value


def classifier_testing(classifier, X_test, y_test):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    # Predicting outcomes from test data
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_value = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    return fpr, tpr, roc_auc_value, y_pred_prob, precision, recall


def roc_auc(y_test, y_pred):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # One hot encoder of test and output variables
    enc = OneHotEncoder(categories='auto').fit(y_test.reshape(-1, 1))
    y_test_enc = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_pred_enc = enc.transform(y_pred.reshape(-1, 1)).toarray()

    for i in range(y_test_enc.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_enc[:, i], y_pred_enc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_enc.ravel(), y_pred_enc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

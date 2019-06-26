# LASAR
Using Machine Learning to Predict Residency Interview Invites at NYU Langone: Learning Algorithm for the Swift Appraisal of Residents (LASAR)


Listed are the variables required from the ERAS dataset.

Here are the variables without quotations

AAMC ID,
Medical Education or Training Interrupted,
ACLS,
BLS,
Board Certification,
Malpractice Cases Pending,
Medical Licensure Problem,
PALS,
Misdemeanor Conviction,
Alpha Omega Alpha (Yes/No),
Citizenship,
Contact State,
Date of Birth,
Gender,
Gold Humanism Honor Society (Yes/No),
Military Service Obligation,
Participating as a Couple in NRMP,
Permanent Country,
Permanent State,
Self Identify,
Sigma Sigma Phi (Yes/No),
US or Canadian Applicant,
Visa Sponsorship Needed,
ECFMG Certification Received,
CSA Exam Status,
ECFMG Certified,
Medical School Transcript Received,
MSPE Received,
Personal Statement Received,
Photo Received,
Medical School Country,
Medical School State/Province,
Medical School of Graduation,
USMLE Step 1 Score,
USMLE Step 2 CK Score,
USMLE Step 2 CS Score,
USMLE Step 3 Score,
Count of Non Peer Reviewed Online Publication,
Count of Oral Presentation,
Count of Other Articles,
Count of Peer Reviewed Book Chapter,
Count of Peer Reviewed Journal Articles/Abstracts,
Count of Peer Reviewed Journal Articles/Abstracts(Other than Published),
Count of Peer Reviewed Online Publication,
Count of Poster Presentation,
Count of Scientific Monograph

NOTE 1: Ground truth labels are under a different file, provided by the NYU Langone residency admissions department.
NOTE 2: Research Ranking, or usnwr_eras is from 2017_school_ranks.csv, and under RESEARCH_RANK. 
NOTE 3: when using ResidencyTools.py and analysis.py, usnwr_eras contains a list of schools in order of their rankings from 2017. This is under usnwr.csv


Below are the variables converted for Python/R readability, and a few descriptions for those variables

'AAMC ID': 'ID',
'Medical Education or Training Interrupted': 'education_interruption',
'ACLS': 'ACLS', : Advanced cardiac life support, or advanced cardiovascular life support, often referred to by its abbreviation as "ACLS", refers to a set of clinical algorithms for the urgent treatment of cardiac arrest, stroke, myocardial infarction, and other life-threatening cardiovascular emergencies.[1] Outside North America, Advanced Life Support (ALS) is used.
'BLS': 'BLS', : Basic life support (BLS) is a level of medical care which is used for victims of life-threatening illnesses or injuries until they can be given full medical care at a hospital. It can be provided by trained medical personnel, including emergency medical technicians, paramedics, and by qualified bystanders.
'Board Certification': 'board_certified',
'Malpractice Cases Pending': 'malpractice_pending',
'Medical Licensure Problem': 'licensure_problem',
'PALS': 'PALS', : Pediatric Advanced Life Support (PALS) is a 2-day (with an additional self study day) American Heart Association training program co-branded with the American Academy of Pediatrics
'Misdemeanor Conviction': 'misdemeanor',
'Alpha Omega Alpha': 'aoa_school', Alpha Omega Alpha Honor Medical Society (ΑΩΑ) is an honor society in the field of medicine
'Alpha Omega Alpha (Yes/No)': 'aoa_recipient', : Any ΑΩΑ recipeint who has provided administrative support for a Chapter, for at least three years.
'Citizenship': 'citizenship', c: US citizenship
'Contact City': 'contact_city',
'Contact Country': 'contact_country',
'Contact State': 'contact_state',
'Contact Zip': 'contact_zip',
'Date of Birth': 'dob',
'Gender': 'gender',
'Gold Humanism Honor Society': 'gold_school', The Gold Humanism Honor Society (GHHS) is a national honor society that honors senior medical students, residents, role-model physician teachers and other exemplars recognized for demonstrated excellence in clinical care, leadership, compassion and dedication to service. It was created by the Arnold P. Gold Foundation for Humanism in Medicine.
'Gold Humanism Honor Society (Yes/No)': 'gold_recipient', GHHS award recipient
'Military Service Obligation': 'military_obligation',
'Participating as a Couple in NRMP': 'couples_matching',
'Permanent City': 'permanent_city',
'Permanent Country': 'permanent_country',
'Permanent State': 'permanent_state',
'Permanent Zip': 'permanent_zip',
'Self Identify': 'race', Sigma Sigma Phi
'Sigma Sigma Phi': 'sigma_school', : Sigma Sigma Phi (ΣΣΦ or SSP), is the national osteopathic medicine honors fraternity for medical students training to be Doctors of Osteopathic Medicine (D.O.)
'Sigma Sigma Phi (Yes/No)': 'sigma_recipient', SSP award recipient
'US or Canadian Applicant': 'us_or_canadian',
'Visa Sponsorship Needed': 'visa_need',
'Application Reviewed': 'app_reviewed', : Filled out by reviewer.
'Withdrawn by Applicant': 'app_withdrawn_stud', : Whether the application was withdrawn by the applicant. Filled out by reviewer.
'Withdrawn by Program': 'app_withdrawn_prog', :  Whether the application was withdrawn by NYU Langones program when applying. Filled out by reviewer.
'On Hold': 'on_hold', : Filled out by reviewer.
'Average Document Score': 'avg_doc_score', : Filled out by reviewer.
'ECFMG Certification Received': 'ecfmg_cert_received', : Filled out by reviewer. Educational Commission for Foreign Medical Graduates (ECFMG) assesses the readiness of international medical graduates to enter residency or fellowship programs in the United States that are accredited by the Accreditation Council for Graduate Medical Education (ACGME). This is mainly for international applicants.
'CSA Exam Status': 'csa_received', : Filled out by reviewer. The term CSA applies to all persons who successfully pass the National Commission for the Certification of Surgical Assistants’ Certification Examination and meets the ongoing requirements for maintaining the credential.
'ECFMG Certified': 'ecfmg_cert', :Filled out by reviewer.
'Medical School Transcript Received': 'transcript_received', :Filled out by reviewer.
'MSPE Received': 'mspe_received', :Filled out by reviewer.
'Personal Statement Received': 'ps_received', :Filled out by reviewer.
'Photo Received': 'photo_received', : Filled out by reviewer.
'Medical School Country': 'ms_country', : Country of Medical School that applicant has graduated. or anticipated to graduate from.
'Medical School State/Province': 'ms_state', : State of Medical School that applicant has graduated. or anticipated to graduate from.
'Medical School of Graduation': 'ms_name',
'COMLEX-USA Level 1 Score': 'comlex_score_1',
'COMLEX-USA Level 2 CE Score': 'comlex_score_2',
'COMLEX-USA Level 2 PE Score': 'complex_pass_pe',
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
'Count of Scientific Monograph': 'count_science_monograph'





Just by running ra_brad.py (with the necessary data files), the results, metrics,
accuracy reports will be displayed on an output. 

Alternatively, you could just run ResidencyTools.py and analysis.py 
with the original dataset. 

Required datasets for ResidencyTools.py and analysis.py: 

ERAS 2017.csv : Contains the original variables from ERAS 2017
ERAS 2018.csv : Contains the original variables from ERAS 2018
Applicants 2017 Consolidated.csv : Contains the ground truth labels from 2017 from ALL tracks
Applicants 2018 Consolidated.csv : Contains the ground truth labels from 2018 from ALL tracks
usnwr.csv : Contains research ranking of the applicant's medical school from 2017

Required datasets for ra_brad.py:

ERAS 2017.csv : Contains the original variables from ERAS 2017
ERAS 2018.csv : Contains the original variables from ERAS 2018
2017_school_ranks.csv : Contains research ranking of the applicant's medical school from 2017
Applicants 2017 CAT.csv : Contains the ground truth labels from 2017 from ONLY traditional track
Applicants 2018 CAT.csv : Contains the ground truth labels from 2018 from ONLY traditional track

Please do not share or distribute without consent from 
Moosun.Kim@nyulangone.org, james.feng@nyulangone.org, Yin.A@nyulangone.org, or r3dtitanium@gmail.com.

Any questions can be directed to the email above as well.

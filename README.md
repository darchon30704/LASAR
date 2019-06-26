# LASAR / Resident Retriever
**Using Machine Learning to Predict Residency Interview Invites at NYU Langone** <br>
*Learning Algorithm for the Swift Appraisal of Residents (LASAR)*

![ERASLOGO](http://wichita.kumc.edu/Images/wichita/psychiatry/logo-eras-data.jpg)
![NYULOGO](https://www.myface.org/wp-content/uploads/2016/09/NYU-Langone-Medical-Center-Logo.png)
![NRMPLOGO](data:image/png;base64,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)
![classifiers](https://devopedia.org/images/article/74/9857.1523796001.png)

With increasing number of applications per applicant in medical residency programs, AI (Artificial Intelligence) is being investigated as a decision support tool for residency applications. This pilot experiment utilizes machine learning techniques to predict interview invites at NYU Langone’s Internal Medicine residency program. By using Electronic Resident Application Services (ERAS) data and medical school rankings, we utilized machine learning algorithms (such as Logistic Regression, Random Forest, and ultimately, Gradient Boosting) to predict probabilities of an applicant being invited for interviews. As a result, we achieved an AUCROC performance of 0.94, 0.95 and 0.95, respectively, for the three algorithms described above). We also found that Step 1 scores, age, and medical school ranking of the applicant were most influential in our model. This experimental analysis demonstrates that using machine learning to predict residency interview invites at NYU Langone is feasible with ERAS data. 

Listed are the variables required from the ERAS dataset.

**Here are the variables without quotations**

AAMC ID<br>
Medical Education or Training Interrupted<br>
ACLS<br>
BLS<br>
Board Certification<br>
Malpractice Cases Pending<br>
Medical Licensure Problem<br>
PALS<br>
Misdemeanor Conviction<br>
Alpha Omega Alpha (Yes/No)<br>
Citizenship<br<br>
Contact State<br>
Date of Birth<br>
Gender<br>
Gold Humanism Honor Society (Yes/No)<br>
Military Service Obligation<br>
Participating as a Couple in NRMP<br>
Permanent Country<br>
Permanent State<br>
Self Identify<br>
Sigma Sigma Phi (Yes/No)<br>
US or Canadian Applicant<br>
Visa Sponsorship Needed<br>
ECFMG Certification Received<br>
CSA Exam Status<br>
ECFMG Certified<br>
Medical School Transcript Received<br>
MSPE Received<br>
Personal Statement Received<br>
Photo Received<br>
Medical School Country<br>
Medical School State/Province<br>
Medical School of Graduation<br>
USMLE Step 1 Score<br>
USMLE Step 2 CK Score<br>
USMLE Step 2 CS Score<br>
USMLE Step 3 Score<br>
Count of Non Peer Reviewed Online Publication<br>
Count of Oral Presentation<br>
Count of Other Articles<br>
Count of Peer Reviewed Book Chapter<br>
Count of Peer Reviewed Journal Articles/Abstracts<br>
Count of Peer Reviewed Journal Articles/Abstracts(Other than Published)<br>
Count of Peer Reviewed Online Publication<br>
Count of Poster Presentation<br>
Count of Scientific Monograph

NOTE 1: Ground truth labels are under a different file, provided by the NYU Langone residency admissions department.
NOTE 2: Research Ranking, or usnwr_eras is from 2017_school_ranks.csv, and under RESEARCH_RANK. 
NOTE 3: when using ResidencyTools.py and analysis.py, usnwr_eras contains a list of schools in order of their rankings from 2017. This is under usnwr.csv


**Below are the variables converted for Python/R readability, and a few descriptions for those variables**

**AAMC ID**: 'ID'<br>
**Medical Education or Training Interrupted**: 'education_interruption',<br>
**ACLS**: 'ACLS', : Advanced cardiac life support, or advanced cardiovascular life support, often referred to by its abbreviation as "ACLS", refers to a set of clinical algorithms for the urgent treatment of cardiac arrest, stroke, myocardial infarction, and other life-threatening cardiovascular emergencies.[1] Outside North America, Advanced Life Support (ALS) is used.<br>
**BLS**: 'BLS', : Basic life support (BLS) is a level of medical care which is used for victims of life-threatening illnesses or injuries until they can be given full medical care at a hospital. It can be provided by trained medical personnel, including emergency medical technicians, paramedics, and by qualified bystanders.<br>
**Board Certification**: 'board_certified',<br>
**Malpractice Cases Pending**: 'malpractice_pending',<br>
**Medical Licensure Problem**: 'licensure_problem'<br>
**PALS**: 'PALS', : Pediatric Advanced Life Support (PALS) is a 2-day (with an additional self study day) American Heart Association training program co-branded with the American Academy of Pediatrics<br>
**Misdemeanor Conviction**: 'misdemeanor'<br>
**Alpha Omega Alpha**: 'aoa_school', Alpha Omega Alpha Honor Medical Society (ΑΩΑ) is an honor society in the field of medicine
**Alpha Omega Alpha (Yes/No)**: 'aoa_recipient', : Any ΑΩΑ recipeint who has provided administrative support for a Chapter, for at least three years.<br>
**Citizenship**: 'citizenship', : US citizenship<br>
**Contact City**: 'contact_city'<br>
**Contact Country**: 'contact_country'<br>
**Contact State**: 'contact_state'<br>
**Contact Zip**: 'contact_zip'<br>
**Date of Birth**: 'dob'<br>
**Gender**: 'gender'<br>
**Gold Humanism Honor Society**: 'gold_school' The Gold Humanism Honor Society (GHHS) is a national honor society that honors senior medical students, residents, role-model physician teachers and other exemplars recognized for demonstrated excellence in clinical care, leadership, compassion and dedication to service. It was created by the Arnold P. Gold Foundation for Humanism in Medicine.
**Gold Humanism Honor Society (Yes/No)**: 'gold_recipient', GHHS award recipient
**Military Service Obligation**: 'military_obligation'<br>
**Participating as a Couple in NRMP**: 'couples_matching'<br>
**Permanent City**: 'permanent_city'<br>
**Permanent Country**: 'permanent_country'<br>
**Permanent State**: 'permanent_state'<br>
**Permanent Zip**: 'permanent_zip'<br>
**Self Identify**: 'race', Sigma Sigma Phi<br>
**Sigma Sigma Phi**: 'sigma_school' : Sigma Sigma Phi (ΣΣΦ or SSP), is the national osteopathic medicine honors fraternity for medical students training to be Doctors of Osteopathic Medicine (D.O.)<br>
**Sigma Sigma Phi (Yes/No)**: 'sigma_recipient', SSP award recipient<br>
**US or Canadian Applicant**: 'us_or_canadian',<br>
**Visa Sponsorship Needed**: 'visa_need',<br>
**Application Reviewed**: 'app_reviewed', : Filled out by reviewer.<br>
**Withdrawn by Applicant**: 'app_withdrawn_stud' : Whether the application was withdrawn by the applicant. Filled out by reviewer.<br>
**Withdrawn by Program**: 'app_withdrawn_prog' :  Whether the application was withdrawn by NYU Langones program when applying. Filled out by reviewer.<br>
**On Hold**: 'on_hold' : Filled out by reviewer.<br>
Average Document Score**: 'avg_doc_score' : Filled out by reviewer.<br>
**ECFMG Certification Received**: 'ecfmg_cert_received', : Filled out by reviewer. Educational Commission for Foreign Medical Graduates (ECFMG) assesses the readiness of international medical graduates to enter residency or fellowship programs in the United States that are accredited by the Accreditation Council for Graduate Medical Education (ACGME). This is mainly for international applicants.
**CSA Exam Status**: 'csa_received', : Filled out by reviewer. The term CSA applies to all persons who successfully pass the National Commission for the Certification of Surgical Assistants’ Certification Examination and meets the ongoing requirements for maintaining the credential.<br>
**ECFMG Certified**: 'ecfmg_cert' :Filled out by reviewer.<br>
**Medical School Transcript Received**: 'transcript_received', :Filled out by reviewer.<br>
**MSPE Received**: 'mspe_received' :Filled out by reviewer.<br>
**Personal Statement Received**: 'ps_received', :Filled out by reviewer.<br>
**Photo Received**: 'photo_received' : Filled out by reviewer.<br>
**Medical School Country**: 'ms_country' : Country of Medical School that applicant has graduated. or anticipated to graduate from.<br>
**Medical School State/Province**: 'ms_state' : State of Medical School that applicant has graduated. or anticipated to graduate from.<br>
**Medical School of Graduation**: 'ms_name'<br>
**COMLEX-USA Level 1 Score**: 'comlex_score_1'<br>
**COMLEX-USA Level 2 CE Score**: 'comlex_score_2'<br>
**COMLEX-USA Level 2 PE Score**: 'complex_pass_pe'<br>
**USMLE Step 1 Score**: 'step_1_score'<br>
**USMLE Step 2 CK Score**: 'step_2ck_score'<br>
**USMLE Step 2 CS Score**: 'step_2cs_score'<br>
**USMLE Step 3 Score**: 'step_3_score'<br>
**Count of Non Peer Reviewed Online Publication**: 'count_nonpeer_online'<br>
**Count of Oral Presentation**: 'count_oral_present'<br>
**Count of Other Articles**: 'count_other_articles'<br>
**Count of Peer Reviewed Book Chapter**: 'count_book_chapters'<br>
**Count of Peer Reviewed Journal Articles/Abstracts**: 'count_peer_journal'<br>
**Count of Peer Reviewed Journal Articles/Abstracts(Other than Published)**: 'count_nonpeer_journal'<br>
**Count of Peer Reviewed Online Publication**: 'count_peer_online',<br>
**Count of Poster Presentation**: 'count_poster_present'<br>
**Count of Scientific Monograph**: 'count_science_monograph'<br>



## Directions ##

Just by running ra_brad.py (with the necessary data files), the results, metrics,
accuracy reports will be displayed on an output. 

Alternatively, you could just run ResidencyTools.py and analysis.py 
with the original dataset. 

## Required datasets for ResidencyTools.py and analysis.py ##

**ERAS 2017.csv** : Contains the original variables from ERAS 2017
**ERAS 2018.csv** : Contains the original variables from ERAS 2018
**Applicants 2017 Consolidated.csv** : Contains the ground truth labels from 2017 from ALL tracks
**Applicants 2018 Consolidated.csv** : Contains the ground truth labels from 2018 from ALL tracks
**usnwr.csv** : Contains research ranking of the applicant's medical school from 2017

Required datasets for ra_brad.py:

**ERAS 2017.csv** : Contains the original variables from ERAS 2017
**ERAS 2018.csv** : Contains the original variables from ERAS 2018
**2017_school_ranks.csv** : Contains research ranking of the applicant's medical school from 2017
**Applicants 2017 CAT.csv** : Contains the ground truth labels from 2017 from ONLY traditional track
**Applicants 2018 CAT.csv** : Contains the ground truth labels from 2018 from ONLY traditional track

Please do not share or distribute without consent from 
Moosun.Kim@nyulangone.org, james.feng@nyulangone.org, Yin.A@nyulangone.org, or r3dtitanium@gmail.com.

Any questions can be directed to the email above as well.

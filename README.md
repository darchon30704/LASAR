# LASAR / Resident Retriever
**Using Machine Learning to Predict Residency Interview Invites at NYU Langone** <br>
*Learning Algorithm for the Swift Appraisal of Residents (LASAR)*

![ERASLOGO](http://wichita.kumc.edu/Images/wichita/psychiatry/logo-eras-data.jpg)
![NYULOGO](https://www.myface.org/wp-content/uploads/2016/09/NYU-Langone-Medical-Center-Logo.png)
![NRMPLOGO](https://img.medscape.com/thumbnail_library/ht_181127_national_resident_matching_program_NRMP_800x450.png)
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

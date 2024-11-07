import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx import bipartite

def mark(ICD):
    # 1-5
    if ICD in ALLERGY:
        return "Allergy"
    elif ICD in ANEMIA:
        return "Anemia"
    elif ICD in ASTHMA:
        return "Asthma"
    elif ICD in ATRIAL_FIBRILLATION:
        return "Atrial Fibriliation"
    elif ICD in AUTOIMMUNE_Diseases:
        return "Automimmune Diseases"
    # 6-10
    elif ICD in Blindness_Visual_Impairment:
        return "Blindness Visual Impairment"
    elif ICD in Blood_And_Blood_Forming_Organ_Diseases:
        return "Blood And Blood Forming Organ Diseases"
    elif ICD in BRADYCARDIAS_AND_CONDUCTION_DISEASES:
        return "Bradycardias and Conduction Diseases"
    elif ICD in CARDIAC_VALVE_DISEASES:
        return "Cardiac Valve Diseases"
    elif ICD in CATARACT_AND_OTHER_LENS_DISEASES:
        return "Cataract and Other Lens Diseases"
    # 11-15
    elif ICD in CEREBROVASCULAR_DISEASE:
        return "Cerebrovascular Disease"
    elif ICD in CHROMOSOMAL_ABNORMALITIES:
        return "Chromosomal Abnormalities"
    elif ICD in CHRONIC_INFECTIOUS_DISEASES:
        return "Chronic Infectious Diseases"
    elif ICD in CHRONIC_KIDNEY_DISEASES:
        return "Chronic Kidney Diseases"
    elif ICD in CHRONIC_LIVER_DISEASES:
        return "Chronic Liver Diseases"
    # 16-20
    elif ICD in CHRONIC_PANCREAS_BILIARY_TRACT_AND_GALLBLADDER_DISEASE:
        return "Chronic Pancreas Biliary Tract and Gallbladder Disease"
    elif ICD in CHRONIC_ULCER_OF_THE_SKIN:
        return "Chronic Ulcer of the Skin"
    elif ICD in COLITIS_AND_RELATED_DISEASES:
        return "Colitis and Related Diseases"
    elif ICD in COPD_EMPHYSEMA_CHRONIC_BRONCHITIS:
        return "COPD Emphysema Chronic Bronchitis"
    elif ICD in DEAFNESS_HEARING_IMPAIRMENT:
        return "Deafness Hearing Impairment"
    # 21-25
    elif ICD in DEMENTIA:
        return "Dementia"
    elif ICD in DIABETES:
        return "Diabetes"
    elif ICD in DORSOPATHIES:
        return "Dorsopathies"
    elif ICD in DYSLIPIDEMIA:
        return "Dyslipidemia"
    elif ICD in EAR_NOSE_THROAT_DISEASES:
        return "Ear Nose Throat Diseases"
    # 26-30
    elif ICD in EPILEPSY:
        return "Epilepsy"
    elif ICD in ESOPHAGUS_STOMACH_AND_DUODENUM_DISEASES:
        return "Esophagus Stomach and Duodenum Diseases"
    elif ICD in GLAUCOMA:
        return "Glaucoma"
    elif ICD in HEART_FAILURE:
        return "Heart Failure"

    # 31-35
    elif ICD in HYPERTENSION:
        return "Hypertension"
    elif ICD in INFLAMMATORY_ARTHROPATHIES:
        return "Inflammatory Arthropathies"
    elif ICD in INFLAMMATORY_BOWEL_DISEASES:
        return "Inflammatory Bowel Diseases"
    elif ICD in ISCHEMIC_HEART_DISEASE:
        return "Ischemic Heart Disease"
    elif ICD in MIGRAINE_AND_FACIAL_PAIN_SYNDROMES:
        return "Migraine and Facial Pain Syndromes"
    # 36-40
    elif ICD in MULTIPLE_SCLEROSIS:
        return "Multiple Sclerosis"
    elif ICD in OBESITY:
        return "Obesity"
    elif ICD in OSTEOARTHRITIS_AND_OTHER_DEGENERATIVE_JOINT_DISEASES:
        return "Osteoarthritis and other Degenerative Joint Diseases"
    elif ICD in OSTEOPOROSIS:
        return "Osteoporosis"
    elif ICD in OTHER_CARDIOVASCULAR_DISEASES:
        return "Other Cardiovascular Diseases"
    # 41-45
    elif ICD in OTHER_DIGESTIVE_DISEASES:
        return "Other Digestive Diseases"
    elif ICD in OTHER_EYE_DISEASES:
        return "Other Eye Diseases"
    elif ICD in OTHER_GENITOURINARY_DISEASES:
        return "Other Genitourinary Diseases"
    elif ICD in OTHER_METABOLIC_DISEASES:
        return "Other Metabolic Diseases"
    elif ICD in OTHER_MUSCULOSKELETAL_AND_JOINT_DISEASES:
        return "Other Musculoskeletal and Joint Diseases"
    # 46-50
    elif ICD in OTHER_NEUROLOGICAL_DISEASES:
        return "Other Neurological Diseases"
    elif ICD in OTHER_RESPIRATORY_DISEASES:
        return "Other Respiratory Diseases"
    elif ICD in OTHER_SKIN_DISEASES:
        return "Other Skin Diseases"
    elif ICD in PARKINSON_AND_PARKINSONISM:
        return "Parkinson and Parkinsonism"
    elif ICD in PERIPHERAL_NEUROPATHY:
        return "Peripheral Neuropathy"
    # 51-55
    elif ICD in PERIPHERAL_VASCULAR_DISEASE:
        return "Peripheral Vascular Disease"
    elif ICD in PROSTATE_DISEASES:
        return "Prostate diseases"
    elif ICD in SLEEP_DISORDERS:
        return "Sleep Disorders"
    elif ICD in SOLID_NEOPLASMS:
        return "Solid Neoplasms"
    elif ICD in THYROID_DISEASES:
        return "Thyroid Diseases"
    # 56-慢性病
    elif ICD in VENOUS_AND_LYMPHATIC_DISEASES:
        return "Venous and Lymphatic Diseases"
    
    # 1-5
    elif ICD in Organic_Mental_Disorders:
        return "Organic Mental Disorders"
    elif ICD in Mental_And_Behavioral_Disorders_Due_To_Psychoactive_Substance_Use:
        return "Mental And Behavioral Disorders Due To Psychoactive Substance Use"
    elif ICD in SCHIZOPHRENIA_AND_DELUSIONAL_DISEASES:
        return "Schizophrenia And Delusional Diseases"
    elif ICD in DEPRESSION_AND_MOOD_DISEASES:
        return "Depression And Mood Diseases"
    elif ICD in NEUROTIC_STRESS_RELATED_AND_SOMATOFORM_DISEASES:
        return "Neurotic Stress Related And Somatoform Diseases"
    # 6-10
    elif ICD in Eating_Disorders:
        return "Eating Disorders"
    elif ICD in SLEEP_DISORDERS_F:
        return "SLEEP DISORDERS_F"
    elif ICD in Specific_Personality_Disorders:
        return "Specific Personality Disorders"
    elif ICD in Mental_Retardation:
        return "Mental Retardation"
    elif ICD in Pervasive_Developmental_Disorders:
        return "Pervasive Developmental Disorders"
    # 11-
    elif ICD in Behavioral_and_Emotional_Disorders_with_Onset_Usually_Occurring_in_Childhood_and_Adolescence:
        return "Behavioral and Emotional Disorders with Onse Usually Occurring in Childhood and Adolescence"
    else:
        return 'Other'

def weighted_projected_graph(B, X):
    
    projected_graph = nx.Graph()
    projected_graph.add_nodes_from(X)  # Add nodes to the new graph

    for u in X:
        # X_copy = X.copy()
        
        # X_copy.remove(u)
        X.remove(u)
        nib_u = set(B[u])
        for v in X:
            nib_v = set(B[v])
            wight = len(nib_u.intersection(nib_v))
            if wight > 0:
                projected_graph.add_edge(u, v, weight=wight)  # 否则，创建一条新边，并设置初始权重为1
    return projected_graph


if __name__ == '__main__':
    data = pd.read_csv('../data/raw/DIAGNOSES_ICD.csv')
    ICD_List = set(data.ICD10_CODE)
    # 1 过敏反应
    ALLERGY = [item for item in ICD_List if item.startswith('J30.1') or item.startswith('J30.2') or item.startswith('J30.3') or item.startswith('J30.4') or item.startswith('J45.0') or item.startswith('K52.2') or item.startswith('L20') or item.startswith('L23') or item.startswith('L50.0') or item.startswith('Z51.6')]
    # 2 贫血症
    ANEMIA = [item for item in ICD_List if item.startswith('D50') or item.startswith('D51') or item.startswith('D52') or item.startswith('D53') or item.startswith('D55') or item.startswith('D56') or item.startswith('D57') or item.startswith('D58') or item.startswith('D59') or item.startswith('D60') or item.startswith('D61') or item.startswith('D63') or item.startswith('D64')]
    # 3 哮喘
    ASTHMA = [item for item in ICD_List if item.startswith('J45')]
    # 4 心房颤抖
    ATRIAL_FIBRILLATION = [item for item in ICD_List if item.startswith('I48')]
    # 5 自身免疫性疾病
    AUTOIMMUNE_Diseases = [item for item in ICD_List if item.startswith('L73.1') or item.startswith('L10') or item.startswith('L12') or item.startswith('L40') or item.startswith('L41') or item.startswith('L93') or item.startswith('L94') or item.startswith('L95') or item.startswith('M30') or item.startswith('M31') or item.startswith('M32') or item.startswith('M33') or item.startswith('M34') or item.startswith('M35') or item.startswith('M36')]

    # 6 失明、视力障碍
    Blindness_Visual_Impairment = [item for item in ICD_List if item.startswith('H54') or item.startswith('Z44.2') or item.startswith('Z97.0')]
    # 7 血液和造血器官疾病
    Blood_And_Blood_Forming_Organ_Diseases = [item for item in ICD_List if item.startswith('D66') or item.startswith('D67') or item.startswith('D68') or item.startswith('D69') or item.startswith('D71') or item.startswith('D72.0') or item.startswith('D73.0') or item.startswith('D73.1') or item.startswith('D73.2') or item.startswith('D74') or item.startswith('D75.0') or item.startswith('D76.1') or item.startswith('D76.3') or item.startswith('D77') or item.startswith('D80') or item.startswith('D81') or item.startswith('D82') or item.startswith('D84') or item.startswith('D86') or item.startswith('D89')]
    # 8 心动过缓和传导性疾病
    BRADYCARDIAS_AND_CONDUCTION_DISEASES = [item for item in ICD_List if item.startswith('I44.1') or item.startswith('I44.2') or item.startswith('I44.3') or item.startswith('I45.3') or item.startswith('I45.5') or item.startswith('Z95.0')]
    # 9 心脏瓣膜疾病
    CARDIAC_VALVE_DISEASES = [item for item in ICD_List if item.startswith('I05') or item.startswith('I06') or item.startswith('I07') or item.startswith('I08') or item.startswith('I09.1') or item.startswith('I09.8') or item.startswith('I34') or item.startswith('I35') or item.startswith('I36') or item.startswith('I37') or item.startswith('I38') or item.startswith('I39.0') or item.startswith('I39.1') or item.startswith('I39.2') or item.startswith('I39.3') or item.startswith('I39.4') or item.startswith('Q22') or item.startswith('Q23') or item.startswith('Z95.2') or item.startswith('Z95.3') or item.startswith('Z95.4')]
    # 10 白内障和其他晶状体疾病
    CATARACT_AND_OTHER_LENS_DISEASES = [item for item in ICD_List if item.startswith('H25') or item.startswith('H26') or item.startswith('H27') or item.startswith('H28') or item.startswith('Q12') or item.startswith('Z96.1')]

    # 11 脑血管疾病
    CEREBROVASCULAR_DISEASE = [item for item in ICD_List if item.startswith('G45') or item.startswith('G46') or item.startswith('I60') or item.startswith('I61') or item.startswith('I62') or item.startswith('I63') or item.startswith('I64') or item.startswith('I67') or item.startswith('I69')]
    # 12 染色体异常
    CHROMOSOMAL_ABNORMALITIES = [item for item in ICD_List if item.startswith('Q90') or item.startswith('Q91') or item.startswith('Q92') or item.startswith('Q93') or item.startswith('Q95') or item.startswith('Q96') or item.startswith('Q97') or item.startswith('Q98') or item.startswith('Q99')]
    # 13 慢性传染病
    CHRONIC_INFECTIOUS_DISEASES = [item for item in ICD_List if item.startswith('A15') or item.startswith('A16') or item.startswith('A17') or item.startswith('A18') or item.startswith('A19') or item.startswith('A30') or item.startswith('A31') or item.startswith('A50') or item.startswith('A52') or item.startswith('A53') or item.startswith('A65') or item.startswith('A66') or item.startswith('A67') or item.startswith('A69.2') or item.startswith('A81') or item.startswith('B20') or item.startswith('B21') or item.startswith('B22') or item.startswith('B23') or item.startswith('B24') or item.startswith('B38.1') or item.startswith('B39.1') or item.startswith('B40.1') or item.startswith('B57.2') or item.startswith('B57.3') or item.startswith('B57.4') or item.startswith('B57.5') or item.startswith('B65') or item.startswith('B92') or item.startswith('B94') or item.startswith('J65') or item.startswith('M86.3') or item.startswith('M86.4') or item.startswith('M86.5') or item.startswith('M86.6')]
    # 14 慢性肾病
    CHRONIC_KIDNEY_DISEASES = [item for item in ICD_List if item.startswith('I12.0') or item.startswith('I13.0') or item.startswith('I13.1') or item.startswith('I13.2') or item.startswith('I13.9') or item.startswith('N01') or item.startswith('N03') or item.startswith('N04') or item.startswith('N05') or item.startswith('N07') or item.startswith('N08') or item.startswith('N11') or item.startswith('N18.3') or item.startswith('N18.4') or item.startswith('N18.5') or item.startswith('N18.9') or item.startswith('Q60') or item.startswith('Q61.1') or item.startswith('Q61.2') or item.startswith('Q61.3') or item.startswith('Q61.4') or item.startswith('Q61.5') or item.startswith('Q61.8') or item.startswith('Q61.9') or item.startswith('Z90.5') or item.startswith('Z94.0')]
    # 15 慢性肝病
    CHRONIC_LIVER_DISEASES = [item for item in ICD_List if item.startswith('B18') or item.startswith('K70') or item.startswith('K71.3') or item.startswith('K71.4') or item.startswith('K71.5') or item.startswith('K717') or item.startswith('K72.1') or item.startswith('K73') or item.startswith('K74') or item.startswith('K75.3') or item.startswith('K75.4') or item.startswith('K75.8') or item.startswith('K76.1') or item.startswith('K76.6') or item.startswith('K76.7') or item.startswith('K77.8') or item.startswith('Q44.6') or item.startswith('Z94.4')]

    # 16 慢性胰腺、胆道和胆囊疾病
    CHRONIC_PANCREAS_BILIARY_TRACT_AND_GALLBLADDER_DISEASE = [item for item in ICD_List if item.startswith('K80.0') or item.startswith('K80.1') or item.startswith('K80.2') or item.startswith('K80.8') or item.startswith('K81.1') or item.startswith('K86') or item.startswith('Q44.0') or item.startswith('Q44.1') or item.startswith('Q44.2') or item.startswith('Q44.3') or item.startswith('Q44.4') or item.startswith('Q44.5') or item.startswith('Q45.0')]
    # 17 慢性皮肤溃疡
    CHRONIC_ULCER_OF_THE_SKIN = [item for item in ICD_List if item.startswith('I83.0') or item.startswith('I83.2') or item.startswith('L89') or item.startswith('L97') or item.startswith('L98.4')]
    # 18 结肠炎及相关疾病
    COLITIS_AND_RELATED_DISEASES = [item for item in ICD_List if item.startswith('K52.0') or item.startswith('K52.8') or item.startswith('K55.1') or item.startswith('K55.2') or item.startswith('K57.2') or item.startswith('K57.3') or item.startswith('K57.4') or item.startswith('K57.5') or item.startswith('K57.8') or item.startswith('K57.9') or item.startswith('K58') or item.startswith('K59.0') or item.startswith('K59.2') or item.startswith('K62') or item.startswith('K63.4') or item.startswith('K64')]
    # 19 肺气肿，慢性支气管炎
    COPD_EMPHYSEMA_CHRONIC_BRONCHITIS = [item for item in ICD_List if item.startswith('J41') or item.startswith('J42') or item.startswith('J43') or item.startswith('J44') or item.startswith('J47')]
    # 20 耳聋、听力障碍
    DEAFNESS_HEARING_IMPAIRMENT = [item for item in ICD_List if item.startswith('H80') or item.startswith('H90') or item.startswith('H91.1') or item.startswith('H91.3') or item.startswith('H91.9') or item.startswith('Q16') or item.startswith('Z45.3') or item.startswith('Z46.1') or item.startswith('Z96.2') or item.startswith('Z97.4')]

    # 21 痴呆
    DEMENTIA = [item for item in ICD_List if item.startswith('G30') or item.startswith('G31')]
    # 22 糖尿病
    DIABETES = [item for item in ICD_List if item.startswith('E10') or item.startswith('E11') or item.startswith('E13') or item.startswith('E14') or item.startswith('E89.1')]
    # 23 背病
    DORSOPATHIES = [item for item in ICD_List if item.startswith('M40') or item.startswith('M41') or item.startswith('M42') or item.startswith('M43') or item.startswith('M47') or item.startswith('M48') or item.startswith('M49') or item.startswith('M50') or item.startswith('M51') or item.startswith('M53') or item.startswith('Q67.5') or item.startswith('Q76.1') or item.startswith('Q76.4')]
    # 24 血脂障碍
    DYSLIPIDEMIA = [item for item in ICD_List if item.startswith('E78')]
    # 25 耳鼻喉病
    EAR_NOSE_THROAT_DISEASES = [item for item in ICD_List if item.startswith('H60.4') or item.startswith('H66.1') or item.startswith('H66.2') or item.startswith('H66.3') or item.startswith('H70.1') or item.startswith('H71') or item.startswith('H73.1') or item.startswith('H74.1') or item.startswith('H81.0') or item.startswith('H83.1') or item.startswith('H83.2') or item.startswith('H95') or item.startswith('J30.0') or item.startswith('J31') or item.startswith('J32') or item.startswith('J33') or item.startswith('J34.1') or item.startswith('J34.2') or item.startswith('J34.3') or item.startswith('J35') or item.startswith('J37') or item.startswith('J38.0') or item.startswith('J38.6') or item.startswith('K05.1') or item.startswith('K05.3') or item.startswith('K07') or item.startswith('K11.0') or item.startswith('K11.7') or item.startswith('Q30') or item.startswith('Q31') or item.startswith('Q32') or item.startswith('Q35') or item.startswith('Q36') or item.startswith('Q37') or item.startswith('Q38')]

    # 26 癫痫
    EPILEPSY = [item for item in ICD_List if item.startswith('G40')]
    # 27 食道、胃和十二指肠疾病
    ESOPHAGUS_STOMACH_AND_DUODENUM_DISEASES = [item for item in ICD_List if item.startswith('I85') or item.startswith('I86.4') or item.startswith('I98.2') or item.startswith('I98.3') or item.startswith('K21') or item.startswith('K22.0') or item.startswith('K22.2') or item.startswith('K22.4') or item.startswith('K22.5') or item.startswith('K22.7') or item.startswith('K23.0') or item.startswith('K23.1') or item.startswith('K25.4') or item.startswith('K25.5') or item.startswith('K25.6') or item.startswith('K25.7') or item.startswith('K26.4') or item.startswith('K26.5') or item.startswith('K26.6') or item.startswith('K26.7') or item.startswith('K27.4') or item.startswith('K27.5') or item.startswith('K27.6') or item.startswith('K27.7') or item.startswith('K28.4') or item.startswith('K28.5') or item.startswith('K28.6') or item.startswith('K28.7') or item.startswith('K29.3') or item.startswith('K29.4') or item.startswith('K29.5') or item.startswith('K29.6') or item.startswith('K29.7') or item.startswith('K29.8') or item.startswith('K29.9') or item.startswith('K31.1') or item.startswith('K31.2') or item.startswith('K31.3') or item.startswith('K31.4') or item.startswith('K31.5') or item.startswith('Q39') or item.startswith('Q40') or item.startswith('Z90.3')]
    # 28 青光眼
    GLAUCOMA = [item for item in ICD_List if item.startswith('H40.1') or item.startswith('H40.2') or item.startswith('H40.3') or item.startswith('H40.4') or item.startswith('H40.5') or item.startswith('H40.6') or item.startswith('H40.8') or item.startswith('H40.9')]
    # 29 心力衰竭
    HEART_FAILURE = [item for item in ICD_List if item.startswith('I11.0') or item.startswith('I13.0') or item.startswith('I13.2') or item.startswith('I27') or item.startswith('I28.0') or item.startswith('I42') or item.startswith('I43') or item.startswith('I50') or item.startswith('I51.5') or item.startswith('I51.7') or item.startswith('I52.8') or item.startswith('Z94.1') or item.startswith('Z94.3')]
    # 30 高血压
    HYPERTENSION = [item for item in ICD_List if item.startswith('I10') or item.startswith('I11') or item.startswith('I12') or item.startswith('I13') or item.startswith('I15')]

    # 31 炎症性关节病
    INFLAMMATORY_ARTHROPATHIES = [item for item in ICD_List if item.startswith('M02.3') or item.startswith('M05') or item.startswith('M06') or item.startswith('M07') or item.startswith('M08') or item.startswith('M09') or item.startswith('M10') or item.startswith('M11') or item.startswith('M12') or item.startswith('M13') or item.startswith('M14') or item.startswith('M45') or item.startswith('M46.0') or item.startswith('M46.1') or item.startswith('M46.8') or item.startswith('M46.9')]
    # 32 炎性肠疾病
    INFLAMMATORY_BOWEL_DISEASES = [item for item in ICD_List if item.startswith('K50') or item.startswith('K51')]
    # 33 缺血性心脏病
    ISCHEMIC_HEART_DISEASE = [item for item in ICD_List if item.startswith('I20') or item.startswith('I21') or item.startswith('I22') or item.startswith('I24') or item.startswith('I25') or item.startswith('Z95.1') or item.startswith('Z95.5')]
    # 34 偏头痛和面部疼痛综合征
    MIGRAINE_AND_FACIAL_PAIN_SYNDROMES = [item for item in ICD_List if item.startswith('G43') or item.startswith('G44.0') or item.startswith('G44.1') or item.startswith('G44.2') or item.startswith('G44.3') or item.startswith('G44.8') or item.startswith('G50')]
    # 35 多发性硬化
    MULTIPLE_SCLEROSIS = [item for item in ICD_List if item.startswith('G35')]

    # 36 肥胖
    OBESITY = [item for item in ICD_List if item.startswith('E66')]
    # 37 骨关节炎和其他退行性关节疾病
    OSTEOARTHRITIS_AND_OTHER_DEGENERATIVE_JOINT_DISEASES = [item for item in ICD_List if item.startswith('M15') or item.startswith('M16') or item.startswith('M17') or item.startswith('M18') or item.startswith('M19') or item.startswith('M36.2') or item.startswith('M36.3')]
    # 38 骨质疏松症
    OSTEOPOROSIS = [item for item in ICD_List if item.startswith('M80') or item.startswith('M81') or item.startswith('M82')]
    # 39 其他心血管疾病
    OTHER_CARDIOVASCULAR_DISEASES = [item for item in ICD_List if item.startswith('I09') or item.startswith('I28.1') or item.startswith('I31.0') or item.startswith('I31.1') or item.startswith('I45.6') or item.startswith('I49.5') or item.startswith('I49.8') or item.startswith('I70') or item.startswith('I71') or item.startswith('I72') or item.startswith('I79.0') or item.startswith('I79.1') or item.startswith('I95.0') or item.startswith('I95.1') or item.startswith('I95.8') or item.startswith('Q20') or item.startswith('Q21') or item.startswith('Q24') or item.startswith('Q25') or item.startswith('Q26') or item.startswith('Q27') or item.startswith('Q28') or item.startswith('Z95.8') or item.startswith('Z95.9')]
    # 40 其他消化系统疾病
    OTHER_DIGESTIVE_DISEASES = [item for item in ICD_List if item.startswith('K66.0') or item.startswith('K90.0') or item.startswith('K90.1') or item.startswith('K90.2') or item.startswith('K91.1') or item.startswith('K93') or item.startswith('Q41') or item.startswith('Q42') or item.startswith('Q43') or item.startswith('R15') or item.startswith('Z90.4') or item.startswith('Z98.0')]

    # 41 其他眼病
    OTHER_EYE_DISEASES = [item for item in ICD_List if item.startswith('H02.2') or item.startswith('H02.3') or item.startswith('H02.4') or item.startswith('H02.5') or item.startswith('H04') or item.startswith('H05') or item.startswith('H10.4') or item.startswith('H17') or item.startswith('H18.4') or item.startswith('H18.5') or item.startswith('H18.6') or item.startswith('H18.7') or item.startswith('H18.8') or item.startswith('H18.9') or item.startswith('H19.3') or item.startswith('H19.8') or item.startswith('H20.1') or item.startswith('H21') or item.startswith('H31.0') or item.startswith('H31.1') or item.startswith('H31.2') or item.startswith('H31.8') or item.startswith('H31.9') or item.startswith('H33') or item.startswith('H35.2') or item.startswith('H35.3') or item.startswith('H35.4') or item.startswith('H35.5') or item.startswith('H35.7') or item.startswith('H35.8') or item.startswith('H35.9') or item.startswith('H36') or item.startswith('H47') or item.startswith('H48') or item.startswith('H49') or item.startswith('H51') or item.startswith('Q10') or item.startswith('Q11') or item.startswith('Q13') or item.startswith('Q14') or item.startswith('Q15') or item.startswith('Z94.7')]
    # 42 其他泌尿生殖系统疾病
    OTHER_GENITOURINARY_DISEASES = [item for item in ICD_List if item.startswith('B90.1') or item.startswith('N20.0') or item.startswith('N20.2') or item.startswith('N20.9') or item.startswith('N21.0') or item.startswith('N21.8') or item.startswith('N21.9') or item.startswith('N22') or item.startswith('N30.1') or item.startswith('N30.2') or item.startswith('N30.3') or item.startswith('N30.4') or item.startswith('N31') or item.startswith('N32.0') or item.startswith('N32.3') or item.startswith('N32.8') or item.startswith('N32.9') or item.startswith('N33') or item.startswith('N35') or item.startswith('N39.3') or item.startswith('N39.4') or item.startswith('N48.0') or item.startswith('N48.4') or item.startswith('N48.9') or item.startswith('N70.1') or item.startswith('N71.1') or item.startswith('N73.1') or item.startswith('N73.4') or item.startswith('N73.6') or item.startswith('N76.1') or item.startswith('N76.3') or item.startswith('N81') or item.startswith('N88') or item.startswith('N89.5') or item.startswith('N90.5') or item.startswith('N95.2') or item.startswith('Q54') or item.startswith('Q62.0') or item.startswith('Q62.1') or item.startswith('Q62.2') or item.startswith('Q62.3') or item.startswith('Q62.4') or item.startswith('Q62.7') or item.startswith('Q62.8') or item.startswith('Q63.8') or item.startswith('Q63.9') or item.startswith('Q64.0') or item.startswith('Q64.1') or item.startswith('Q64.3') or item.startswith('Q64.4') or item.startswith('Q64.5') or item.startswith('Q64.6') or item.startswith('Q64.7') or item.startswith('Q64.8') or item.startswith('Q64.9') or item.startswith('Z90.6') or item.startswith('Z90.7') or item.startswith('Z96.0')]
    # 43 其他代谢性疾病
    OTHER_METABOLIC_DISEASES = [item for item in ICD_List if item.startswith('E20') or item.startswith('E21') or item.startswith('E22') or item.startswith('E23') or item.startswith('E24') or item.startswith('E25') or item.startswith('E26') or item.startswith('E27') or item.startswith('E28') or item.startswith('E29') or item.startswith('E31') or item.startswith('E34') or item.startswith('E35') or item.startswith('E40') or item.startswith('E41') or item.startswith('E42') or item.startswith('E43') or item.startswith('E44') or item.startswith('E45') or item.startswith('E46') or item.startswith('E64') or item.startswith('E70') or item.startswith('E71') or item.startswith('E72') or item.startswith('E74') or item.startswith('E75') or item.startswith('E76') or item.startswith('E77') or item.startswith('E79') or item.startswith('E80') or item.startswith('E83') or item.startswith('E84') or item.startswith('E85') or item.startswith('E88') or item.startswith('E89') or item.startswith('K90.3') or item.startswith('K90.4') or item.startswith('K90.8') or item.startswith('K90.9') or item.startswith('K91.2') or item.startswith('M83') or item.startswith('M88') or item.startswith('N25')]
    # 44 其他肌肉骨骼和关节疾病
    OTHER_MUSCULOSKELETAL_AND_JOINT_DISEASES = [item for item in ICD_List if item.startswith('B90.2') or item.startswith('M21.2') or item.startswith('M21.3') or item.startswith('M21.4') or item.startswith('M21.5') or item.startswith('M21.6') or item.startswith('M21.7') or item.startswith('M21.8') or item.startswith('M21.9') or item.startswith('M22') or item.startswith('M23') or item.startswith('M24') or item.startswith('M25.2') or item.startswith('M25.3') or item.startswith('M35.7') or item.startswith('M61') or item.startswith('M65.2') or item.startswith('M65.3') or item.startswith('M65.4') or item.startswith('M70.0') or item.startswith('M72.0') or item.startswith('M72.2') or item.startswith('M72.4') or item.startswith('M75.0') or item.startswith('M75.1') or item.startswith('M75.3') or item.startswith('M75.4') or item.startswith('M79.7') or item.startswith('M84.1') or item.startswith('M89') or item.startswith('M91') or item.startswith('M93') or item.startswith('M94') or item.startswith('M96') or item.startswith('M99') or item.startswith('Q65') or item.startswith('Q66') or item.startswith('Q68') or item.startswith('Q71') or item.startswith('Q72') or item.startswith('Q73') or item.startswith('Q74') or item.startswith('Q77') or item.startswith('Q78') or item.startswith('Q79.6') or item.startswith('Q79.8') or item.startswith('Q87') or item.startswith('S38.2') or item.startswith('S48') or item.startswith('S58') or item.startswith('S68') or item.startswith('S78') or item.startswith('S88') or item.startswith('S98') or item.startswith('T05') or item.startswith('T09.6') or item.startswith('T11.6') or item.startswith('T13.6') or item.startswith('T14.7') or item.startswith('T90') or item.startswith('T91') or item.startswith('T92') or item.startswith('T93') or item.startswith('T94') or item.startswith('T95') or item.startswith('T96') or item.startswith('T97') or item.startswith('T98') or item.startswith('Z44.0') or item.startswith('Z44.1') or item.startswith('Z89.1') or item.startswith('Z89.2') or item.startswith('Z89.3') or item.startswith('Z89.4') or item.startswith('Z89.5') or item.startswith('Z89.6') or item.startswith('Z89.7') or item.startswith('Z89.8') or item.startswith('Z89.9') or item.startswith('Z94.6') or item.startswith('Z96.6') or item.startswith('Z97.1')]
    # 45 其他神经系统疾病 
    OTHER_NEUROLOGICAL_DISEASES = [item for item in ICD_List if item.startswith('B90.0') or item.startswith('D48.2') or item.startswith('G04.1') or item.startswith('G09') or item.startswith('G10') or item.startswith('G11') or item.startswith('G12') or item.startswith('G13') or item.startswith('G24') or item.startswith('G25') or item.startswith('G26') or item.startswith('G32') or item.startswith('G37') or item.startswith('G51') or item.startswith('G52') or item.startswith('G53') or item.startswith('G70') or item.startswith('G71') or item.startswith('G72.3') or item.startswith('G72.4') or item.startswith('G72.8') or item.startswith('G72.9') or item.startswith('G73') or item.startswith('G80') or item.startswith('G81') or item.startswith('G82') or item.startswith('G83') or item.startswith('G90') or item.startswith('G91') or item.startswith('G93.8') or item.startswith('G93.9') or item.startswith('G95') or item.startswith('G99') or item.startswith('M47.1') or item.startswith('Q00') or item.startswith('Q01') or item.startswith('Q02') or item.startswith('Q03') or item.startswith('Q04') or item.startswith('Q05') or item.startswith('Q06') or item.startswith('Q07') or item.startswith('Q76.0')]

    # 46 其他呼吸道疾病
    OTHER_RESPIRATORY_DISEASES = [item for item in ICD_List if item.startswith('B90.9') or item.startswith('E66.2') or item.startswith('J60') or item.startswith('J61') or item.startswith('J62') or item.startswith('J63') or item.startswith('J64') or item.startswith('J65') or item.startswith('J66') or item.startswith('J67') or item.startswith('J68.4') or item.startswith('J70.1') or item.startswith('J70.3') or item.startswith('J70.4') or item.startswith('J84') or item.startswith('J92') or item.startswith('J94.1') or item.startswith('J95.3') or item.startswith('J95.5') or item.startswith('J96.1') or item.startswith('J98') or item.startswith('Q33') or item.startswith('Q34') or item.startswith('Z90.2') or item.startswith('Z94.2') or item.startswith('Z94.3') or item.startswith('Z96.3')]
    # 47 其他皮肤病
    OTHER_SKIN_DISEASES = [item for item in ICD_List if item.startswith('L13') or item.startswith('L28') or item.startswith('L30.1') or item.startswith('L43') or item.startswith('L50.8') or item.startswith('L58.1') or item.startswith('L85') or item.startswith('Q80') or item.startswith('Q81') or item.startswith('Q82.1') or item.startswith('Q82.2') or item.startswith('Q82.9')]
    # 48 帕金森
    PARKINSON_AND_PARKINSONISM = [item for item in ICD_List if item.startswith('G20') or item.startswith('G21') or item.startswith('G22') or item.startswith('G23')]
    # 49 周围神经病
    PERIPHERAL_NEUROPATHY = [item for item in ICD_List if item.startswith('B91') or item.startswith('G14') or item.startswith('G54') or item.startswith('G55') or item.startswith('G56') or item.startswith('G57') or item.startswith('G58') or item.startswith('G59') or item.startswith('G60') or item.startswith('G62.8') or item.startswith('G62.9') or item.startswith('G63') or item.startswith('M47.2') or item.startswith('M53.1') or item.startswith('M54.1')]
    # 50 周围性血管疾病
    PERIPHERAL_VASCULAR_DISEASE = [item for item in ICD_List if item.startswith('I70.2') or item.startswith('I73') or item.startswith('I79.2') or item.startswith('I79.8')]

    # 51 前列腺疾病
    PROSTATE_DISEASES = [item for item in ICD_List if item.startswith('N40') or item.startswith('N41.1') or item.startswith('N41.8')]
    # 52 睡眠障碍
    SLEEP_DISORDERS = [item for item in ICD_List if item.startswith('G47')]
    # 53 实体肿瘤
    SOLID_NEOPLASMS = [item for item in ICD_List if item.startswith('C') or item.startswith('D00') or item.startswith('D01') or item.startswith('D02') or item.startswith('D03') or item.startswith('D04') or item.startswith('D05') or item.startswith('D06') or item.startswith('D07') or item.startswith('D09') or item.startswith('D32.0') or item.startswith('D32.1') or item.startswith('D32.9') or item.startswith('D33.0') or item.startswith('D33.1') or item.startswith('D33.2') or item.startswith('D33.3') or item.startswith('D33.4') or item.startswith('Q85')]
    # 54 甲状腺疾病
    THYROID_DISEASES = [item for item in ICD_List if item.startswith('E00') or item.startswith('E01') or item.startswith('E02') or item.startswith('E03') or item.startswith('E05') or item.startswith('E06.2') or item.startswith('E06.3') or item.startswith('E06.5') or item.startswith('E07') or item.startswith('E35.0') or item.startswith('E89.0')]
    # 55 静脉和淋巴疾病
    VENOUS_AND_LYMPHATIC_DISEASES = [item for item in ICD_List if item.startswith('I78.0') or item.startswith('I83') or item.startswith('I87') or item.startswith('I89') or item.startswith('I97.2') or item.startswith('Q82.0')]

    # 1 器质性（包括症状）精神障碍 F00-F09
    Organic_Mental_Disorders = [item for item in ICD_List if item.startswith('F0')]
    # 2 精神和行为障碍是由于精神活性物质使用所致 F10-F19
    Mental_And_Behavioral_Disorders_Due_To_Psychoactive_Substance_Use = [item for item in ICD_List if item.startswith('F1')]
    # 3 精神分裂和妄想症 F20、F22、F24、F25、F28
    SCHIZOPHRENIA_AND_DELUSIONAL_DISEASES = [item for item in ICD_List if item.startswith('F20') or item.startswith('F22') or item.startswith('F24') or item.startswith('F25') or item.startswith('F28')]
    # 4 抑郁症和情绪疾病 F30、F31、F32、F33、F34、F38、F39
    DEPRESSION_AND_MOOD_DISEASES = [item for item in ICD_List if item.startswith('F30') or item.startswith('F31') or item.startswith('F32') or item.startswith('F33') or item.startswith('F34') or item.startswith('F38') or item.startswith('F39')]
    # 5 神经相关躯体疾病 F40、F41、F42、F43、F44、F45、F48
    NEUROTIC_STRESS_RELATED_AND_SOMATOFORM_DISEASES = [item for item in ICD_List if item.startswith('F40') or item.startswith('F41') or item.startswith('F42') or item.startswith('F43') or item.startswith('F44') or item.startswith('F45') or item.startswith('F48')]
    # 6 进食障碍 F50
    Eating_Disorders = [item for item in ICD_List if item.startswith('F50')]
    # 7 睡眠障碍 F51.0、F51.1、F51.2、F51.3
    SLEEP_DISORDERS_F = [item for item in ICD_List if item.startswith('F51.0') or item.startswith('F51.1') or item.startswith('F51.2') or item.startswith('F51.3')]
    # 8 特定人格障碍 F60
    Specific_Personality_Disorders = [item for item in ICD_List if item.startswith('F60')]
    # 9 精神发育障碍 F70-F79
    Mental_Retardation = [item for item in ICD_List if item.startswith('F7')]
    # 10 广泛性发育障碍 F84
    Pervasive_Developmental_Disorders = [item for item in ICD_List if item.startswith('F84')]
    # 11 行为和情绪障碍通常起源于儿童和青少年时期 F90-F98
    Behavioral_and_Emotional_Disorders_with_Onset_Usually_Occurring_in_Childhood_and_Adolescence = [item for item in ICD_List if item.startswith('F90') or item.startswith('F91') or item.startswith('F92') or item.startswith('F93') or item.startswith('F94') or item.startswith('F95') or item.startswith('F96') or item.startswith('F97') or item.startswith('F98')]
    data1 = data.copy()
    data1['disease'] = data1['ICD10_CODE'].apply(mark)
    data2 = data1.drop_duplicates(subset=['SUBJECT_ID','disease'])
    data2 = data2[data2['disease'] != 'Other']
    edges = [tuple(x) for x in data2[['SUBJECT_ID','disease']].values.tolist()]
    B = nx.Graph()
    B.add_nodes_from(data2['SUBJECT_ID'].unique(), bipartite=0, label='Person')
    B.add_nodes_from(data2['disease'].unique(), bipartite=1, label='Disease')
    for row in edges:
        B.add_edge(row[0], row[1], rating=1)
    X = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]

    # 生成自定义单侧投影图
    projected_graph = weighted_projected_graph(B, X)

    # 打印投影图的节点和边
    print("Projected Graph Nodes:", projected_graph.nodes())
    print("Projected Graph Edges:", projected_graph.edges(data=True))    
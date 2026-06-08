# Comparatif des strategies de chunking (retrieval dense fixe)


## Q1. What medications is the patient taking?

**words250**
The patient is not taking any regular prescription medications.
_Sources : 10_IM_Followup_2025_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.1, example patient 2 history clinical note.docx p.-, 01_ENT_Consult_2021_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.2_

**tokens512**
The patient is not taking any regular prescription medications.
_Sources : 10_IM_Followup_2025_EXPANDED.pdf p.1, example patient 2 history clinical note.docx p.-, 10_IM_Followup_2025_EXPANDED.pdf p.2, 01_ENT_Consult_2021_EXPANDED.pdf p.2, 01_ENT_Consult_2021_EXPANDED.pdf p.1_

**semantic**
The patient is not taking any regular prescription medications.
_Sources : 10_IM_Followup_2025_EXPANDED.pdf p.1, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 01_ENT_Consult_2021_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.2_


## Q2. What was the interpretation of the left cervical lymph node fine-needle aspiration pathology and flow cytometry findings from 2021?

**words250**
The fine-needle aspiration pathology and flow cytometry of the left cervical lymph node in 2021 showed a polymorphous lymphoid population with reactive features, including small mature lymphocytes, scattered centroblast-like forms, and occasional immunoblasts. There was no evidence of granulomas, necrosis, or metastatic carcinoma. Flow cytometry revealed a predominantly T-cell population with a CD4:CD8 ratio of approximately 3.13:1 and a small B-cell population without aberrant marker expression or light-chain restriction. Overall, there was no immunophenotypic evidence of a clonal B-cell lymphoproliferative disorder. The final interpretation was negative for malignancy.
_Sources : 03_FNA_Pathology_2021_EXPANDED.pdf p.1, 03_FNA_Pathology_2021_EXPANDED.pdf p.1, 01_ENT_Consult_2021_EXPANDED.pdf p.1, 02_US_Neck_2021_EXPANDED.pdf p.1, example patient 2 history clinical note.docx p.-_

**tokens512**
The fine-needle aspiration pathology and flow cytometry findings of the left cervical lymph node from 2021 were interpreted as negative for malignancy. The cytology showed a polymorphous lymphoid population with reactive features and no evidence of granulomas, necrosis, or metastatic carcinoma. Flow cytometry revealed a predominantly T-cell population with a CD4:CD8 ratio of approximately 3.13:1 and a small B-cell population without aberrant markers or clonal B-cell lymphoproliferative disorder. Overall, the findings supported a reactive lymphoid population. It was recommended to correlate with imaging and clinical course, with consideration for repeat sampling or excisional biopsy if lymphadenopathy persists or systemic symptoms develop.
_Sources : 03_FNA_Pathology_2021_EXPANDED.pdf p.1, 01_ENT_Consult_2021_EXPANDED.pdf p.1, 03_FNA_Pathology_2021_EXPANDED.pdf p.1, 02_US_Neck_2021_EXPANDED.pdf p.1, example patient 2 history clinical note.docx p.-_

**semantic**
The fine-needle aspiration pathology and flow cytometry findings of the left cervical lymph node in 2021 were interpreted as negative for malignancy. The results supported a reactive lymphoid population without evidence of a clonal B-cell lymphoproliferative disorder. The cytology showed a polymorphous lymphoid population with features consistent with reactive follicular activity, and flow cytometry demonstrated a predominantly T-cell population with a normal CD4:CD8 ratio and no aberrant marker expression.
_Sources : 03_FNA_Pathology_2021_EXPANDED.pdf p.1, 02_US_Neck_2021_EXPANDED.pdf p.1, 03_FNA_Pathology_2021_EXPANDED.pdf p.1, 01_ENT_Consult_2021_EXPANDED.pdf p.1, 04_MRI_Head_Neck_2021_EXPANDED.pdf p.1_


## Q3. What pulmonary nodule findings were reported on the 2022 CT chest exam and what follow-up surveillance was recommended?

**words250**
The 2022 CT chest exam reported multiple small solid pulmonary nodules bilaterally, with sizes up to 4 mm (right upper lobe 4 mm; right middle lobe 3 mm; right lower lobe subpleural 4 mm; left lower lobe 3 mm). These nodules showed no cavitation, ground-glass component, or suspicious spiculation, and the appearance favored a benign etiology in a low-risk patient. Mild emphysematous background changes were also noted.

The recommended follow-up surveillance was optional surveillance CT in 6 to 12 months depending on risk factors and shared decision-making. If prior imaging exists, comparison for stability was considered valuable. Shared decision-making was emphasized to balance radiation exposure, patient anxiety, and the likelihood of clinically meaningful change.
_Sources : 06_CT_Chest_2022_EXPANDED.pdf p.2, 06_CT_Chest_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.1_

**tokens512**
The 2022 CT chest exam reported multiple small solid pulmonary nodules bilaterally, with sizes ranging from 3 to 4 mm, located in various lobes without cavitation, ground-glass components, or suspicious spiculation. There was no associated lymphadenopathy or pleural abnormalities. The nodules were assessed as very small and most consistent with benign incidental findings, such as healed granulomas, with an overall low malignancy risk.

Regarding follow-up surveillance, routine imaging was considered optional and could be deferred to limit radiation exposure. Shared decision-making was recommended, taking into account factors such as radiation exposure, patient anxiety, and the likelihood of clinically meaningful change. Interval surveillance might be considered if there is uncertainty about risk factors, multiple nodules, or broader clinical concerns. The patient was advised to return if respiratory symptoms develop and was counseled on red flags warranting prompt reassessment.
_Sources : 06_CT_Chest_2022_EXPANDED.pdf p.2, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.1, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1_

**semantic**
The 2022 CT chest exam reported multiple small solid pulmonary nodules bilaterally, with sizes up to 4 mm (including nodules in the right upper lobe, right middle lobe, right lower lobe subpleural area, and left lower lobe). These nodules showed no cavitation, ground-glass component, or suspicious spiculation and were considered to favor a benign etiology in a low-risk patient. Mild emphysematous changes were also noted.

The recommended follow-up surveillance was optional, with consideration for a surveillance CT scan in 6 to 12 months depending on risk factors and shared decision-making. If prior imaging exists, comparison for stability was advised. Surveillance imaging may be deferred to limit radiation exposure, and clinical judgment should balance radiation risks, patient anxiety, and the likelihood of clinically meaningful change.
_Sources : 06_CT_Chest_2022_EXPANDED.pdf p.2, 06_CT_Chest_2022_EXPANDED.pdf p.1, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.1_


## Q4. Quelle surveillance ou suivi a ete recommande pour les nodules pulmonaires?

**words250**
Une surveillance optionnelle par tomodensitométrie (CT) est recommandée dans un délai de 6 à 12 mois, en fonction des facteurs de risque et d'une prise de décision partagée. Si des images antérieures existent, leur comparaison pour évaluer la stabilité des nodules est utile.
_Sources : 06_CT_Chest_2022_EXPANDED.pdf p.2, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1_

**tokens512**
La surveillance recommandée pour les nodules pulmonaires consiste en une tomodensitométrie (CT) de suivi optionnelle dans un délai de 6 à 12 mois, en fonction des facteurs de risque et après une prise de décision partagée. Ce suivi peut être différé pour limiter l'exposition aux radiations. En l'absence de symptômes respiratoires, le patient doit revenir en consultation au besoin. Il est également conseillé de comparer avec des imageries antérieures si elles existent, afin d’évaluer la stabilité des nodules.
_Sources : 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.2, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 10_IM_Followup_2025_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1_

**semantic**
La surveillance recommandée pour les nodules pulmonaires consiste en une tomodensitométrie (CT) de suivi optionnelle à réaliser dans un délai de 6 à 12 mois, en fonction des facteurs de risque et après une prise de décision partagée. Si des examens antérieurs existent, leur comparaison pour évaluer la stabilité des nodules est utile. Le suivi peut être différé afin de limiter l'exposition aux radiations. Le patient doit également être informé des signes d'alerte (nouvelle toux, hémoptysie, perte de poids inexpliquée) et revenir rapidement en cas d'apparition de symptômes respiratoires.
_Sources : 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.2, 07_Pulmonology_Consult_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1, 06_CT_Chest_2022_EXPANDED.pdf p.1_

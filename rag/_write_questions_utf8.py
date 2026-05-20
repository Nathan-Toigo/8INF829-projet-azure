#!/usr/bin/env python3
from pathlib import Path

CONTENT = """# Questions d'évaluation RAG — dossier AURALIS (synthétique)
# Projet 8INF829 — documents dans docs/
# Encodage : UTF-8
#
# Usage :
#   cd rag
#   python run.py ask "votre question"
#   python run.py chat
#
# Patient principal : AURALIS TEST PATIENT (synthétique) — dossier 01 à 10
# Dossier secondaire : example patient 2 (fichiers example_*)

================================================================================
QUESTIONS PAR DOCUMENT
================================================================================

--------------------------------------------------------------------------------
01_ENT_Consult_2021_EXPANDED.pdf — Consultation ORL (2021)
--------------------------------------------------------------------------------
1. Quel était le motif principal de consultation ORL en 2021 ?
2. Quels signes ou symptômes cervicaux ou ORL étaient rapportés à la consultation initiale ?
3. Quelle était l'impression clinique ou le plan de suivi proposé par l'ORL à ce stade ?
4. Y avait-il des antécédents pertinents mentionnés lors de la consultation ENT ?

--------------------------------------------------------------------------------
02_US_Neck_2021_EXPANDED.pdf — Échographie cou (2021)
--------------------------------------------------------------------------------
5. Quels ganglions ou structures ont été décrits à l'échographie du cou ?
6. Quelles étaient les dimensions ou caractéristiques des adénopathies cervicales à l'US ?
7. Quelle recommandation de suivi ou d'examen complémentaire suit ce rapport d'échographie ?
8. L'échographie a-t-elle identifié des lésions thyroïdiennes ou d'autres findings hors ganglions ?

--------------------------------------------------------------------------------
03_FNA_Pathology_2021_EXPANDED.pdf — Cytopathologie FNA (2021)
--------------------------------------------------------------------------------
9. Quelle était l'interprétation du prélèvement par ponction à l'aiguille fine du ganglion cervical gauche ?
10. Quels résultats de cytométrie en flux ou immunophénotypage sont rapportés ?
11. Le diagnostic était-il suffisant pour exclure une lymphoprolifération, ou un complément était-il requis ?
12. Quelle corrélation clinique ou recommandation le pathologiste a-t-il indiquée ?

--------------------------------------------------------------------------------
04_MRI_Head_Neck_2021_EXPANDED.pdf — IRM tête et cou (2021)
--------------------------------------------------------------------------------
13. Quelles adénopathies cervicales sont décrites à l'IRM et dans quelles chaînes ?
14. Y a-t-il des anomalies intracrâniennes ou des lésions de la sphère ORL sur cette IRM ?
15. Quelle est l'impression radiologique globale de l'IRM tête et cou ?
16. Des examens de surveillance ou une biopsie étaient-ils recommandés après l'IRM ?

--------------------------------------------------------------------------------
05_GI_Endoscopy_2022_EXPANDED.pdf — Endoscopie digestive (2022)
--------------------------------------------------------------------------------
17. Quel était l'indication de l'endoscopie digestive en 2022 ?
18. Quels résultats à l'endoscopie haute et/ou basse sont documentés ?
19. Des biopsies ont-elles été prises et quels en sont les résultats ou la conclusion ?
20. Y a-t-il un plan de surveillance digestive lié aux lymphadénopathies ou à un autre contexte ?

--------------------------------------------------------------------------------
06_CT_Chest_2022_EXPANDED.pdf — CT thorax (2022)
--------------------------------------------------------------------------------
21. Quels nodules pulmonaires ont été rapportés au CT thorax de 2022 et quelles tailles ?
22. Quelle surveillance ou suivi a été recommandé pour les nodules pulmonaires ?
23. Y avait-il des adénopathies médiastinales ou hilaires sur le CT thorax ?
24. Quels autres findings thoraciques sont mentionnés (plèvre, emphysème, etc.) ?

--------------------------------------------------------------------------------
07_Pulmonology_Consult_2022_EXPANDED.pdf — Consultation pneumologie (2022)
--------------------------------------------------------------------------------
25. Comment la pneumologie a-t-elle interprété les nodules pulmonaires du CT ?
26. Quel plan de suivi pulmonaire a été proposé après la consultation de 2022 ?
27. Des symptômes respiratoires ou un tabagisme sont-ils documentés dans la note pneumo ?
28. Y a-t-il eu coordination avec d'autres spécialités (ORL, médecine interne) ?

--------------------------------------------------------------------------------
08_Lab_Trends_2021_2025_EXPANDED.pdf — Tendances laboratoire (2021–2025)
--------------------------------------------------------------------------------
29. Quelles anomalies de numération ou de formule sanguine apparaissent dans les tendances ?
30. Y a-t-il une évolution des marqueurs inflammatoires ou de la fonction rénale sur la période ?
31. Des résultats de LDH, protéines ou immunoglobulines sont-ils mentionnés et comment évoluent-ils ?
32. Quels résultats de laboratoire pourraient soutenir ou infirmer une lymphoprolifération ?

--------------------------------------------------------------------------------
09_Urology_Workup_2021_2025_EXPANDED.pdf — Bilan urologique / hématurie (2021–2025)
--------------------------------------------------------------------------------
33. Quel est le statut du bilan d'hématurie microscopique du patient ?
34. Quels examens urologiques (cytologie, imagerie, cystoscopie) ont été réalisés et quels en sont les résultats ?
35. Quel plan de surveillance urologique est prévu jusqu'en 2025 ?
36. L'hématurie est-elle considérée comme résolue, stable ou nécessitant un suivi actif ?

--------------------------------------------------------------------------------
10_IM_Followup_2025_EXPANDED.pdf — Suivi médecine interne (2025)
--------------------------------------------------------------------------------
37. Quelle est l'évaluation de médecine interne concernant les lymphadénopathies persistantes en 2025 ?
38. Quel plan de prise en charge global (examens, spécialistes, surveillance) est proposé en mars 2025 ?
39. Quels problèmes actifs sont listés dans la note de suivi IM ?
40. Y a-t-il des changements de médication ou de priorisation diagnostique en 2025 ?

--------------------------------------------------------------------------------
example patient 2 history clinical note.docx — Dossier patient 2 (note clinique)
--------------------------------------------------------------------------------
41. Quels antécédents médicaux sont documentés pour le patient 2 ?
42. Quel est le motif de consultation ou le problème principal dans la note clinique du patient 2 ?
43. Quels diagnostics ou hypothèses sont mentionnés pour le deuxième patient ?

--------------------------------------------------------------------------------
example_patient_2.pdf — Dossier patient 2 (PDF)
--------------------------------------------------------------------------------
44. Quelles informations démographiques ou d'identification figurent pour le patient 2 ?
45. Quels éléments du dossier PDF du patient 2 diffèrent du dossier AURALIS ?
46. Y a-t-il un plan de traitement ou de suivi spécifique au patient 2 ?

================================================================================
QUESTIONS TRANSVERSALES (plusieurs documents)
================================================================================

47. Résumez l'évolution de la lymphadénopathie cervicale du patient AURALIS de 2021 à 2025.
48. Les résultats de FNA 2021 sont-ils cohérents avec l'IRM cou et l'échographie du cou ?
49. Le CT thorax de 2022 et la consultation pneumologie sont-ils alignés sur la même stratégie de suivi ?
50. Quels examens restent en attente ou à répéter selon l'ensemble du dossier ?
51. Y a-t-il des contradictions entre les rapports de laboratoire et les notes de spécialistes ?
52. Quels problèmes de santé nécessitent un suivi actif en 2025 (liste par spécialité) ?
53. Le patient AURALIS présente-t-il des findings pulmonaires nécessitant un suivi radiologique structuré ?
54. Quelle est la chronologie des examens d'imaging majeurs (US, IRM, CT) pour ce patient ?

================================================================================
QUESTIONS COURTES (tests rapides RAG)
================================================================================

55. Taille du plus grand nodule pulmonaire au CT 2022 ?
56. Date de l'examen CT thorax ?
57. MRN ou identifiant du patient AURALIS ?
58. Le dossier indique-t-il qu'il s'agit de données synthétiques / non cliniques ?

# Fin du fichier — 58 questions
"""

out = Path(__file__).resolve().parent.parent / "docs" / "QUESTIONS_RAG.txt"
out.write_text(CONTENT, encoding="utf-8", newline="\n")
out.read_bytes().decode("utf-8")
print("UTF-8 OK:", out)

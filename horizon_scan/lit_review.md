# Statement of work
The goal is to create, from a network graph of research papers, a review of literature using ChatGPT's API. For the first version, I will only be looking at research papers on the topic of tree-based modeling techniques in conflict modeling. 

## Network 
The preliminary network of 26 papers was generated from the origin node of a paper from 2016 titled: "Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data"
- Note that this means my review of literature will only span up to 2016. Next goal is to recreate this with a more recent paper (and/or expand my network trawler to not only look at references, but also citations for that paper).
- Note^2: nvm i fixed it to also look at citations.

A visualization of the preliminary 26 paper network is below. 
![Alt text](images/rf26.png)

## Prompting
As suggested by GPT: 

"""
I have compiled a list of research papers focusing on the use of tree-based modeling techniques in conflict research. I am seeking a comprehensive literature review based on these papers. The review should highlight key methodologies, findings, and trends over time. Below is a summary of the most significant papers:

1. Title: "Understanding Conflicts through Random Forest Analysis"
   Authors: Smith, J., Doe, A.
   Year: 2019
   Summary: This paper uses random forest models to analyze conflict patterns in region X, revealing insights into...

2. Title: "Gradient Boosting in Predicting Conflict Outcomes"
   Authors: Lee, K., Johnson, M.
   Year: 2021
   Summary: A study that employs gradient boosting methods to predict the outcomes of political conflicts in...

[...additional summaries...]

Please provide a detailed literature review based on these papers, focusing on the methodologies used, the evolution of these techniques over time, and their effectiveness in conflict analysis.
"""


## Result: AI-Generated Review of Literature
In the literature concerning the use of tree-based algorithms for modeling armed civil conflicts, the evolution of methodologies and their effectiveness in various contexts have been extensively studied. This review synthesizes the findings and methodologies utilized across several research papers spanning from 2008 to 2022.

The earliest work by Weidmann (2008) laid the groundwork for employing machine learning in the prediction of conflict. By using ensemble classifiers and focusing on decision tree learning, attention was drawn to addressing the issue of rare events in conflict data - a recurring theme in subsequent studies. 

Siroky (2009) pursued a more in-depth exploration of Random Forests, tracing its intellectual history and providing an understanding of its non-parametric nature. This marks an early indication of a move away from traditional logistic regression models, which aligns with the findings of Muchlinski (2016) that randomized forests outperform logistic regression in predicting civil war onsets.

As methods advanced, the literature indicated an emerging preference for algorithmic approaches over classical statistical methods due to their superior predictive capabilities. Jones (2016) introduced 'edarf', elucidating how Random Forests can be applied in exploratory data analysis to detect complex interactions in conflict data.

Freiman (2010) incorporated Simulated Annealing alongside Random Forests to refine parameter selection in predictive models. This study emphasizes the potential pitfalls of overfitting, suggesting a need for careful model evaluation.

The role of Random Forests in conflict prediction continued to expand, as indicated by the study of Ettensperger (2021), which further corroborated the utility of multi-model ensemble methods. The shift towards ensembles suggests that combining various tree-based models can harness different strengths and potentially lead to improved predictions.

Gradient boosting entered the conversation with the work of Vestby (2022), indicating that while Random Forests might be superior in terms of mean square error, Gradient Boosting might be more effective in specific predictive scenarios. This highlights the evolution of tree-based methodologies and the importance of context and performance metrics in evaluating these models.

Further refining the predictive capabilities, Metternich (2019) presented an actor-centric view focusing on the characteristics of rebel organizations, emphasizing the role of actors in predicting conflict severity through Random Forests.

Alongside these advancements, significant contributions were made to address technical challenges. Darbon (2021) proposed an efficient algorithm to handle high-dimensional sparse logistic regression, hinting at the growing importance of computational efficiency in large-scale conflict data analysis.

Parallel to the methodological advances, scholars such as Schellens (2020) and Henrickson (2020) have been employing these tree-based algorithms to uncover underlying patterns in conflict data, ranging from the role of natural resources in violent conflicts to the computation of expected war costs.

Several studies (e.g., Montgomery, 2018; Musumba, 2021) underscore the applicability of machine learning and tree-based models beyond predictions, also using them to ascertain the relative importance of features and improve our theoretical understanding of conflicts. 

In conclusion, tree-based algorithms have significantly evolved and increasingly demonstrated their suitability for modeling complex phenomena such as armed civil conflicts over the last decades. From addressing the rarity of certain events to leveraging ensemble methods for refined predictions, these algorithms have been confirmed to be powerful tools for conflict analysis. Their flexibility and capability to capture non-linear interactions make them superior to traditional regression methods in many scenarios. This evolution has led to a more nuanced and sophisticated understanding of the factors driving armed conflict, permitting researchers to make increasingly accurate predictions, and allowing policymakers to respond more effectively to the threat of civil war.

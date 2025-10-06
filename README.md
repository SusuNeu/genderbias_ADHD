# genderbias_ADHD
This project addresses gender bias in ADHD diagnoses. 
The code covers the analyses as described in the publication  
"Toward a fair, gender-debiased classifier for the diagnosis of attention deficit/hyperactivity disorder- a Machine-Learning based classification study"


## Description
This project addresses gender bias in ADHD diagnoses. 
The analysis is implemented using Python and includes scikit-learn components for machine learning tasks.
For bias mitigation, algorithms of the AI Fairness 360 repository are used. 


## Prerequisites
- Python 3.x
- Jupyter Notebook
- scikit-learn
- AI fairness 360
- shap, lime


## Installation
```bash
# Clone the repository
git clone [repository -URL]

# Install required packages
pip install -r requirements.txt
```

## Usage
1. Open the python or jupyter notebook file:
```bash
py GenderBias_ADHD.py
code GenderBias_ADHD.ipynb
```


## Project Structure
## file overview
- `GenderBias_ADHD.py`: Main file containing the analysis
- 'GenderBias_ADHD.ipynb': Main file as jupyter notebook
- 'CMI_400.csv': fMRI and clincal training data 
- 'CMI_400_test.csv': fMRI and clincal test data
- 'dtrain.csv': structural MRI training data (radiomic features)
- 'dtest.csv': structural MRI test data (radiomic features)
- 'license.txt': MIT license
- 'requirements.txt': requirements
- 'README.md': instructions and information

## Functions Overview
### (1) 'general functions' part covers 
- training/evaluation of SVM and XGBoost models  
- bias metrics determination
- XAI

### (2) 'creating input' part includes 
### Preprocessing functions
- loading and cleaning of the data
- creating binary datasets
- split and preprocess data using SMOTE and StandardScaler

### (3) 'Debiasing Algorithms' includes 
### Analysis Functions for 5 different algorithms 
- Adversarial Debiasing
- Prejudice Remover
- Calibrated Equal Odds Post-Processing
- Reweighing
- Disparate Impact Remover 

### (4) Main Programme includes
- data load functions
- perform analysis

### output
- model performance: balanced accuracy
- Bias metrics: mean difference, 
		disparate impact ratio
		average odds difference 
		statistical parity difference
		Equal opportunity difference 
		Theil Index
- XAI:	SHaP: summary bar plot
	LIME: notebook incl table   


## Contact
suneu@tutanota.com





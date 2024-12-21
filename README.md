# AI-Powered Resume Parser and Job Matcher

## Abstract
The AI-powered Resume Parser and Job Matcher project aims to address recruitment inefficiencies by automating resume parsing and job matching. By leveraging Natural Language Processing (NLP) models such as BERT/RoBERTa, Spacy NER, and XGBoost, this project extracts relevant information from resumes and matches candidates with job descriptions. The solution is deployed through a Gradio-based web interface, providing a user-friendly experience. The project demonstrates over 50% accuracy in both resume data extraction and job matching, validating its effectiveness.

## Introduction
Recruitment processes are often slowed down by the manual effort involved in screening resumes. This project automates the parsing of resumes and the matching of candidates to job descriptions using state-of-the-art NLP techniques and machine learning models. The system extracts structured data from resumes, matches it to job roles, and delivers the results via a Gradio-based interface. Additionally, data visualizations and performance evaluations provide insights into the system's effectiveness.

## CRISP-DM Methodology
The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

### 1. Business Understanding:
- **Objective**: Automate the recruitment process by building a resume parser and job matcher.
- **Success Metrics**: Achieved 50% extraction accuracy and job matching validated by recruiters.

### 2. Data Understanding:
- **Sources**: Resumes in PDF, DOC, DOCX formats, and job descriptions sourced from LinkedIn and curated datasets.
- **Initial Analysis**: Performed Exploratory Data Analysis (EDA) using word clouds, histograms, and term frequency distributions related to job roles.

### 3. Data Preparation:
- **Techniques Used**: 
  - SpaCy, NLTK, and custom Python functions for tokenization, lemmatization, and stopword removal.
  - Regex and SpaCy NER models for extracting names, emails, education, and skills.
  - SMOTE for addressing class imbalance and generating synthetic data.
  - Custom parsers for handling diverse resume formats.

### 4. Modeling:
- **Models Used**:
  - Fine-tuned pre-trained BERT and RoBERTa models via Hugging Face Transformers.
  - Applied XGBoost with hyperparameter tuning and early stopping for better model performance.
  - Mitigated class imbalance using class-weighted sampling.

### 5. Evaluation:
- **Metrics**: Precision, recall, F1-score, ROC-AUC, and custom performance metrics.
- **Visualization Tools**: TensorBoard for tracking training progress, confusion matrices, and classification reports.
- **Analysis**: Feature importance analysis to understand the key features influencing the predictions.

### 6. Deployment:
- **Deployment Pipeline**: Integrated the trained model into a Gradio-based web interface.
- **Functionality**: Users can upload resumes and job descriptions, view extraction results, and get matching scores in real time.
- **Scalability**: REST API endpoints were implemented for future scalability.

### 7. Model Retraining:
- **Retraining Logic**: The model can be retrained automatically with new datasets.
- **Continuous Improvement**: Data versioning and retraining ensure the model remains state-of-the-art.

## Detailed Analysis

### Data Balancing:
- **SMOTE**: Applied to balance class distribution, enhancing the representation of the minority class.
- **Sampling Strategies**: Implemented class-specific sampling strategies to address imbalance in the dataset.

### Evaluation Metrics:
- **Reports**: Generated detailed classification reports using `scikit-learn`'s `classification_report` and `confusion_matrix`.
- **Visualization**: Used Seaborn and Matplotlib to visualize precision, recall, F1-scores, and feature importance.

### Model Training Details:
- **XGBoost**: Utilized CUDA support for accelerated training and class-weighted DMatrix for better model performance.
- **Hyperparameter Tuning**: Tuned learning rate, max depth, gamma, min_child_weight, and max_delta_step.
- **Early Stopping**: Implemented early stopping and checkpoint saving to avoid overfitting.

### Data Visualization:
- **EDA**: Created word clouds, histograms, and bar plots to understand data distributions.
- **SMOTE Visualization**: Visualized class distributions before and after SMOTE.
- **Feature Importance**: Plotted the importance of features and prediction breakdowns to provide insights into the model's decisions.

## Colab Link

You can access the project in Google Colab using the following link:

[AI-Powered Resume Parser and Job Matcher on Colab](https://colab.research.google.com/drive/161pF2qfsgN2y0UYrmCGZafcC_joffWCn?usp=sharing)

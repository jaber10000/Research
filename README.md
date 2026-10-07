# 🔬 Research & Academic Projects

Welcome to my **Research Repository**.

This repository contains my research projects, academic works, implementations, experiments, source code, research papers, and supporting materials in the fields of **Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, Federated Learning, Privacy-Preserving AI, and Explainable AI**.

The repository is organized by research topic, with each project containing its corresponding **research paper/report, source code, notebooks, experiments, results, and documentation** where available.

---

## 👨‍💻 About My Research

My research interests focus on developing practical and responsible AI systems, particularly in areas where **privacy, interpretability, distributed learning, and low-resource language processing** are important.

My major research interests include:

* 🤖 Artificial Intelligence & Machine Learning
* 🧠 Deep Learning
* 🏥 Medical Artificial Intelligence
* 🔐 Privacy-Preserving AI
* 🌐 Federated Learning
* 👁️ Computer Vision
* 📝 Natural Language Processing
* 🇧🇩 Bengali/Bangla NLP
* 🚨 Fake News & Misinformation Detection
* 🔍 Explainable AI (XAI)
* 🧩 Multi-Task Learning
* 🧬 Medical Image Analysis

---

# 📚 Research Projects

## 1. 🧠 Privacy-Preserving Explainable Federated Transfer Learning Framework for Brain Tumor Segmentation in Distributed Healthcare Systems

### Research Area

**Federated Learning · Medical Imaging · Computer Vision · Privacy-Preserving AI · Deep Learning**

### Overview

This research investigates a **privacy-preserving federated learning framework for brain tumor segmentation from MRI images** in distributed healthcare environments.

The research explores how deep learning models can perform medical image segmentation while keeping healthcare data distributed rather than requiring all data to be collected in a centralized location.

The framework investigates both **centralized and federated learning settings**, including different data-distribution scenarios such as **IID and Non-IID data**.

The research also explores CNN-based and Transformer-based segmentation architectures and evaluates their performance using standard medical image segmentation metrics.

### Key Objectives

* Develop a deep learning framework for brain tumor segmentation.
* Investigate federated learning for distributed medical image analysis.
* Compare centralized and federated learning approaches.
* Study model performance under IID and Non-IID data distributions.
* Investigate privacy-preserving collaborative learning for healthcare.
* Evaluate segmentation performance using multiple evaluation metrics.

### Models & Techniques

* CNN-based U-Net
* Transformer-based U-Net
* Federated Learning
* FedAvg
* Centralized Learning
* IID Learning
* Non-IID Learning
* MRI Image Processing
* Medical Image Segmentation

### Data Processing

The research includes preprocessing steps such as:

* Image resizing
* Image normalization
* Duplicate removal
* Binary mask preparation
* Dataset partitioning
* Client-based data distribution

The implemented federated environment uses multiple simulated clients, where data partitions represent independent local training processes.

### Federated Learning

The federated learning workflow follows the general process:

```text
Global Model
     ↓
Distribute Model
     ↓
Local Client Training
     ↓
Local Model Updates
     ↓
Model Aggregation
     ↓
Updated Global Model
     ↓
Repeat
```

The research uses **FedAvg-style aggregation** for combining local model updates.

### Evaluation Metrics

* Dice Score
* Intersection over Union (IoU)
* Precision
* Recall
* F1-Score



### Technologies

`Python` `TensorFlow/Keras` `Deep Learning` `CNN` `U-Net` `Transformer` `Federated Learning`

---

# 2. 📰 An Explainable Hybrid Deep Learning Framework for Bangla Fake News Detection Using Transformer-Based Language Models

### Research Area

**Natural Language Processing · Bangla NLP · Fake News Detection · Transformer Models · Explainable AI**

### Overview

This research focuses on developing an **explainable Bangla fake news detection framework** using transformer-based language models and hybrid machine learning approaches.

The study investigates transformer-based models for understanding Bangla news text and compares them with traditional machine learning approaches.

The research also incorporates **Explainable AI (XAI)** to provide insight into why a particular news item is classified as fake or authentic.

### Research Objectives

* Develop a Bangla fake news detection system.
* Investigate transformer-based language models for Bangla NLP.
* Compare transformer models with traditional machine learning approaches.
* Improve classification performance through hybrid approaches.
* Incorporate Explainable AI into fake news detection.
* Analyze the features contributing to model predictions.

### Dataset

The research uses the **BanFakeNews dataset**, which contains Bangla news articles with authentic and fake news examples.

The dataset includes information such as:

* Article ID
* News domain
* Publication date
* Category
* Headline
* Content
* Fake/Authentic label

The dataset contains significant class imbalance, making preprocessing and model evaluation important parts of the research.

### Models & Techniques

#### Traditional Machine Learning

* Logistic Regression
* Linear SVM
* Random Forest
* LightGBM
* XGBoost
* Multinomial Naive Bayes

#### Transformer Models

* Bangla-BERT
* XLM-RoBERTa

#### Deep Learning Components

* Attention Pooling
* Transformer-based representations

### Explainable AI

The research incorporates:

**LIME — Local Interpretable Model-Agnostic Explanations**

LIME is used to identify important words/features that contribute to individual model predictions, helping make the classification process more interpretable.

### Evaluation

The research evaluates models using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

### Research Workflow

```text
Bangla News Dataset
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Feature Representation
        ↓
Traditional ML / Transformer Models
        ↓
Model Training
        ↓
Evaluation
        ↓
Explainable AI
        ↓
Prediction Interpretation
```


### Technologies

`Python` `Scikit-learn` `Transformers` `Bangla-BERT` `XLM-RoBERTa` `NLP` `LIME`

---

# 3. 🇧🇩 Category-Guided Intelligence: An Explainable Multi-Task Framework for Bengali Fake News Detection

### Research Area

**Bengali NLP · Fake News Detection · Multi-Task Learning · Transformer Models · Explainable AI**

### Overview

This research investigates an **explainable category-aware framework for Bengali fake news detection**.

The research explores whether information about the **category of a news article** can provide additional information for the primary fake/real classification task.

The proposed framework combines contextual language representation, sequential modeling, attention mechanisms, and category-aware multi-task learning.

### Dataset

The research starts with a Bengali fake-news dataset containing **11,839 records**.

After data-quality assessment, missing-value handling, duplicate removal, Bengali text normalization, and other preprocessing operations, **10,436 instances** were retained for experimentation.

The dataset contains multiple categories of Bengali news and includes information such as:

* Category
* Headline
* Location
* Content
* Label

### Research Objectives

* Develop a Bengali misinformation detection framework.
* Establish traditional machine learning baselines.
* Evaluate BanglaBERT as a contextual baseline.
* Develop a Category-Aware Hybrid Transformer.
* Integrate category information through multi-task learning.
* Investigate attention mechanisms for better contextual representation.
* Perform hyperparameter optimization.
* Conduct ablation studies.
* Apply Explainable AI techniques.
* Perform error analysis on model predictions.

### Baseline Models

The research evaluates traditional machine learning approaches using TF-IDF representations:

* Logistic Regression
* Linear SVM
* XGBoost

A **BanglaBERT-based baseline** is also investigated.

### Proposed Framework

The proposed Category-Aware Hybrid Transformer combines:

```text
                 Bengali News
                      ↓
                Preprocessing
                      ↓
                  BanglaBERT
                      ↓
               Contextual Features
                      ↓
                 BiLSTM Layer
                      ↓
               Attention Mechanism
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
   Fake/Real Task          Category Task
          ↓                       ↓
          └───────────┬───────────┘
                      ↓
             Multi-Task Learning
```

### Main Components

* BanglaBERT
* BiLSTM
* Attention Mechanism
* Category-Aware Learning
* Multi-Task Learning
* Hyperparameter Optimization
* Ablation Study

### Explainable AI

The research incorporates multiple explainability approaches:

* LIME
* SHAP
* Attention Visualization

These techniques are used to investigate which textual features contribute to model predictions.

### Experimental Analysis

The research includes:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC
* Five-fold Cross-Validation
* Ablation Study
* Error Analysis
* False Positive Analysis
* False Negative Analysis
* Category-wise Error Analysis



### Technologies

`Python` `PyTorch/TensorFlow` `BanglaBERT` `BiLSTM` `Transformers` `NLP` `LIME` `SHAP`

---

# 🏥 4. Privacy-Preserving Colorectal Cancer Detection

### Research Area

**Federated Learning · Medical AI · Deep Learning · Privacy-Preserving AI**

### Overview

This research focuses on applying **federated learning to colorectal cancer detection** in a privacy-preserving healthcare setting.

The research investigates how distributed healthcare data can be used to collaboratively train machine learning models without requiring all sensitive data to be centralized.

### Key Concepts

* Federated Learning
* Privacy-Preserving Machine Learning
* Medical AI
* Deep Learning
* Distributed Healthcare Data
* FedAvg

### Research Direction

```text
Healthcare Institution A ──┐
                           │
Healthcare Institution B ──┼──→ Federated Server
                           │          ↓
Healthcare Institution C ──┘     Global Model
                                      ↓
                              Updated Model


# 🧪 Research Methodology

Across my research projects, I follow a structured research workflow:

```text
Research Problem
      ↓
Literature Review
      ↓
Dataset Collection
      ↓
Data Quality Assessment
      ↓
Data Preprocessing
      ↓
Baseline Development
      ↓
Model Development
      ↓
Hyperparameter Optimization
      ↓
Experimental Evaluation
      ↓
Explainability Analysis
      ↓
Ablation / Error Analysis
      ↓
Results & Discussion
      ↓
Research Documentation
```

---

# 🛠️ Technologies & Tools

## Programming Languages

* Python
* SQL
* C

## Machine Learning

* Scikit-learn
* Pandas
* NumPy

## Deep Learning

* TensorFlow
* Keras
* PyTorch

## Natural Language Processing

* Transformers
* BanglaBERT
* XLM-RoBERTa
* LangChain

## Computer Vision

* OpenCV
* CNN
* U-Net
* Medical Image Segmentation

## Federated Learning

* FedAvg
* Distributed Training
* IID / Non-IID Learning

## Explainable AI

* LIME
* SHAP
* Attention Visualization

## Data Analysis & Visualization

* Matplotlib
* Seaborn
* Jupyter Notebook
* Excel
* Power BI

## Development Tools

* Git
* GitHub
* VS Code
* Jupyter Notebook

#5. 🏥 Privacy-Preserving Federated Learning Framework for Primary Pancreatic Malignancy Detection
Research Area

Federated Learning · Healthcare AI · Medical Data Mining · Privacy-Preserving AI · Deep Learning

###Overview

This research investigates a privacy-preserving federated learning framework for primary pancreatic malignancy detection using structured clinical records from the SEER dataset. The framework enables multiple healthcare institutions to collaboratively train predictive models without sharing raw patient data. The study compares centralized and federated learning approaches while preserving patient privacy through distributed model training and FedAvg-based aggregation. The proposed system demonstrates that high diagnostic performance can be achieved while maintaining compliance with healthcare privacy requirements.

Key Objectives
Develop a privacy-preserving framework for pancreatic malignancy detection.
Enable collaborative model training without sharing patient data.
Compare centralized and federated learning performance.
Preserve healthcare data confidentiality during training.
Evaluate federated learning in distributed healthcare environments.
Analyze model effectiveness for clinical decision support.
Models & Techniques
Artificial Neural Network (ANN)
Federated Learning
FedAvg Algorithm
Centralized Learning
Flower Framework
Binary Classification
Distributed Healthcare Training
Privacy-Preserving AI
Data Processing

The research includes:

Missing value handling
Label encoding
Data normalization
Feature selection
Class balancing
Client-wise data partitioning
Training/testing split
Multi-hospital simulation using SEER data
Federated Learning Workflow
Plain Text
Global Model
↓
Distribute to Clients
↓
Local ANN Training
↓
Model Weight Updates
↓
FedAvg Aggregation
↓
Updated Global Model
↓
Repeat
Evaluation Metrics
Accuracy
Precision
Recall
Confusion Matrix
Loss Analysis
Technologies

Python TensorFlow Keras Federated Learning Flower ANN FedAvg Pandas Scikit-Learn

#6. 🔬 Ensemble Deep Learning Approach for Leukemia Stage Classification from Microscopic Blood Cell Images
Research Area

Medical Imaging · Computer Vision · Deep Learning · Explainable AI · Cancer Diagnosis

###Overview

This research proposes an ensemble deep learning framework for leukemia stage classification from microscopic peripheral blood smear images. The study combines ResNet18 and VGG16 architectures to classify four leukemia stages: Benign, Early Pre-B, Pre-B, and Pro-B. Extensive preprocessing, augmentation, class balancing, and explainable AI methods were used to improve classification performance and model interpretability. The proposed ensemble model achieved superior performance compared to multiple baseline CNN architectures.

Key Objectives
Develop an ensemble deep learning framework for leukemia staging.
Classify four leukemia stages from PBS images.
Improve classification accuracy using model fusion.
Address class imbalance through augmentation and oversampling.
Enhance model interpretability through explainable AI.
Compare ensemble performance with baseline CNN models.
Models & Techniques
ResNet18
VGG16
Ensemble Learning
Soft Voting
Guided Backpropagation
Data Augmentation
Weighted Random Sampling
CNN-Based Classification
Data Processing

The research includes:

Image resizing (224×224)
Image normalization
Data augmentation
Class balancing
Oversampling
WeightedRandomSampler
Noise reduction
Training/Validation/Test splitting
Ensemble Learning Workflow
Plain Text
PBS Images
↓
Preprocessing
↓
ResNet18
↓
Feature Extraction
↓
Soft Voting Fusion
↓
Final Prediction
↑
VGG16
↓
Feature Extraction
Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report
Explainability

The framework incorporates Guided Backpropagation to generate saliency maps, helping visualize the cellular regions most influential for stage prediction and improving model interpretability.

Technologies

Python PyTorch Deep Learning ResNet18 VGG16 Ensemble Learning Computer Vision Guided Backpropagation


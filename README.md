
# Stress Detection System Using Machine Learning

A multimodal machine-learning project designed to identify potential stress indicators through text, facial expressions, and voice-based analysis.

## 📌 Project Overview

The **Stress Detection System** is a machine-learning project that explores how different types of input data can be used to identify patterns associated with stress.

The system uses three analysis modules:

- **Text-based detection:** Uses Natural Language Processing (NLP) to analyze textual input.
- **Facial expression-based detection:** Uses computer vision and facial emotion analysis to identify visual patterns.
- **Voice-based detection:** Extracts acoustic features from speech to analyze vocal patterns.

The project applies Logistic Regression models to the extracted features and evaluates the performance of each modality separately.

> **Disclaimer:** This project is intended for educational and research purposes. Its predictions are experimental and should not be used as a medical diagnosis or as a substitute for professional mental health assessment.

## 🎯 Objectives

- Explore machine-learning approaches for stress-related pattern recognition.
- Analyze text, facial expressions, and voice as different sources of information.
- Apply preprocessing and feature extraction techniques to prepare data for classification.
- Train and evaluate Logistic Regression models.
- Compare model performance across the three modalities.
- Understand the potential applications and limitations of machine learning in stress-related analysis.

## ✨ Features

### 1. Text-Based Stress Detection
- Processes textual data using NLP techniques.
- Converts text into numerical features using TF-IDF.
- Uses Logistic Regression for classification.

### 2. Facial Expression-Based Analysis
- Uses OpenCV for computer vision processing.
- Extracts facial expression and emotion-related features.
- Applies Logistic Regression to the extracted features.

### 3. Voice-Based Stress Detection
- Processes speech audio and extracts acoustic features.
- Uses MFCC (Mel-Frequency Cepstral Coefficients) through Librosa.
- Applies Logistic Regression for classification.

### 4. Model Evaluation
- Uses an 80:20 training-testing split.
- Applies 10-fold cross-validation.
- Evaluates accuracy, F1-score, and cross-validation performance.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Logistic Regression |
| Natural Language Processing | NLP, TF-IDF |
| Computer Vision | OpenCV |
| Audio Processing | Librosa, MFCC |
| Data Processing | NumPy, Pandas |
| Model Evaluation | Accuracy, F1-score, 10-fold cross-validation |
| Development | Caffeine AI Builder, Python-based ML workflow |

## 🧠 Methodology

The project follows a machine-learning pipeline consisting of data preparation, feature extraction, model training, and evaluation.

### Step 1: Data Collection

The project uses datasets associated with the three modalities:

- **Dreaddit:** Text-based stress-related data.
- **AffectNet:** Facial expression and emotion-related data.
- **RAVDESS:** Audio and speech-related data.

### Step 2: Data Preprocessing

The input data is prepared according to its modality.

- Text: Clean and transform textual input for NLP processing.
- Facial: Process facial images and prepare expression-related features.
- Voice: Process audio signals and extract acoustic representations.

### Step 3: Feature Extraction

Different feature extraction techniques are used for each input type.

| Modality | Feature Extraction |
|---|---|
| Text | TF-IDF |
| Facial | Facial and emotion-related features |
| Voice | MFCC features |

### Step 4: Model Training

Logistic Regression is used as the classification algorithm for the three analysis modules.

Each modality is trained and evaluated separately using its corresponding features.

### Step 5: Model Evaluation

The dataset is divided using an 80:20 train-test split, and 10-fold cross-validation is used to examine model performance.

The evaluation metrics include:

- Accuracy
- F1-score
- Cross-validation accuracy

### Step 6: Performance Comparison

The results of the text, facial, and voice models are compared to understand their individual performance and identify opportunities for improvement.

## 📊 Results

The following are the reported evaluation results from the project.

| Modality | Test Accuracy | F1-Score | 10-Fold CV |
|---|---:|---:|---:|
| Text | 87.4% | 87.6% | 87.8% |
| Facial | 81.2% | 81.2% | 81.6% |
| Voice | 84.6% | 84.5% | 84.7% |

### Results Interpretation

- The text-based model achieved 87.4% test accuracy.
- The facial analysis model achieved 81.2% test accuracy.
- The voice-based model achieved 84.6% test accuracy.
- The reported results show differences in classification performance across the three modalities.

These results describe performance on the project's evaluation setup. They do not establish clinical validity or guarantee performance on new users, devices, or real-world situations.

## ⚙️ Installation

### Prerequisites

Make sure the following are available:

- Python 3.9 or a compatible version supported by the project's dependencies.
- Git.
- pip.
- A code editor such as Visual Studio Code or PyCharm.

### 1. Clone the Repository

```bash
git clone https://github.com/Mounika-1903/stress-detection-system.git
```

### 2. Navigate to the Project Directory

```bash
cd stress-detection-system
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If a dependency file is not included, install the packages required by the relevant modules. For example:

```bash
pip install numpy pandas scikit-learn opencv-python librosa
```

Additional dependencies may be required depending on the project's implementation.

## 🚀 Usage

1. Clone the repository and install the required dependencies.
2. Review the project folders and identify the entry-point scripts or notebooks for each modality.
3. Ensure the required datasets are available in the locations expected by the code.
4. Run the relevant text, facial, or voice analysis module.
5. Follow the input instructions provided by the implementation.
6. Review the model output and evaluation results.

### Running the Project

The exact commands depend on the scripts included in the repository.

For example, if the project contains a Python entry-point file:

```bash
python main.py
```

If the project is organized into separate modules, run the appropriate module or notebook instead.

**Note:** Replace the example command with the actual entry point included in the repository. Dataset paths and dependency versions may also need to be configured before execution.

## 📁 Project Structure

The following is a suggested organization for the project. Adjust it to match the actual files in the repository.

```text
stress-detection-system/
│
├── datasets/
│   ├── text/
│   ├── facial/
│   └── voice/
│
├── notebooks/
│   ├── text_analysis.ipynb
│   ├── facial_analysis.ipynb
│   └── voice_analysis.ipynb
│
├── src/
│   ├── text_detection.py
│   ├── facial_detection.py
│   └── voice_detection.py
│
├── models/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

This structure is illustrative, not a verified inventory of the current repository.

## ⚠️ Limitations

- **Dataset limitations:** Model performance depends on dataset quality, diversity, labeling, and representativeness.
- **Generalization:** Results from the evaluation datasets may not generalize to different populations, environments, languages, or recording conditions.
- **Facial analysis limitations:** Lighting, camera quality, pose, occlusion, and individual differences can affect extracted features.
- **Voice analysis limitations:** Background noise, microphones, accents, language, and speaking styles can influence audio features.
- **Text analysis limitations:** Sarcasm, context, language differences, and ambiguous wording can make textual classification unreliable.
- **Model limitations:** Logistic Regression may not capture complex relationships in multimodal data.
- **Evaluation limitations:** Accuracy and F1-score alone do not establish reliability, fairness, or suitability for real-world decision-making.
- **Not a clinical tool:** The system is not validated for diagnosis, treatment, or clinical assessment of stress.

## 🔮 Future Enhancements

- Develop a unified multimodal model that combines text, facial, and voice features.
- Explore additional machine-learning and deep-learning algorithms.
- Improve dataset diversity and evaluate performance across different groups and environments.
- Add more comprehensive evaluation metrics, including precision, recall, confusion matrices, and modality-specific error analysis.
- Investigate explainability techniques to make model predictions easier to interpret.
- Improve preprocessing, feature selection, and hyperparameter tuning.
- Develop a user-friendly interface for submitting inputs and viewing results.
- Add secure data handling and privacy-conscious processing.
- Conduct broader validation before considering any real-world application.

## 👩‍💻 Project Team

This is a group project developed as part of an academic machine-learning project.

- **Team Leader:** Ummadisetti Mounika
- **Team Members:** Hema Haindavi, Akshay, Ganesh
- **Project Guide:** Dr. P. Madhuri

## 🎓 Learning Outcomes

Through this project, we explored:

- Applying machine learning to a practical classification problem.
- Text preprocessing and TF-IDF feature extraction.
- Computer vision and facial expression-related analysis.
- Audio processing and MFCC feature extraction.
- Training and evaluating Logistic Regression models.
- Comparing model performance using standard evaluation metrics.
- Understanding the limitations of machine-learning predictions.

## 📜 Disclaimer

This project is intended only for academic learning, experimentation, and research. It does not provide a medical diagnosis, psychological assessment, or professional advice. Predictions may be inaccurate and should not be used to make consequential decisions about a person's health, employment, education, or well-being.


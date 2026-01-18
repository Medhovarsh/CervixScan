# CervixScan
### AI-Powered Cervical Cancer Screening System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deployment](https://img.shields.io/badge/Deployment-Flask-lightgrey)
![Status](https://img.shields.io/badge/Status-Research_Prototype-blue)

## About The Project
**CervixScan** is a deep learning-based diagnostic tool developed to assist clinicians at **Amrita Institute of Medical Sciences (AIMS), Kochi**.

Cervical cancer screening often suffers from delays and human error. [cite_start]We built this system to serve as an automated "second reader," improving early diagnosis rates by **20%** through advanced preprocessing and model optimization[cite: 3]. [cite_start]By deploying the solution via a cloud-based **Flask** application, we enabled clinicians to access screening results **95% faster** than traditional manual workflows[cite: 3].

## Key Features
* **High-Speed Screening:** Reduced screening time by 95% via optimized cloud deployment.
* **Enhanced Accuracy:** Improved early diagnosis by 20% using a custom DeepLabV3+ architecture.
* **Advanced Preprocessing:** Implemented Non-Local Means (NLM) denoising to remove artifacts and improve image clarity for the model.
* **User-Friendly Interface:** Deployed as a web application using Flask for easy access by medical personnel.

## How We Built It
### 1. The Architecture (DeepLabV3+ & ResNet50)
[cite_start]We leveraged the **DeepLabV3+** framework with a **ResNet50** backbone[cite: 3]. This combination allowed us to capture multi-scale contextual features—essential for distinguishing between normal cells and subtle pre-cancerous lesions (Dysplasia).

### 2. Preprocessing
To ensure consistent model performance, we utilized **scikit-learn** and **OpenCV** to build a robust preprocessing pipeline, applying NLM filtering to clean raw slide images before analysis.

### 3. Deployment
The model was wrapped in a **Flask** API and deployed to the cloud, allowing for real-time inference on uploaded slide images.

## Performance
* **Accuracy:** ~98% on the test set.
* **Impact:** 20% improvement in early diagnosis detection.
* **Speed:** 95% reduction in screening time compared to manual review.

## Tech Stack
* **Core:** Python 3, TensorFlow, Keras
* **Model:** DeepLabV3+, ResNet50
* **Data Processing:** NumPy, Pandas, scikit-learn
* **Deployment:** Flask, Cloud Platform
* **Visualization:** Matplotlib

## The Team
* **Medhovarsh Bayyapureddi**
* **Supreeth Amartaluru**
* **Sri Harshitha Anantatmula**
* **Karthik Aduri**

**Project Guide:** Mrs. Vinitha Panicker J (Dept. of CSE, Amrita School of Engineering)

## How to Run locally

### Prerequisites
* Python 3.8+
* TensorFlow
* Flask

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Medhovarsh/CervixScan.git](https://github.com/Medhovarsh/CervixScan.git)
    cd CervixScan
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Run the Web App
Start the Flask server:
```bash
python app.py

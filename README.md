# 🇺🇸 US Visa Approval Prediction — End-to-End MLOps Project

> **An end-to-end Machine Learning pipeline that predicts whether a US visa application will be _Certified_ or _Denied_, built with a modular, production-grade MLOps architecture.**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS_S3-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Architecture & Pipeline Workflow](#-architecture--pipeline-workflow)
  - [Data Ingestion](#1--data-ingestion)
  - [Data Validation](#2--data-validation)
  - [Data Transformation](#3--data-transformation)
  - [Model Trainer](#4--model-trainer)
  - [Model Evaluation](#5--model-evaluation)
  - [Model Pusher](#6--model-pusher)
- [Prediction Pipeline](#-prediction-pipeline)
- [How to Run](#-how-to-run)
- [Environment Variables](#-environment-variables)
- [API Endpoints](#-api-endpoints)
- [Project Highlights](#-project-highlights)
- [License](#-license)
- [Author](#-author)

---

## 🔍 Overview

This project implements a **complete MLOps pipeline** for predicting the approval status of US work visa applications. It covers every stage of the ML lifecycle — from data ingestion from MongoDB, through validation, transformation, model training with hyperparameter tuning, evaluation against production models, and deployment to AWS S3. A **FastAPI** web application serves as the user-facing prediction interface.

The entire codebase follows **modular, production-grade software engineering principles** with:
- Custom exception handling & structured logging
- Dataclass-driven configuration management
- Artifact-based pipeline stage tracking
- Cloud-native model registry (AWS S3)
- Containerization support (Docker)

---

## 🎯 Problem Statement

The **Office of Foreign Labor Certification (OFLC)** processes job certification applications for employers seeking to bring foreign workers into the United States. The goal of this project is to build a classification model that can predict whether a visa application will be **Certified** or **Denied** based on the given features, helping streamline the review process.

**Business Impact:** Automating visa status prediction helps reduce manual review time, flag high-risk applications, and improve operational efficiency for immigration processing.

---

## 📊 Dataset

The project uses the **EasyVisa** dataset containing US visa application records with the following features:

| Feature | Type | Description |
|---|---|---|
| `case_id` | Categorical | Unique identifier for each visa application |
| `continent` | Categorical | Continent of the employee's origin (Asia, Europe, Africa, etc.) |
| `education_of_employee` | Categorical | Education level (High School, Bachelor's, Master's, Doctorate) |
| `has_job_experience` | Categorical | Whether the employee has prior job experience (Y/N) |
| `requires_job_training` | Categorical | Whether the job requires training (Y/N) |
| `no_of_employees` | Numerical | Number of employees in the employer's company |
| `yr_of_estab` | Numerical | Year the employer's company was established |
| `region_of_employment` | Categorical | US region of employment (West, Northeast, South, Midwest, Island) |
| `prevailing_wage` | Numerical | Average wage for the job in the area of employment |
| `unit_of_wage` | Categorical | Unit of prevailing wage (Hour, Week, Month, Year) |
| `full_time_position` | Categorical | Whether the position is full-time (Y/N) |
| `case_status` | Categorical | **Target variable** — Certified or Denied |

> **Engineered Feature:** `company_age` is derived at runtime as `current_year - yr_of_estab`.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.8+ |
| **Web Framework** | FastAPI, Uvicorn, Jinja2 |
| **Database** | MongoDB Atlas |
| **Cloud Storage** | AWS S3 (model registry) |
| **ML Libraries** | scikit-learn, XGBoost, CatBoost, Evidently |
| **Data Handling** | Pandas, NumPy, imbalanced-learn (SMOTEENN) |
| **Model Selection** | neuro_mf (ModelFactory with GridSearchCV) |
| **Serialization** | dill, pickle |
| **Config Management** | PyYAML, Python dataclasses |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Containerization** | Docker |

---

## 📁 Folder Structure

<img width="4450" height="2511" alt="Image" src="https://github.com/user-attachments/assets/64bade1e-923e-4152-bc56-956a2e1cd298" />

**Text Reference of the Project Structure:**

```
US-VISA-approval-/
│
├── app.py                        # FastAPI application (entry point)
├── demo.py                       # Script to trigger training pipeline
├── template.py                   # Project scaffolding script
├── setup.py                      # Package setup (pip install -e .)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker containerization config
├── .dockerignore                 # Docker ignore rules
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
│
├── config/
│   ├── model.yaml                # Model hyperparameter search space
│   └── schema.yaml               # Dataset schema & column definitions
│
├── notebooks/
│   ├── 1_EDA_US_visa.ipynb       # Exploratory Data Analysis
│   ├── 2_Feature_Engineering_and_Model_Training.ipynb
│   ├── mongodb_demo.ipynb        # MongoDB connection demo
│   └── EasyVisa.csv              # Raw dataset
│
├── static/css/
│   └── style.css                 # Frontend custom styles
│
├── templates/
│   └── usvisa.html               # Jinja2 HTML prediction form
│
├── logs/                         # Auto-generated log files
│
└── us_visa/                      # Core ML package
    ├── __init__.py
    │
    ├── components/               # Pipeline stage implementations
    │   ├── data_ingestion.py
    │   ├── data_validation.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    │   ├── model_evaluation.py
    │   └── model_pusher.py
    │
    ├── configuration/            # External service connections
    │   ├── mongo_db_connection.py
    │   └── aws_connection.py
    │
    ├── constants/                # Project-wide constants
    │   └── __init__.py
    │
    ├── entity/                   # Data classes for configs & artifacts
    │   ├── config_entity.py
    │   ├── artifact_entity.py
    │   ├── estimator.py
    │   └── s3_estimator.py
    │
    ├── cloud_storage/            # AWS S3 operations
    │   └── aws_storage.py
    │
    ├── data_access/              # Database access layer
    │   └── usvisa_data.py
    │
    ├── pipline/                  # Pipeline orchestrators
    │   ├── training_pipeline.py
    │   └── prediction_pipeline.py
    │
    ├── utils/                    # Utility functions
    │   └── main_utils.py
    │
    ├── exception/                # Custom exception handling
    │   └── __init__.py
    │
    └── logger/                   # Structured logging
        └── __init__.py
```

---

## 🏗️ Architecture & Pipeline Workflow

The **Training Pipeline** runs through 6 sequential stages. Each stage takes the artifact from the previous stage as input, processes it, and produces a new artifact — creating a clean, traceable lineage.

```
MongoDB → Data Ingestion → Data Validation → Data Transformation → Model Trainer → Model Evaluation → Model Pusher → AWS S3
```

---

### 1. 📥 Data Ingestion

<img width="2871" height="1986" alt="Image" src="https://github.com/user-attachments/assets/9ee0ab09-424e-4999-8334-b1f46061c157" />

**What it does:**
- Connects to **MongoDB Atlas** and exports the `visa_data` collection as a Pandas DataFrame
- Saves the raw data to the **feature store** as `usvisa.csv`
- Performs an **80/20 train-test split** using `sklearn.model_selection.train_test_split`
- Saves the resulting `train.csv` and `test.csv` to the ingested data directory

**Key Configuration:**
| Parameter | Value |
|---|---|
| Database | `US_VISA` |
| Collection | `visa_data` |
| Split Ratio | 0.2 (20% test) |
| Output Format | CSV |

**Artifacts Produced:**
- `artifact/<timestamp>/data_ingestion/feature_store/usvisa.csv`
- `artifact/<timestamp>/data_ingestion/ingested/train.csv`
- `artifact/<timestamp>/data_ingestion/ingested/test.csv`

**Source:** `us_visa/components/data_ingestion.py`

---

### 2. ✅ Data Validation

<img width="4882" height="4359" alt="Image" src="https://github.com/user-attachments/assets/bb8914e0-30f7-4c23-ab6e-4fa13fa4edf5" />

**What it does:**
- **Column Count Validation** — Checks that the number of columns in train/test data matches the schema definition (`config/schema.yaml`)
- **Column Existence Check** — Verifies all expected numerical and categorical columns are present
- **Data Drift Detection** — Uses **Evidently's `DataDriftProfileSection`** to statistically compare the training and testing distributions
- Generates a drift report saved as `report.yaml`

**Validation Checks:**
1. ✅ Number of columns matches schema
2. ✅ All numerical columns present (`no_of_employees`, `prevailing_wage`, `yr_of_estab`)
3. ✅ All categorical columns present (8 columns)
4. ✅ No significant data drift between train and test sets

**Artifacts Produced:**
- `artifact/<timestamp>/data_validation/drift_report/report.yaml`
- Validation status (pass/fail) + error message

**Source:** `us_visa/components/data_validation.py`

---

### 3. 🔄 Data Transformation

<img width="5278" height="4464" alt="Image" src="https://github.com/user-attachments/assets/da473c37-6e44-4456-9aa3-803670d108f4" />

**What it does:**
- **Feature Engineering:** Creates `company_age` feature from `current_year - yr_of_estab`
- **Column Dropping:** Removes `case_id` and `yr_of_estab` (no longer needed)
- **Target Encoding:** Maps `case_status` → `Certified = 0`, `Denied = 1`
- **Preprocessing Pipeline** using `sklearn.compose.ColumnTransformer`:

| Transformer | Columns | Description |
|---|---|---|
| `OneHotEncoder` | `continent`, `unit_of_wage`, `region_of_employment` | Encodes nominal categories |
| `OrdinalEncoder` | `has_job_experience`, `requires_job_training`, `full_time_position`, `education_of_employee` | Encodes ordinal categories |
| `PowerTransformer` (Yeo-Johnson) | `no_of_employees`, `company_age` | Normalizes skewed distributions |
| `StandardScaler` | `no_of_employees`, `prevailing_wage`, `company_age` | Scales to zero mean, unit variance |

- **Class Imbalance Handling:** Applies **SMOTEENN** (SMOTE + Edited Nearest Neighbors) on both train and test sets to handle the imbalanced target distribution
- Saves transformed arrays as `.npy` files and the preprocessor object as `preprocessing.pkl`

**Artifacts Produced:**
- `artifact/<timestamp>/data_transformation/transformed/train.npy`
- `artifact/<timestamp>/data_transformation/transformed/test.npy`
- `artifact/<timestamp>/data_transformation/transformed_object/preprocessing.pkl`

**Source:** `us_visa/components/data_transformation.py`

---

### 4. 🤖 Model Trainer

<!-- ==================== DIAGRAM PLACEHOLDER: MODEL TRAINER ==================== -->

> ### 📌 **Diagram — MODEL TRAINER**
>
> _Insert your Model Trainer pipeline diagram here._
>
> ⬇️ **PLACEHOLDER** ⬇️

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│             [ MODEL TRAINER DIAGRAM ]                   │
│                                                         │
│            Insert your diagram/image here               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

<!-- ==================== END PLACEHOLDER ==================== -->

**What it does:**
- Uses **`neuro_mf.ModelFactory`** to automate model selection and hyperparameter tuning
- Reads model search space from `config/model.yaml`
- Performs **GridSearchCV (cv=3)** across all configured models
- Evaluates the best model on the test set using multiple metrics

**Models in Search Space:**

| Model | Key Hyperparameters |
|---|---|
| **KNeighborsClassifier** | `algorithm`: [auto, ball_tree, kd_tree, brute], `weights`: [uniform, distance], `n_neighbors`: [3, 5, 9] |
| **RandomForestClassifier** | `max_depth`: [10, 15, 20], `max_features`: [sqrt, log2], `n_estimators`: [3, 5, 9] |

**Evaluation Metrics:**
- ✅ Accuracy Score
- ✅ F1 Score
- ✅ Precision Score
- ✅ Recall Score

**Minimum Threshold:** The model must achieve at least **60% accuracy** (`MODEL_TRAINER_EXPECTED_SCORE = 0.6`) to be accepted.

**Artifacts Produced:**
- `artifact/<timestamp>/model_trainer/trained_model/model.pkl` — Serialized `USvisaModel` (preprocessor + trained model bundled together)

**Source:** `us_visa/components/model_trainer.py`

---

### 5. 📊 Model Evaluation

<!-- ==================== DIAGRAM PLACEHOLDER: MODEL EVALUATION ==================== -->

> ### 📌 **Diagram — MODEL EVALUATION**
>
> _Insert your Model Evaluation pipeline diagram here._
>
> ⬇️ **PLACEHOLDER** ⬇️

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│           [ MODEL EVALUATION DIAGRAM ]                  │
│                                                         │
│            Insert your diagram/image here               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

<!-- ==================== END PLACEHOLDER ==================== -->

**What it does:**
- Fetches the **currently deployed model** from AWS S3 bucket (`usvisa-model2024`)
- Compares the newly trained model's **F1 score** against the production model's F1 score
- The new model is **accepted only if** its F1 score exceeds the production model's score
- If no production model exists yet (first deployment), the new model is automatically accepted

**Decision Logic:**
```
IF trained_model_f1 > production_model_f1:
    → Model ACCEPTED (proceed to Model Pusher)
ELSE:
    → Model REJECTED (pipeline stops here)
```

**Change Threshold:** `MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02` (2% minimum improvement)

**Artifacts Produced:**
- `ModelEvaluationArtifact` containing:
  - `is_model_accepted` (bool)
  - `changed_accuracy` (float — difference between new and old)
  - `s3_model_path` and `trained_model_path`

**Source:** `us_visa/components/model_evaluation.py`

---

### 6. 🚀 Model Pusher

<!-- ==================== DIAGRAM PLACEHOLDER: MODEL PUSHER ==================== -->

> ### 📌 **Diagram — MODEL PUSHER**
>
> _Insert your Model Pusher pipeline diagram here._
>
> ⬇️ **PLACEHOLDER** ⬇️

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│             [ MODEL PUSHER DIAGRAM ]                    │
│                                                         │
│            Insert your diagram/image here               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

<!-- ==================== END PLACEHOLDER ==================== -->

**What it does:**
- Takes the accepted model from the evaluation stage
- Uploads the serialized `model.pkl` file to the **AWS S3 model registry**
- The uploaded model becomes the new **production model** for the prediction pipeline

**Deployment Details:**
| Parameter | Value |
|---|---|
| S3 Bucket | `usvisa-model2024` |
| S3 Key | `model.pkl` |
| Region | `us-east-1` |

**Artifacts Produced:**
- `ModelPusherArtifact` with the S3 bucket name and model path

**Source:** `us_visa/components/model_pusher.py`

---

## 🔮 Prediction Pipeline

The **Prediction Pipeline** serves real-time predictions through the FastAPI web application:

1. User fills out the **US Visa Prediction Form** with applicant details
2. The form data is parsed into a `USvisaData` object and converted to a DataFrame
3. The `USvisaClassifier` loads the latest production model from **AWS S3**
4. The model (preprocessor + classifier) transforms the input and returns a prediction
5. The result (`Visa-approved` or `Visa Not-Approved`) is displayed on the web page

**Input Features for Prediction:**
- Continent, Education, Job Experience, Job Training Requirement
- Number of Employees, Region of Employment, Prevailing Wage
- Unit of Wage, Full-time Position, Company Age

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (with the EasyVisa dataset loaded)
- AWS account with S3 access
- Conda (recommended)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/atulbhaskar1034/US-VISA-approval-.git
cd US-VISA-approval-
```

### Step 2 — Create & Activate Environment

```bash
conda create -n visa python=3.8 -y
conda activate visa
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set Environment Variables

```bash
# MongoDB
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"

# AWS
export AWS_ACCESS_KEY_ID="<your-aws-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-aws-secret-key>"
```

### Step 5 — Run the Training Pipeline

```bash
python demo.py
```

### Step 6 — Start the Web Application

```bash
python app.py
```

Visit **http://localhost:8080** in your browser to access the prediction form.

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `MONGODB_URL` | MongoDB Atlas connection string | ✅ Yes |
| `AWS_ACCESS_KEY_ID` | AWS IAM access key for S3 | ✅ Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key for S3 | ✅ Yes |

---

## 🌐 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Renders the US Visa prediction form |
| `POST` | `/` | Accepts form data and returns visa prediction |
| `GET` | `/train` | Triggers the full training pipeline |

---

## ✨ Project Highlights

- **🏗️ Modular Architecture** — Every pipeline stage is a self-contained component with its own config, artifact, and exception handling
- **📝 Structured Logging** — Timestamped log files under `logs/` for debugging and audit trails
- **🎯 Automated Model Selection** — `neuro_mf.ModelFactory` with GridSearchCV eliminates manual model comparison
- **⚖️ Class Imbalance Handling** — SMOTEENN combines over-sampling (SMOTE) and under-sampling (ENN) for better performance on imbalanced data
- **📊 Data Drift Detection** — Evidently profiles ensure training and testing distributions stay aligned
- **☁️ Cloud-Native Model Registry** — AWS S3 as the single source of truth for production models
- **🔄 CI/CD Ready** — Model evaluation gates deployment: only improved models get pushed to production
- **🐳 Docker Support** — Dockerfile included for containerized deployment
- **🌐 Web Interface** — Clean Bootstrap-based prediction form powered by FastAPI + Jinja2

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Atul Bhaskar**  
📧 magnetar1034@gmail.com  
🔗 [GitHub](https://github.com/atulbhaskar1034)

---

<p align="center">
  <i>If you found this project helpful, consider giving it a ⭐ on GitHub!</i>
</p>

# Amazon Music Reviews Sentiment Analysis
 TMS ML Assesment - 1

## Overview
This project involves sentiment analysis on Amazon Music reviews using machine learning techniques. The goal is to predict sentiment labels (positive, neutral, negative) based on customer reviews.

## Dataset
The dataset used for this project is available on Kaggle: [Amazon Music Reviews Dataset](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews).

### Data Exploration and Cleaning
- Loaded and inspected the dataset.
- Removed rows with missing review text.
- Dropped unnecessary columns (`reviewerID`, `reviewerName`, `unixReviewTime`, `reviewTime`).

### Feature Engineering
- Combined `reviewText` and `summary` columns into a single `review` column.
- Created a `sentiment` column based on the `overall` ratings:
  - Positive for ratings > 3
  - Neutral for ratings = 3
  - Negative for ratings < 3
- Calculated the `helpful_rate` from the `helpful` column.

### Text Preprocessing
- Cleaned text data by removing special characters, URLs, and punctuation.
- Converted text to lowercase and removed stopwords.
- Applied stemming to reduce words to their root form using PorterStemmer.

### Feature Extraction
- Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.

### Handling Imbalanced Data
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by oversampling the minority classes.

## Modeling
- Trained a Logistic Regression model using the TF-IDF transformed data.
- Achieved a train accuracy of 92.47% and test accuracy of 92.25%.
- Utilized PyCaret for model comparison and setup, identifying Logistic Regression as the best performing model.

### PyCaret Usage
- Leveraged PyCaret for automating preprocessing, setup, and model comparison.
- Explored various classification models including Logistic Regression, XGBoost, Random Forest, etc.

## Model Performance
- **Train Accuracy:** 92.47%
- **Test Accuracy:** 92.25%

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- textblob
- scikit-learn
- imbalanced-learn
- pycaret

## Usage
1. Clone the repository or download the dataset from Kaggle.
2. Install the required dependencies (`pip install -r requirements.txt`).
3. Run the Jupyter Notebook to execute the analysis and train the model.

## Contributors
- **ArvindJawahar**
  AI Researcher
  Thirumoolar Software

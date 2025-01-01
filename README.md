# Iris Dataset Analysis

This project performs analysis and classification on the Iris dataset. The dataset is commonly used for machine learning tasks and contains data about three different species of Iris flowers, including their petal and sepal length and width.

## Libraries Used

The following libraries are used for data manipulation, visualization, and machine learning:

- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical operations.
- **Matplotlib**: Data visualization.
- **Seaborn**: Statistical data visualization.
- **Scikit-learn**: Machine learning algorithms and tools.

## Algorithms Implemented

The following machine learning algorithms are applied to classify the Iris flower species:

- **Logistic Regression**: A statistical method for binary classification that can be extended to multi-class classification.
- **K-Nearest Neighbors (KNN)**: A simple, non-parametric algorithm that classifies based on the majority class among the k-nearest neighbors.
- **Decision Tree**: A tree-based model that splits data based on feature values to classify.

## Model Comparison

After evaluating multiple algorithms, the **Decision Tree** model was found to be the best for this classification task. 

### Best Model: Decision Tree

- **Model Accuracy**: 97.78%

The Decision Tree classifier performed the best due to its ability to handle both continuous and categorical data, providing a higher accuracy than other models.

## Dataset Information

The dataset used in this analysis is the **Iris Dataset**, which contains data about three different species of Iris flowers: **Setosa**, **Versicolor**, and **Virginica**. It includes the following features:

- **Sepal Length**: Length of the sepal in centimeters.
- **Sepal Width**: Width of the sepal in centimeters.
- **Petal Length**: Length of the petal in centimeters.
- **Petal Width**: Width of the petal in centimeters.
- **Species**: The target variable representing the flower species.

### Dataset Source:
The dataset is publicly available on [Kaggle](https://www.kaggle.com/).

## Steps Involved in the Analysis

1. **Data Loading**: The Iris dataset is loaded into a pandas DataFrame for further analysis.
2. **Data Preprocessing**:
   - Handle missing data (if any).
   - Split the dataset into features (X) and target (y).
   - Perform data scaling for algorithms that require it (like KNN).
3. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of features.
   - Check correlations between features.
   - Explore the class distribution of different species using **Seaborn**.
4. **Model Training and Evaluation**:
   - Train each of the three algorithms: Logistic Regression, KNN, and Decision Tree.
   - Evaluate model accuracy using **cross-validation**.
   - Compare model performance and select the best model (Decision Tree).
5. **Model Performance**:
   - The Decision Tree classifier achieved an accuracy of **97.78%** on the test set, making it the best model for this task.
6. **Visualization**:
   - Visualize feature importance and decision boundaries for the Decision Tree model.

## Installation

To run the project, you’ll need to install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Analysis
To run the analysis, execute the Python script in your preferred Python environment.

```bash
python iris_analysis.py
```

This will:
1. Load the Iris dataset.
2. Perform data preprocessing and EDA.
3. Train and evaluate the models.
4. Output the best performing model and its accuracy.

## Results
- Logistic Regression Accuracy: 95.56%
- K-Nearest Neighbors Accuracy: 97.77%
- Decision Tree Accuracy: 97.78%

The **Decision Tree** classifier achieved the highest accuracy and was selected as the best model for the task.

## Conclusion
This project demonstrates the power of different machine learning algorithms applied to a classic dataset, the Iris dataset. The Decision Tree model provides an accurate solution for classifying Iris flower species and is highly interpretable due to its tree-like structure.

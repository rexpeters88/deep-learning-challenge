# **Neural Network Model for Alphabet Soup**

## **Overview of the Analysis**  
The purpose of this analysis was to create a deep learning model to assist Alphabet Soup, a nonprofit organization, in identifying funding applicants with the highest likelihood of success. By predicting whether an applicant will effectively utilize funding, the model aims to support Alphabet Soup in making strategic, data-driven funding decisions.

## **Results**

**Data Preprocessing**

- **Target Variable:**  
The target variable for the model is IS_SUCCESSFUL, which indicates whether an organization that received funding was ultimately successful.
- **Features:**  
The features used for the model were derived from the following columns:

  - **APPLICATION_TYPE**
  - **AFFILIATION**
  - **CLASSIFICATION**
  - **USE_CASE**
  - **ORGANIZATION**
  - **STATUS**
  - **INCOME_AMT**
  - **SPECIAL_CONSIDERATIONS**
  - **ASK_AMT**
  - **NAME_CATEGORY**  
 These columns were processed using np.get_dummies() to convert categorical data into numerical values suitable for training the model.

- **Removing Variables:**  
  The EIN column was removed from the dataset as it is an identifier and does not provide predictive value for the model. While the NAME column was not removed, it was clustered into common categories based on frequency and patterns, allowing it to provide meaningful insights to the model. Similarly, the ASK_AMT column was grouped into bins to reduce outliers and improve generalization.

**Compiling, Training, and Evaluating the Model**

- **Neurons, Layers, and Activation Functions:**  
The updated model architecture is designed to balance capacity and regularization, as follows:

Input layer:
The input layer contains 64 neurons, matching the number of features in the dataset.
Hidden layers:
First hidden layer: 64 neurons with the ReLU activation function.
Second hidden layer: 32 neurons with the ReLU activation function.
Third hidden layer: 16 neurons with the ReLU activation function to reduce complexity and enforce abstraction.
Dropout:
A dropout layer with a rate of 0.1 is added after the third hidden layer to prevent overfitting by randomly disabling 10% of neurons during training.
Output layer:
A single neuron with a sigmoid activation function produces a probability score for binary classification (success or failure).

The ReLU activation function is used in all hidden layers to introduce non-linearity, enabling the model to learn complex relationships in the data. The sigmoid function in the output layer ensures the output is a probability between 0 and 1.

Compilation:
The model is compiled using:

Optimizer: Adam, with a reduced learning rate of 0.0005, which allows the model to converge slowly and carefully.
Loss Function: Binary cross-entropy, suitable for binary classification problems.
Metrics: Accuracy, to evaluate model performance.

- **Achieving Target Performance:**  
  The model was not able to achieve the target accuracy of 75%. Despite efforts to optimize the model, it consistently achieved a lower accuracy on the test dataset, with a final accuracy of approximately 74%.

- **Steps Taken to Increase Model Performance:**  
To improve the model's accuracy, the following strategies were implemented:

Clustering the NAME column: Instead of dropping the NAME column, organization names were grouped into categories based on their frequency and keywords. This allowed the model to utilize patterns in the names to enhance predictions.
Binning the ASK_AMT column: The ASK_AMT column, which had a wide range of values, was grouped into bins to reduce the impact of outliers and help the model generalize better.
Adjusting application types: Application types with low frequency were grouped into an "Other" category to prevent rare categories from negatively affecting model performance.
Regularization and Dropout: Regularization techniques, such as dropout layers, were added to mitigate overfitting.


## **Summary**

The deep learning model provided a baseline for predicting the success of funding applicants for Alphabet Soup. However, it fell short of the target accuracy of 75%, achieving approximately 74% accuracy on the test dataset. The clustering of features such as NAME and ASK_AMT, as well as efforts to reduce overfitting, were implemented but did not fully bridge the gap.

## **Recommendation for a Different Model:**

To solve this classification problem, a simpler and interpretable model such as logistic regression could be a viable alternative. Logistic regression is effective for binary classification tasks and would allow Alphabet Soup to understand the impact of each feature on the likelihood of success through the model's coefficients. Additionally, an ensemble method like Random Forest could be explored to handle feature interactions and provide better generalization.

Further fine-tuning and exploration of alternative models are recommended to achieve the desired performance.

# Face Obfuscation Effectiveness Model

## Project Overview
This model is not designed for facial recognition but for evaluating the effectiveness of face obfuscation methods. It aims to empower individuals to test their privacy tools and improve digital anonymity in high-risk situations. By analyzing how well obfuscation techniques prevent identity detection, the model supports activists, journalists, and vulnerable individuals in safeguarding their privacy. This project prioritizes ethical considerations and is not intended for surveillance or law enforcement applications.


## Dataset Used 
The project utilized two datasets:
- SoF (Specs of Faces) - A dataset containing various face obfuscation techniques applied to images, helping evaluate how well different methods obscure identities. [Dataset](https://sites.google.com/view/sof-dataset?pli=1)

- FFHQ (Flickr-Faces-HQ) - A high-quality dataset of diverse, high-resolution face images. [Dataset](https://datasets.activeloop.ai/docs/ml/datasets/ffhq-dataset/)



## Findings

| **Training instance** | **Optimizer used** | **Regularizer used** | **Epochs** | **Early Stopping** | **Number of Layers** | **Learning Rate** | **Accuracy** | **F1 Score** | **Recall** | **Precision** | **ROC-AUC** |
|-----------------------|--------------------|----------------------|------------|--------------------|----------------------|-------------------|--------------|--------------|------------|---------------|-------------|
| Instance 1            | none               | none                 | 20         | yes                | 10                   | none              | 91.76%       | 0.9103       | 0.8353     | 1.0           | 0.9176      |
| Instance 2            | SGD                | L1+L2                | 15         | yes                | 14                   | 0.001             | 99.61%       | 0.9961       | 0.9922     | 1.0           | 0.9999      |
| Instance 3            | Adam               | L2                   | 12         | yes                | 14                   | 0.001             | 100%         | 1.00         | 1.00       | 1.00          | 1.00        |
| Instance 4            | Nadam              | L1                   | 10         | yes                | 14                   | 0.001             | 100%         | 1.00         | 1.00       | 1.00          | 1.00        |
| Instance 5 (LogReg)   | SAGA               | L2                   | N/A        | N/A                | N/A                  | N/A               | 100%         | 1.00         | 1.00       | 1.00          | 1.00        |
|                       |                    |                      |            |                    |                      |                   |              |              |            |               |             |

## Comparison of Model Performance
### Best Performing Neural Network Configuration
- **Regularization**: L1 + L2 (Elastic Net) 
- **Dropout Rate**: 0.5 (to prevent overfitting)
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Key Takeaways**:
  - Strong generalization with reduced overfitting
  -  Avoided 100% accuracy, which suggests better real-world performance
  - Efficient training despite being computationally intensive



### Neural Network vs. Classic Machine Learning Algorithm
I compared the best performing neural network model to a Logistic Regression model to understand their strengths and weaknesses:

#### Neural Network:
- Able to capture complex patterns and relationships in the dataset.
- Works well even when data is obscured or noisy (which is important in real-world applications).
- However, higher computational cost and more difficult to interpret compared to classical methods.


#### Machine Learning (Logistic Regression):
- Works well when data is clean and simple, but struggles with more complex patterns


#### Key Hyperparameters for Logistic Regression:
- Regularization: L2 (Ridge Regression)
- Solver: SAGA (best for handling L1/L2 penalties and large-scale datasets)
- Other Optimizations: Standardization of input data, parameter tuning with RandomizedSearchCV


### Why instance 2 was the best neural network model
Although Instances 3 and 4 achieved 100% accuracy, this was a red flag. Overfitting to the training data can lead to poor generalization. Instance 2 (SGD + L1/L2 regularization) struck the perfect balance by:

- Achieving 99.61% accuracy, which is high but not suspiciously perfect
- Using strong regularization (L1+L2) to prevent overfitting
- Efficient optimization with SGD, which is great for handling large datasets
- Early stopping and dropout (0.5) to ensure better generalization


Instead of chasing a perfect 100%, we prioritized a model that performs well while remaining robust and generalizable


### Why did some models still reach 100% accuracy
#### (Possible reasons for overfitting despite extensive training)

1. **Dataset Size and Complexity**
- The models trained on a subset of 1700 images from each dataset. The number 1700 was reached after much experimentation with trying to find a size that is not too small but also not too large to be difficult to compute.
- If the dataset is too small or lacks diversity, the model might memorize patterns instead of learning generalizable features.

2. **Feature Leakage or Bias**
- Data leakage was checked for to ensure that there is no overlap in the validation and test sets.
- However, if some features are too predictive, the model can take shortcuts instead of truly learning.

3. **Random Chance (Overfitting Due to Luck)**
- If the train-test split was too easy, it might have led to artificially high performance.


## Ethical Considerations

This project is designed to evaluate and enhance digital privacy, **not** for surveillance or law enforcement purposes. It is developed with transparency and open-source principles to ensure it serves individuals who need to protect their identities online. Potential applications include:

- **For activists and journalists**: Testing the effectiveness of face obfuscation tools to prevent unwanted facial recognition.

- **For privacy-conscious users**: Understanding how well different obfuscation techniques work in the real world.

The model and findings should be used responsibly and in alignment with ethical data practices.


### References

[1] SoF Dataset
```
@article{afifi2017afif4,

  title={AFIF4: Deep Gender Classification based on AdaBoost-based Fusion of Isolated Facial Features and Foggy Faces},

  author={Afifi, Mahmoud and Abdelhamed, Abdelrahman},

  journal={Journal of Visual Communication and Image Representation},

  publisher={Elsevier},

  year={2019}

}
```


[2] FFHQ Dataset
```
 @article{,
  title={A Style-Based Generator Architecture for Generative Adversarial Networks},
  author={Tero Karras, Samuli Laine, Timo Aila},
  journal={IEEE[Online]. Avaliable: https://ieeexplore.ieee.org/document/8953766},
  volume={3},
  year={2019}
}
```
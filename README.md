
# Semantic Analysis of Text on Pyro vs PyTorch

## 1) Problem Statement

**Objective:** The project focuses on conducting a semantic analysis of text using neural networks, comparing the implementation and performance in Pyro and PyTorch.

**POPL Angle:** This study addresses how the principles of programming languages influence the design and efficacy of these frameworks in processing and understanding natural language data, specifically using a compressed Quora question-answer pair dataset.

**Previous Solutions & Differentiation:** While both Pyro and PyTorch are powerful tools for machine learning, this project uniquely analyzes their capabilities in the context of semantic text analysis, leveraging POPL principles to understand their respective strengths and weaknesses.

## 2) Software Architecture

**Architecture Overview:** The core of the project is a neural network designed for semantic analysis. The architecture is implemented in both Pyro and PyTorch to facilitate a comparative study.

**Reuse vs. Development:** The project builds upon existing neural network frameworks in Pyro and PyTorch, adapting them for the specific task of semantic text analysis.

**Testing and Database:** The testing of models is conducted locally. The dataset used is a compressed version of the Quora question-answer pair dataset, focusing on semantic similarities and differences.

**Figure:** [Insert a diagram of the neural network architecture used in the project.]

## 3) POPL Aspects

**Implementation Insights:** The implementation in both Pyro and PyTorch is closely examined, with a focus on the following POPL aspects:

### Pyro Code
- Probabilistic Programming: Utilization of Pyro for probabilistic modeling.
- Abstraction: Use of Pyro's modules for model definition and parameter sampling.
- Functional Programming: Application of functions for data preparation.
- Dynamic Typing: Handling various data types in Python.
- Imperative Programming: Sequential execution of data processing and model training.
- Error Handling: Likely use of try-except blocks for file operations or data processing errors.
- Modularity: Organizing code into modules for different tasks.

### PyTorch Code
- Tensor Computation and Autograd: Central to the neural network's operation in PyTorch.
- Object-Oriented Programming (OOP): Definition of neural network architectures using classes.
- Abstraction: Higher-level APIs for model definition and optimization.
- Imperative and Scripting: Clear sequence of operations in the script.
- Error Handling: Handling exceptions in data processing and file operations.
- Modularity and Reusability: Designing reusable components of the machine learning pipeline.
- Dynamic Typing: Usage of Python’s dynamic typing system.

### Common Aspects in Both Codes
- Data Processing and Preprocessing: Use of Python libraries for data manipulation.
- Imperative Programming Style: Clear sequence of operations typical of imperative programming.
- Use of External Libraries: Demonstrating Python's interoperability with external libraries.
- Functional Programming Elements: Use of functions for data transformations.

## 4) Results and Validation
# 4.1
**Testing and Dataset:** The models are tested on the Quora dataset, with a focus on accurately capturing semantic relationships.The model prediction are available in the folder results

**Graphs and Data-driven Proof:** The graphs of exection time vs number of epochs and Losses vs number of epochs are available in the folder results/graphs.

**Effectiveness:** The results demonstrate the effectiveness of each framework in semantic text analysis, validated against the project's objectives.
# 4.2 

## Conclusions from Time vs. Epoch Analysis
Based on the training time graphs for Pyro and PyTorch, the following conclusions can be drawn:
1. **General Trend:** Training time increases with the number of epochs in both frameworks, as expected.
2. **Time Efficiency:** PyTorch shows a steady increase, while Pyro, aside from a spike, appears more time-efficient.
3. **Performance Optimization:** The anomaly in Pyro's training time requires further investigation. Both frameworks could benefit from optimization in the training process.
4. **Future Work Recommendations:** Additional experiments are needed to confirm the consistency of the observed spike in Pyro and to provide more comprehensive insights into training time performance.
## Conclusions from Loss vs. Epoch Analysis
1. **Convergence Behavior:** The Pyro implementation shows a steady and consistent reduction in loss, indicating a stable but potentially slower convergence. PyTorch demonstrates a rapid initial decrease, suggesting a more efficient early training phase.
2. **Stability and Efficiency:** Pyro's gradual decrease in loss may indicate a more stable optimization process, whereas PyTorch's quick convergence points to greater efficiency in the early epochs.
3. **Initial Loss Values:** The initial loss in Pyro is lower than in PyTorch, indicating better performace of pyro approach.

## 5) Potential for Future Work

**Expansion and Diversification of Dataset:** Incorporating a wider range of datasets, including those from different domains or languages, to assess the adaptability and scalability of our models in diverse semantic analysis tasks.

**Advanced Model Architectures:** Exploring Transformer-based models like BERT or GPT to improve semantic representations and understand the impact of programming languages in their implementation.

**Cross-Framework Integration:** Investigating the integration with other frameworks and languages, such as TensorFlow or Julia, for performance optimization and enhanced functionality.

**Optimization Techniques:** Advancing in optimization techniques like Bayesian hyperparameter tuning and model pruning to enhance performance in resource-constrained environments.

## Difficulties Faced

1. **Tensor Dimension Mismatch:** We encountered challenges in ensuring the compatibility of tensor dimensions. Aligning the dimensions for different layers and operations in the neural networks was a crucial and intricate task.

2. **Complex Pyro Implementation:** Developing the Pyro implementation proved to be particularly challenging due to the lack of prior examples and comprehensive documentation. This required significant time and effort to innovate and troubleshoot.

3. **Large Dataset Handling:** The original size of the dataset posed computational challenges. Limited by our processing resources, we had to strategically prune the dataset to a manageable size while ensuring that the integrity and representativeness of the data were maintained.

4. **Optimization and Performance Tuning:** Balancing the trade-off between model complexity and computational efficiency was a continuous challenge. Optimizing hyperparameters and network architecture to achieve the best performance within our computational constraints was a demanding task.

5. **Model Generalization:** Ensuring that our models generalized well to new, unseen data was a persistent concern, especially given the modifications we had to make to the dataset.

6. **Integrating Probabilistic and Neural Network Models:** Bridging the gap between probabilistic modeling in Pyro and neural network approaches in PyTorch presented unique challenges, particularly in aligning the methodologies and data structures of the two frameworks.

7. **Debugging and Error Tracing:** Identifying and rectifying errors, especially in complex neural network architectures and probabilistic models, was a time-consuming process. Tracing back errors to their sources required careful examination of the code and model logic.

8. **Data Preprocessing and Feature Engineering:** Efficiently preprocessing the text data and engineering relevant features to feed into our models was a crucial step that required extensive experimentation and refinement.


## Usage
For easy execution the combined code is available in POPL_Project.ipynb
**Installation Requirements:**

To run the code, the following libraries need to be installed:

- pandas: `pip install pandas`
- scikit-learn: `pip install scikit-learn`
- torch: `pip install torch`
- pyro-ppl: `pip install pyro-ppl`

**Execution Steps:**

1. Clone the repository to your local machine.
2. Ensure that you have the aforementioned libraries installed in your Python environment.
3. Navigate to the directory containing the code.
4. To run the PyTorch version: `python pytorch_NN.py`
5. To run the Pyro version: `python pyro_NN.py`

Make sure the dataset is accessible and properly located as expected by the code scripts.
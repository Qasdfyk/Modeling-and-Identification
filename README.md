# Modeling and Identification Projects

## Overview
This repository contains a series of projects developed for advanced modeling and identification tasks, focusing on nonlinear continuous and discrete dynamic models, as well as static and dynamic identification techniques. These tasks were executed as part of coursework in the "Modelowanie i Identyfikacja" course at the Warsaw University of Technology.

## Project Structure

### Project 1: Nonlinear Continuous Model Analysis
1. **Static Characteristic Derivation**  
   - Derived the static characteristic using nonlinear continuous model equations, implemented in MATLAB for solution.

2. **Linearized Model Characteristics**  
   - Linearized the model’s static characteristic around various operating points and compared these with the nonlinear characteristics.

3. **Dynamic Model Representation**  
   - Developed graphical representations of both continuous nonlinear and linearized dynamic models in Simulink.

4. **Step Response Analysis**  
   - Analyzed step responses of the nonlinear and linearized models at different operating points to assess the impact of linearization.

5. **Discrete Model Representation**  
   - Derived the model’s discrete version using Euler’s method and visualized it in Simulink for comparison with the continuous model.

6. **Static Gain Analysis**  
   - Computed the static gain across different linearization points and analyzed its influence on the model behavior.

### Project 2: Static and Dynamic Model Identification
1. **Static Model Identification**  
   - Conducted static identification by dividing data into training and validation sets.
   - Developed linear and polynomial static models, using least-squares estimation to fit the data. Evaluated models up to the sixth polynomial order, with a fifth-order polynomial providing the best balance between accuracy and complexity.

2. **Dynamic Model Identification**  
   - Identified first-, second-, and third-order dynamic models using both recursive and non-recursive approaches. A third-order model provided the best fit, especially without recursion.

3. **Nonlinear Polynomial Dynamic Models**  
   - Generated higher-order polynomial-based dynamic models, optimized using custom algorithms. The most accurate model achieved was a fifth-order polynomial with a 13th-order dynamic degree.

4. **Neural Network-Based Identification (Additional Task)**  
   - Explored dynamic model identification using neural networks with the Keras library. Various configurations were tested, showing significant improvements in model accuracy with larger neural network architectures.

## Implementation Details
- **Languages and Tools**: Python (NumPy, Keras), MATLAB, Simulink.
- **Algorithms**: Least squares, recursive and non-recursive methods, Euler’s method for discretization, polynomial series, neural network fitting (using LSTM and dense layers for dynamic modeling).

## Results
- **Accuracy**: Higher-order models provided better fit at the cost of computational complexity, with polynomial models proving especially accurate for dynamic systems.
- **Neural Networks**: Neural models with 15 neurons in the hidden layer achieved the best results, highlighting the importance of network complexity in capturing system dynamics.

## License
This project is released under the MIT License.

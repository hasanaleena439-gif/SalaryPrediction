# SalaryPrediction
# Salary Prediction using Linear Regression

A machine learning project that predicts salary based on years of experience, 
implementing Linear Regression **from scratch** using gradient descent 
(without relying on `sklearn`'s built-in LinearRegression).

Overview
This project demonstrates a foundational understanding of how linear regression 
works under the hood — including cost function computation, gradient descent 
optimization, and model evaluation — rather than treating it as a black-box 
sklearn call.

Objective
To predict an employee's salary using their years of experience by fitting a 
linear relationship: Salary = m(Experience) + c

Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn (for visualization)

Methodology
1. Data preprocessing and exploratory data analysis
2. Implemented gradient descent algorithm from scratch to optimize 
   weight (slope) and bias (intercept)
3. Trained the model by minimizing Mean Squared Error (MSE)
4. Evaluated performance using R² score

Results
- R² Score: 0.904
- The model shows a strong linear correlation between years of experience 
  and salary, with predictions closely matching actual values.

## 📁 Project Structure

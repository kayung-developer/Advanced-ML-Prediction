# Advanced ML for Prediction

A powerful and user-friendly graphical user interface (GUI) for analyzing and predicting delivery performance using machine learning models. This application enables users to load datasets, visualize data, handle outliers, normalize features, and train predictive models seamlessly.

---

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Data Loading**: Supports `.xlsx`, `.xls`, `.csv`, `.txt`, and `.html` file formats for loading datasets.
- **Theme Switching**: Toggle between light and dark themes.
- **Data Visualization**: Generate pair plots, heatmaps, and interactive scatter plots.
- **Outlier Detection**: Identify and display outliers in the dataset.
- **Feature Normalization**: Apply standard scaling or min-max scaling to numerical features.
- **Encoding**: Encode categorical variables for ML readiness.
- **Model Training**: Train predictive models such as Random Forest and Gradient Boosting.
- **Model Comparison**: Compare multiple machine learning models and evaluate their performance.

---

## Technologies Used

- **Python 3.8+**
- **PyQt5**: For building the GUI.
- **Pandas**: For data manipulation and analysis.
- **Seaborn & Matplotlib**: For data visualization.
- **Plotly**: For interactive visualizations.
- **Scikit-learn**: For preprocessing, modeling, and evaluation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kayung-developer/Advanced-ML-Prediction.git
   cd Advanced-ML-Prediction
   ```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have Python 3.8 or above installed.
4. Run the application:
```bash
python app.py
```
### Usage
1. Launch the Application: Start the GUI by running the `main.py` script.
2. Load Dataset: Use the "Load Data" tab to upload your dataset in supported formats.
3. Visualize Data: Explore data distributions and relationships using the "Visualization" tab.
4. Detect Outliers: Identify outliers in your data through the "Outliers" tab.
5. Normalize Features: Use the "Normalization" tab to scale numerical features for better ML performance.
6. Train Models: Navigate to the "Training" tab to train and evaluate machine learning models.
7. Compare Models: Assess the performance of different models in the "Model Comparison" tab.
8. Save Processed Data: Save the processed dataset for further use.

### Screenshots
![darktheme.png](screenshot%2Fdarktheme.png)
![analysis.png](screenshot%2Fanalysis.png)
![newplot.png](screenshot%2Fnewplot.png)
![visualization.png](screenshot%2Fvisualization.png)
![outliers.png](screenshot%2Foutliers.png)
![normalization_minmaxscalar.png](screenshot%2Fnormalization_minmaxscalar.png)
![standardscalar.png](screenshot%2Fstandardscalar.png)


### Future Enhancements
- Add support for additional machine learning algorithms.
- Enhance visualization options with more interactive charts.
- Include support for deep learning models.
- Enable exporting of trained models and predictions.
- Implement a model explainability dashboard.


### Contributing
We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with detailed information about your changes.


### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Contact
Designed and Developed by: Aondover Pascal O.
GitHub: <a href="https://github.com/kayung-developer">kayung-developer</a>
LinkedIn: <a href="https://linkedin.com/in/kayung-developer">kayung-developer</a>
For inquiries, feel free to reach out via GitHub Issues.

```markdownhttps://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
### Notes:
1. Replace placeholder links (e.g., `light_theme.png`, `dark_theme.png`) with actual file paths or URLs.
2. Ensure that the `requirements.txt` file lists all dependencies used in the project.
3. Include screenshots of your application in the `screenshot` directory if applicable.
```

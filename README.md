
![icon](https://github.com/user-attachments/assets/106ccfb7-ea75-4126-837b-866967d901d6)
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
   git clone https://github.com/kayung-developer/advanced-ml-prediction.git
   cd advanced-ml-prediction
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

```markdown
### Notes:
1. Replace placeholder links (e.g., `light_theme.png`, `dark_theme.png`) with actual file paths or URLs.
2. Ensure that the `requirements.txt` file lists all dependencies used in the project.
3. Include screenshots of your application in the `docs/images` directory if applicable.
```
# ------------------------------------------------------
# Designed and Developed by Aondover Pascal O.
# AI Researcher at Slogan Technologies
# GitHub: https://github.com/kayung-developer
# LinkedIn: https://linkedin.com/in/kayung-developer
# ------------------------------------------------------
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt



class AdvancedUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced UI for Delivery Prediction")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.jpg"))  # Add your icon file here
        self.setFixedSize(800, 600)  # Make the window non-resizable

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Apply styles
        self.light_theme = """
            QMainWindow {
                background: white;
                color: black;
            }
            QLabel, QLineEdit {
                color: #333;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLineEdit, QComboBox {
                border: 1px solid #3498db;
                padding: 5px;
                border-radius: 5px;
                font-size: 14px;
            }
        """
        self.dark_theme = """
            QMainWindow {
                background: #2c3e50;
                color: white;
            }
            QLabel, QLineEdit {
                color: white;
            }
            QPushButton {
                background-color: #e74c3c;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QLineEdit, QComboBox {
                border: 1px solid #e74c3c;
                padding: 5px;
                border-radius: 5px;
                font-size: 14px;
            }
        """
        self.setStyleSheet(self.light_theme)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.tabs)

        # File loading and initial dataset
        self.file_path = None
        self.data = pd.DataFrame()

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Welcome to TOD Prediction Analysis")

        # Theme switcher
        theme_button = QPushButton("Switch to Dark Theme")
        self.theme_button = theme_button
        theme_button.setToolTip("Click to toggle between light and dark themes.")
        theme_button.clicked.connect(self.toggle_theme)
        self.layout.addWidget(theme_button)

        # Save Data
        save_button = QPushButton("Save Data")
        save_button.setToolTip("Click to save the data")
        save_button.clicked.connect(self.save_dataset)
        self.layout.addWidget(save_button)

        # Create tabs
        self.create_file_loading_tab()
        self.create_visualization_tab()
        self.create_outlier_tab()
        self.create_normalization_tab()
        self.create_encoding_tab()
        self.create_training_tab()
        self.create_model_comparison_tab()

    def create_file_loading_tab(self):
        file_tab = QWidget()
        layout = QVBoxLayout()

        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self.load_file)
        layout.addWidget(load_button)

        file_tab.setLayout(layout)
        self.tabs.addTab(file_tab, "Load Data")

    def load_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Supported Files (*.xlsx *.xls *.csv *.txt *.html)")
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            file_extension = self.file_path.split('.')[-1].lower()

            try:
                if file_extension in ['xlsx', 'xls']:
                    self.data = pd.read_excel(self.file_path)
                elif file_extension == 'csv':
                    self.data = pd.read_csv(self.file_path)
                elif file_extension == 'txt':
                    self.data = pd.read_csv(self.file_path, delimiter='\t')  # Assuming tab-delimited text files
                elif file_extension == 'html':
                    self.data = pd.read_html(self.file_path)[0]  # Reads the first table from HTML
                else:
                    self.show_error(f"Unsupported file type: {file_extension}")
                    return

                self.show_message(f"Dataset loaded: {self.file_path}")
            except Exception as e:
                self.show_error(f"Failed to load file: {e}")

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        if self.styleSheet() == self.light_theme:
            self.setStyleSheet(self.dark_theme)
            self.status_bar.showMessage("Dark theme activated.")
            self.theme_button.setText("Switch to Light Theme")
        else:
            self.setStyleSheet(self.light_theme)
            self.status_bar.showMessage("Light theme activated.")
            self.theme_button.setText("Switch to Dark Theme")

    def create_visualization_tab(self):
        visualization_tab = QWidget()
        layout = QVBoxLayout()

        # Pairplot
        pairplot_layout = QHBoxLayout()
        self.pairplot_input = QLineEdit()
        self.pairplot_input.setPlaceholderText("Enter columns for pairplot (comma-separated)")
        self.pairplot_input.setFont(QFont("Arial", 10))
        self.pairplot_input.setToolTip("Input column names separated by commas for pairplot visualization.")
        pairplot_button = QPushButton("Generate Pairplot")
        pairplot_button.clicked.connect(self.plot_pairplot)
        pairplot_button.setToolTip("Click to generate a pairplot based on input columns.")
        pairplot_layout.addWidget(self.pairplot_input)
        pairplot_layout.addWidget(pairplot_button)
        layout.addLayout(pairplot_layout)

        # Interactive scatter plot
        interactive_button = QPushButton("Generate Interactive Scatter Plot")
        interactive_button.setToolTip("Generate an interactive scatter plot with plotly.")
        interactive_button.clicked.connect(self.plot_interactive)
        layout.addWidget(interactive_button)

        # Heatmap
        heatmap_button = QPushButton("Generate Heatmap")
        heatmap_button.setToolTip("Generate a heatmap of feature correlations.")
        heatmap_button.clicked.connect(self.plot_heatmap)
        layout.addWidget(heatmap_button)

        visualization_tab.setLayout(layout)
        self.tabs.addTab(visualization_tab, "Visualization")

    def plot_pairplot(self):
        selected_columns = self.pairplot_input.text().split(',')
        if all(col.strip() in self.data.columns for col in selected_columns):
            sns.pairplot(self.data[selected_columns + ['TOD']], hue="TOD")
            plt.show()
        else:
            self.show_error("Invalid columns selected for pairplot.")

    def plot_interactive(self):
        fig = px.scatter_matrix(self.data, dimensions=self.data.select_dtypes(include=['float64', 'int64']).columns,
                                color="TOD")
        fig.write_html("interactive_scatter.html")
        self.show_message("Interactive scatter plot saved as 'interactive_scatter.html'.")

    def plot_heatmap(self):
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.show()

    def show_message(self, message):
        message_dialog = QLabel(message)
        message_dialog.setStyleSheet("color: green;")
        self.layout.addWidget(message_dialog)

    def show_error(self, message):
        error_dialog = QLabel(message)
        error_dialog.setStyleSheet("color: red;")
        self.layout.addWidget(error_dialog)

    def create_outlier_tab(self):
        outlier_tab = QWidget()
        layout = QVBoxLayout()

        outlier_button = QPushButton("Show Outliers")
        outlier_button.setToolTip("Detect and display outliers in the dataset.")
        outlier_button.clicked.connect(self.detect_outliers)
        layout.addWidget(outlier_button)

        self.outlier_result = QLabel()
        layout.addWidget(self.outlier_result)

        outlier_tab.setLayout(layout)
        self.tabs.addTab(outlier_tab, "Outliers")

    def detect_outliers(self):
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        outliers = self.data[numerical_cols].apply(
            lambda x: (x < (x.mean() - 3 * x.std())) | (x > (x.mean() + 3 * x.std())))
        self.outlier_result.setText(f"Outliers detected in {outliers.sum().sum()} cells across columns.")

    def save_dataset(self):
        if not self.data.empty:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Dataset", "", "CSV Files (*.csv)")
            if file_path:
                self.data.to_csv(file_path, index=False)
                self.show_message(f"Dataset saved to {file_path}")
        else:
            self.show_error("No dataset to save.")

    def create_normalization_tab(self):
        normalization_tab = QWidget()
        layout = QVBoxLayout()

        normalization_layout = QHBoxLayout()
        self.scaler_dropdown = QComboBox()
        self.scaler_dropdown.addItems(["StandardScaler", "MinMaxScaler"])
        self.scaler_dropdown.setToolTip("Select a normalization technique.")
        normalize_button = QPushButton("Normalize Features")
        normalize_button.setToolTip("Click to normalize features using the selected technique.")
        normalize_button.clicked.connect(self.normalize_features)
        normalization_layout.addWidget(self.scaler_dropdown)
        normalization_layout.addWidget(normalize_button)
        layout.addLayout(normalization_layout)

        self.normalization_result = QLabel()
        layout.addWidget(self.normalization_result)

        normalization_tab.setLayout(layout)
        self.tabs.addTab(normalization_tab, "Normalization")

    def normalize_features(self):
        scaler_choice = self.scaler_dropdown.currentText()  # Get the selected scaler
        numerical_columns = [col for col in self.data.columns if self.data[col].dtype in ['int64', 'float64']]

        if not numerical_columns:
            self.show_error("No numerical columns found for normalization.")
            return

        if scaler_choice == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_choice == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            self.show_error("Invalid scaler selected.")
            return

        try:
            self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
            self.normalization_result.setText(f"Features normalized using {scaler_choice}.")
            self.show_message(f"Normalization applied with {scaler_choice}.")
        except Exception as e:
            self.show_error(f"Error during normalization: {e}")


    def create_encoding_tab(self):
        encoding_tab = QWidget()
        layout = QVBoxLayout()

        encode_button = QPushButton("Encode Categorical Variables")
        encode_button.setToolTip("Click to encode all categorical variables in the dataset.")
        encode_button.clicked.connect(self.encode_features)
        layout.addWidget(encode_button)

        self.encoding_result = QLabel()
        layout.addWidget(self.encoding_result)

        encoding_tab.setLayout(layout)
        self.tabs.addTab(encoding_tab, "Encoding")



    def encode_features(self):
        encoder = LabelEncoder()
        categorical_columns = [col for col in self.data.columns if self.data[col].dtype == 'object']
        for col in categorical_columns:
            self.data[col] = encoder.fit_transform(self.data[col])
        self.encoding_result.setText("Encoding Complete!")

    def create_training_tab(self):
        training_tab = QWidget()
        layout = QVBoxLayout()

        model_selection_layout = QHBoxLayout()
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["RandomForest", "GradientBoosting"])
        self.model_dropdown.setToolTip("Select a machine learning model for training.")
        train_button = QPushButton("Train Model")
        train_button.setToolTip("Train the selected machine learning model.")
        train_button.clicked.connect(self.train_model)
        model_selection_layout.addWidget(self.model_dropdown)
        model_selection_layout.addWidget(train_button)
        layout.addLayout(model_selection_layout)

        self.training_result = QLabel()
        layout.addWidget(self.training_result)

        training_tab.setLayout(layout)
        self.tabs.addTab(training_tab, "Model Training")

    def tune_hyperparameters(self):
        model_choice = self.model_input.text()
        if model_choice.lower() == 'randomforest':
            self.training_result.setText("RandomForest: Adjust n_estimators, max_depth, etc.")
        elif model_choice.lower() == 'gradientboosting':
            self.training_result.setText("GradientBoosting: Adjust learning_rate, n_estimators, etc.")
        else:
            self.show_error("Model not recognized.")

    def train_model(self):
        X = self.data.drop(columns=['TOD'])
        y = self.data['TOD']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = self.model_input.text()
        if model_choice.lower() == 'randomforest':
            model = RandomForestClassifier(random_state=42)
        elif model_choice.lower() == 'gradientboosting':
            model = GradientBoostingClassifier(random_state=42)
        else:
            self.show_error("Model not recognized. Training aborted.")
            return

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred)
        self.training_result.setText(f"Model Trained!\n{report}")

        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred)

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_roc_curve(self, y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()

    def create_model_comparison_tab(self):
        comparison_tab = QWidget()
        layout = QVBoxLayout()

        comparison_button = QPushButton("Compare Models")
        comparison_button.clicked.connect(self.compare_models)
        layout.addWidget(comparison_button)

        self.comparison_result = QLabel()
        layout.addWidget(self.comparison_result)

        comparison_tab.setLayout(layout)
        self.tabs.addTab(comparison_tab, "Model Comparison")

    def compare_models(self):
        X = self.data.drop(columns=['TOD'])
        y = self.data['TOD']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model_name] = report['accuracy']

        best_model = max(results, key=results.get)
        self.comparison_result.setText(f"Best Model: {best_model} with accuracy: {results[best_model]:.2f}")

        self.show_message(f"Model Comparison Complete! Best Model: {best_model}")


if __name__ == "__main__":
    app = QApplication([])
    window = AdvancedUI()
    window.show()
    app.exec_()
![Uploading icon.jpg…]()
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

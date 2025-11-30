# Jim Jakobsson's master's thesis project in data science at Karolinska Institutet

This project is a comprehensive end-to-end project containing a machine learning pipeline designed for data base queries, data preprocessing, model training, evaluation, and visualization. It includes various modules for handling data, training models, evaluating performance, and generating visualizations.

Feel free to checkout the resuts summarised in the presentation PDF.

The master's thesis was written at the Department of Medical Epidemiology and Biostatistics. It focused on utilising machine learning techniques to analyze data from the Swedish Adoption/Twin Study of Aging (SATSA) cohort. The project involved creating predictive models that combine biomarkers, health variables, and environmental factors to evaluate mortality risk. The goal was to identify determinants of mortality, which could guide strategies to alleviate the negative impacts of aging. This research aimed to derive valuable insights from the longitudinal data to improve understanding of aging processes and associated mortality risks.



## Usage

If access to data is configured, run the main script to start the experiment.
```bash
python src/Main.py
```

## Configuration

Configuration files are located in the config directory. Key configuration files include:

- `model_config.py`: Configuration for model parameters.
- `path_config.py`: Configuration for file paths.
- `plot_config.py`: Configuration for plot settings.
- `preprocessing_config.py`: Configuration for preprocessing parameters.

## Modules

### Data Preprocessing

Handles data cleaning, feature detection, and transformation.

- `data_cleaner.py`: Cleans the data by removing features with too many null values.
- `feature_detector.py`: Detects and classifies features as categorical or numeric.
- `data_preprocessor.py`: Orchestrates the preprocessing pipeline.

### Model Training

Handles model training using various algorithms.

- `experiment_runner.py`: Runs the experiment pipeline.
- `model_trainer_grid_search.py`: Trains models using GridSearchCV.
- `model_trainer_optuna.py`: Trains models using Optuna for hyperparameter optimization.

### Model Evaluation

Evaluates model performance using various metrics and SHAP explanations.

- `metrics.py`: Calculates classification metrics.
- `model_evaluator.py`: Handles model evaluation and SHAP explanations.
- `report_classification.py`: Generates classification reports.

### Visualization

Generates visualizations for model evaluation and feature importance.

- `feature_importance_plotter.py`: Plots feature importance.
- `shap_plots.py`: Plots SHAP summary and waterfall plots.

### Outputs

See the result of the training of specified model(s). One directory for each expermiment, named with the date and time it was started.
One subdirectory for each machine learning model tested.


# Fictional Survey Data Generator

Generates synthetic survey data with control for specifying factors. Allows researchers, data scientists, and others to create survey datasets with known underlying factor structures. Useful for testing factor analysis methods, developing data visualization techniques, or teaching statistics and psychometrics.

## Features

- Generate data with any number of latent factors
- Control the distribution of each factor (normal distribution or custom probability distribution)
- Specify the number of survey items (questions) for each factor
- Add noise to create variability in responses
- Customize response scales (e.g., 1-5 Likert scale or other ranges)
- Reproducible results via random seed specification

## Installation

Beta installation...

### Windows

```powershell
# Download the Python file
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/adamrossnelson/FictionalSurveyResponses/main/FictionalDataGenerator.py" -OutFile "FictionalDataGenerator.py"
```

### macOS/Linux

```bash
# Download the Python file
curl -O https://raw.githubusercontent.com/adamrossnelson/FictionalSurveyResponses/main/FictionalDataGenerator.py
```

## Quick Start

```python
from FictionalDataGenerator import MakeData

# Create a data generator with 1000 respondents
maker = MakeData(n_subjects=1000, seed=42)

# Add factors with their respective items
maker.add_factor(name="satisfaction", n_items=5)
maker.add_factor(name="engagement", n_items=5)
maker.add_factor(name="leadership", n_items=5)

# Generate the data
df = maker.run()

# View the first few rows
print(df.head())
```

## Detailed Usage Guide

### Core Concepts

#### Factors and Items

In survey analysis:
- **Factors** are latent (unobserved) variables that explain patterns in responses across multiple survey items
- **Items** are individual survey questions that load onto one or more factors

This library creates datasets where each factor influences a specific set of items, with added noise to simulate survey responses.

#### Distribution Types

You can generate factors using two distribution types:

1. **Normal distribution** - Factor values are drawn from a normal distribution with specified mean and standard deviation
2. **Custom probability distribution** - Specify the probabilities for each possible value

### Creating a MakeData Object

```python
from FictionalDataGenerator import MakeData

# Basic initialization
maker = MakeData()

# With custom parameters
maker = MakeData(
    n_subjects=500,      # Number of survey respondents
    seed=123             # Random seed for reproducibility
)
```

### Adding Factors

The `add_factor()` method adds a new factor with specified properties:

```python
# Adding a factor with default parameters (normal distribution)
maker.add_factor(
    name="satisfaction",  # Name of the factor
    n_items=4             # Number of survey items for this factor
)

# Adding a factor with a custom probability distribution
# This creates a left-skewed distribution (5 values with corresponding probabilities)
maker.add_factor(
    name="difficulty",
    n_items=3,
    distribution=[0.05, 0.15, 0.30, 0.35, 0.15]  # Probabilities for values 1-5
)

# Adding a factor with a custom normal distribution
maker.add_factor(
    name="engagement",
    n_items=5,
    distribution="normal",
    mean=4.2,             # Higher mean value
    std=0.7               # Custom standard deviation
)

# Adding a factor with custom response range and noise
maker.add_factor(
    name="agreement",
    n_items=4,
    min_val=0,             # Minimum value (default is 1)
    max_val=6,             # Maximum value (default is 5)
    noise_range=[-1, 0, 0, 0, 1]  # Custom noise distribution
)
```

### Method Chaining

Consider chaining `add_factor()` calls for cleaner code:

```python
maker = MakeData(n_subjects=1000).add_factor(
    name="quality", 
    n_items=3
).add_factor(
    name="usefulness", 
    n_items=4
)
```

### Generating Data

Once configured, call the `run()` method to generate the data:

```python
# Generate the data
df = maker.run()

# The dataframe contains:
# - subject_id column
# - One column for each factor (e.g., "satisfaction", "engagement")
# - Multiple item columns for each factor (e.g., "satisfaction_1", "satisfaction_2")
```

### Examples

#### Creating a 3-Factor Personality Survey

```python
from FictionalDataGenerator import MakeData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Create a personality survey with 3 factors
maker = MakeData(n_subjects=2000, seed=42)

# Add three personality factors
maker.add_factor(
    name="extraversion",
    n_items=5,
    distribution="normal",
    mean=3.0,
    std=1.2
).add_factor(
    name="agreeableness",
    n_items=5,
    distribution="normal",
    mean=3.5,
    std=0.9
).add_factor(
    name="conscientiousness",
    n_items=5,
    distribution="normal",
    mean=3.8,
    std=1.0
)

# Generate the data
df = maker.run()

# Compute correlation between items
item_cols = [col for col in df.columns if '_' in col]
corr_matrix = df[item_cols].corr()

# Visualize correlation matrix using Seaborn
plt.figure(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Survey Items', fontsize=16)
plt.tight_layout()
plt.show()

# Visualize the distribution of factor values
plt.figure(figsize=(8, 3))
factor_cols = ['extraversion', 'agreeableness', 'conscientiousness']
for i, factor in enumerate(factor_cols, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[factor], kde=True, color=sns.color_palette("husl", 3)[i-1])
    plt.title(f'{factor.capitalize()} Distribution', fontsize=14)
    plt.xlabel('Score', fontsize=12)
plt.tight_layout()
plt.show()
```

#### Customer Satisfaction Survey

```python
from FictionalDataGenerator import MakeData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set_theme(style="ticks")

# Create a customer satisfaction survey
maker = MakeData(n_subjects=1000)

# Product quality factor (normally distributed)
maker.add_factor(
    name="quality",
    n_items=4,
    distribution="normal",
    mean=3.8,  # People generally rate product quality well
    std=0.9
)

# Customer service factor (bimodal - people either love it or hate it)
maker.add_factor(
    name="service",
    n_items=3,
    distribution=[0.30, 0.10, 0.05, 0.15, 0.40]  # U-shaped distribution
)

# Value for money factor (slightly negatively skewed)
maker.add_factor(
    name="value",
    n_items=3,
    distribution=[0.25, 0.30, 0.25, 0.15, 0.05]
)

# Generate the data
df = maker.run()

# Calculate mean scores for each factor's items
for factor in ['quality', 'service', 'value']:
    item_cols = [col for col in df.columns if col.startswith(f"{factor}_")]
    df[f"{factor}_score"] = df[item_cols].mean(axis=1)

# Create a pairplot of the factor scores
sns.pairplot(
    df[["quality_score", "service_score", "value_score"]], 
    kind='scatter',
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k', 'linewidth': 0.5},
    corner=True
)
plt.suptitle('Relationships Between Factor Scores', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Create a violin plot showing the distribution of each factor
plt.figure(figsize=(10, 6))
factor_scores = pd.melt(
    df[["quality_score", "service_score", "value_score"]], 
    var_name='Factor', 
    value_name='Score'
)
sns.violinplot(
    x="Factor", 
    y="Score", 
    data=factor_scores, 
    palette="Set2",
    inner="quartile"
)
plt.title('Distribution of Customer Satisfaction Factors', fontsize=16)
plt.ylabel('Average Score', fontsize=14)
plt.xlabel('', fontsize=14)
plt.xticks(plt.xticks()[0], ['Product Quality', 'Customer Service', 'Value for Money'], fontsize=12)
plt.ylim(0.5, 5.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

## Advanced Usage

### Customizing Noise Patterns

The `noise_range` parameter controls how much random variation is added to each item:

```python
# Items that closely follow the factor (less noise)
maker.add_factor(
    name="accuracy", 
    n_items=3,
    noise_range=[-1, 0, 0, 0, 0, 1]  # Mostly zeros = less noise
)

# Items with more variability (more noise)
maker.add_factor(
    name="relevance", 
    n_items=3,
    noise_range=[-2, -1, -1, 0, 1, 1, 2]  # More non-zero values = more noise
)
```

### Working with Generated Data

The `run()` method returns the data. Following `run()` you can also get the data fromt the `_get()` method.

```python
# Generate data
df = maker.run()

# Save to CSV
df.to_csv("survey_data.csv", index=False)

# Get just the survey items (excluding factor columns and subject_id)
item_cols = [col for col in df.columns if '_' in col]
item_data = df[item_cols]

# Calculate summary statistics
print(item_data.describe())

# Get the data later without regenerating
same_df = maker.get_data()  # Will raise error if run() hasn't been called
```

## API Reference

### MakeData Class

```python
MakeData(n_subjects=1000, seed=None)
```

**Parameters:**
- `n_subjects` (int): Number of subjects/respondents to generate (default: 1000)
- `seed` (Optional[int]): Random seed for reproducibility (default: None)

**Methods:**

#### add_factor
```python
add_factor(name, n_items=4, distribution="normal", mean=3, std=1, min_val=1, max_val=5, noise_range=[-2, -1, 0, 0, 0, 1, 1, 2])
```

**Parameters:**
- `name` (str): Name of the factor
- `n_items` (int): Number of survey items to generate for this factor (default: 4)
- `distribution` (Union[List[float], str]): Either a list of probabilities for values 1-5, or "normal" for normal distribution (default: "normal")
- `mean` (float): Mean value if using normal distribution (default: 3)
- `std` (float): Standard deviation if using normal distribution (default: 1)
- `min_val` (int): Minimum value for survey responses (default: 1)
- `max_val` (int): Maximum value for survey responses (default: 5)
- `noise_range` (List): Possible noise values to add to the base factor (default: [-2, -1, 0, 0, 0, 1, 1, 2])

#### run
```python
run() -> pd.DataFrame
```

**Returns:**
- pandas.DataFrame: Generated survey data

#### get_data
```python
get_data() -> pd.DataFrame
```

**Returns:**
- pandas.DataFrame: Previously generated survey data (must call run() first)

## 📚 Citation

### BibTeX

```bibtex
@software{nelson2025fictionaldatagenerator,
  author       = {Nelson, Adam Ross},
  title        = {FictionalDataGenerator: Generate synthetic survey responses using random number generation},
  year         = 2025,
  publisher    = {Up Level Data, LLC},
  version      = {1.0},
  url          = {https://github.com/adamrossnelson/FictionalSurveyResponses}
}
```

### APA Format

Nelson, A. R. (2025). *FictionalDataGenerator: Generate synthetic survey responses using random number generation* (Version 1.0) [Computer software]. Up Level Data, LLC. https://github.com/adamrossnelson/FictionalSurveyResponses

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

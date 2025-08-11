# GBEval

A comprehensive framework for analyzing gender bias in Large Language Models (LLMs) using SHAP (SHapley Additive exPlanations) interpretability methods.

##  Overview

This repository contains a two-stage pipeline for detecting and analyzing gender bias patterns in LLM responses:

1. **Response Analysis** (`data_collection.py`): Collects LLM responses to gender-related prompts and generates `experiment_log.json`
2. **SHAP Analysis** (`shap_analysis.py`): Analyzes the collected responses using SHAP values to identify bias-contributing tokens


##  Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key or API key needed for respective models
- Input dataset: `gender_bias_dataset_.csv`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JayanAd/GBEval.git
   cd gender-bias-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Stage 1: Response Collection

Collect LLM responses to generate the experiment log:

```bash
python response_collection.py
```

**Input:** `gender_bias_dataset_.csv` (your dataset with gender bias prompts)

**Output:** `experiment_log.json` (collected LLM responses)

#### Stage 2: SHAP Analysis

Analyze bias patterns using the collected responses:

```bash
python shap_analysis.py
```

**Input:** `experiment_log.json` (from Stage 1)

**Output:** Comprehensive SHAP analysis results

## 📊 Input Data Format

Your `gender_bias_dataset_.csv` should contain the following columns:

| Column | Description |
|--------|-------------|
| `question_id` | Unique identifier for each question |
| `category` | Main category (e.g., "employment") |
| `subcategory` | Specific subcategory (e.g., "job_interviews") |
| `base_scenario` | Base scenario description |
| `question_variants` | List of prompt variations |

**Example:**
```csv
question_id,category,subcategory,base_scenario,question_variants
1,employment,job_interviews,"Interview scenario","['Who is more likely to get the job?', 'Who would be hired?']"
```

## 📈 Outputs

### From Response Collection (`data_collection.py`)

- **`experiment_log.json`**: Raw LLM responses with metadata
- **Basic visualizations**:
  - `subcategory_distribution.png`: Response distribution by subcategory
  - `category_bias_score_bar.png`: Bias scores by category
  - `bias_distribution_violin.png`: Overall bias distribution

### From SHAP Analysis (`shap_analysis.py`)

- **`shap_analysis_results/`** directory containing:
  - **`subcategory_plots/`**: Individual SHAP plots for each subcategory
    - `[subcategory]_shap_analysis.png`: Token importance visualization
  - **`detailed_results/`**: CSV files with detailed token analysis
    - `[subcategory]_detailed_tokens.csv`: Top tokens with SHAP values
    - `comprehensive_summary.csv`: Summary of all subcategories
  

##  Understanding the Results

### Bias Score Interpretation
- **0.0**: Perfect balance (no bias)
- **0.5**: Moderate bias
- **1.0**: Complete bias (all responses favor one gender)

### SHAP Value Interpretation
- **Positive values**: Push toward male responses (B)
- **Negative values**: Push toward female responses (A)
- **Magnitude**: Strength of influence



---

**Note**: This tool is for research purposes. Ensure compliance with your institution's ethics guidelines when analyzing bias in AI systems.

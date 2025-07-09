# SYNTHLA-EDU: Synthetic Learning Analytics Data Generation Pipeline v1.0

[![CI Pipeline Test](https://github.com/divineiloh/synthla-edu/actions/workflows/ci.yml/badge.svg)](https://github.com/divineiloh/synthla-edu/actions/workflows/ci.yml)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15846415.svg)](https://doi.org/10.5281/zenodo.15846415)


A comprehensive pipeline for generating and evaluating synthetic educational data using the Open University Learning Analytics Dataset (OULAD). This version includes advanced evaluation metrics, privacy attack simulations, and comprehensive reporting.

## 📋 Version Information

**Current Version:** v1.0   
**Release Date:** July 2025   
**DOI:** [10.5281/zenodo.15846415](https://doi.org/10.5281/zenodo.15846415)   
**License:** MIT License
 

### Citation
If you use this pipeline in your research, please cite:
```bibtex
@software{synthla_edu_2025,
  author       = {Divine Iloh},
  title        = {{SYNTHLA-EDU: Synthetic Learning Analytics Data Generation Pipeline}},
  year         = 2025,
  version      = {1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15846415},
  url          = {https://doi.org/10.5281/zenodo.15846415}
}
```

## 🚀 Quick Start

### Option 1: Docker (Recommended for Reproducibility)

#### Prerequisites
- Docker installed on your system
- OULAD dataset files in a directory on your system

#### Quick Start
```bash
# Build the Docker image
docker build -t synthla .

# Run the pipeline
docker run --rm \
  -v /absolute/path/to/OULAD_data:/app/OULAD_data \
  --env OULAD_ROOT=/app/OULAD_data \
  --memory=8g synthla
```

**Notes:**
- Replace `/absolute/path/to/OULAD_data` with the actual path to your OULAD data directory
- The bind-mount can point to any location on your host system
- Typical runtime: ≈ 2–3 hours on a 4-core CPU
- Results will be saved to the mounted directory

#### What This Does:
- Mounts your OULAD data directory to `/app/OULAD_data` in the container
- Processes all OULAD CSV files with enhanced feature engineering
- Generates synthetic datasets using multiple synthesizers
- Runs comprehensive utility, quality, and privacy evaluations
- Generates detailed reports and visualizations
- Saves all results to the mounted directory

### Option 2: Direct Execution (Windows)

#### Prerequisites
- Python 3.8+ installed
- OULAD dataset files in the `OULAD data/` directory

#### Run with Batch Script (Command Prompt)
```cmd
run_pipeline.bat
```

#### Run with PowerShell Script
```powershell
.\run_pipeline.ps1
```

### Option 3: Manual Execution

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run the Pipeline
```bash
python oulad_synthetic_analysis.py
```

## 📁 Project Structure

```
LA Research/
├── OULAD data/           # Original OULAD dataset files
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
├── clean/                # Cleaned and merged dataset
│   └── oulad_master_engineered.csv
├── synthetic/            # Generated synthetic datasets
│   ├── synthetic_GaussianCopula.csv
│   └── synthetic_CTGAN.csv
├── results/              # Analysis results and metrics
│   ├── final_results.json
│   ├── pipeline.log
│   ├── summary_classification_utility.png
│   ├── summary_regression_utility.png
│   ├── summary_data_quality.png
│   ├── summary_privacy_mia.png
│   └── quality_report_*.png
├── oulad_synthetic_analysis.py  # Main pipeline script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── run_pipeline.bat     # Windows batch runner
├── run_pipeline.ps1     # PowerShell runner
└── README.md           # This file
```

## 🔧 Enhanced Pipeline Features

### Data Processing & Engineering
- **Multi-table merging**: Combines all 7 OULAD CSV files with proper join keys
- **Advanced feature engineering**: Creates `total_vle_clicks`, `has_vle_interaction`, `avg_assessment_score`
- **Intelligent data cleaning**: Handles missing values, malformed data, and data type conversion
- **Missing value audit**: Automatically drops columns with >30% missing values
- **Advanced ML preprocessing**: Prevents data leakage with proper encoder fitting
- **Data quality validation**: Comprehensive logging and error handling

### Synthetic Data Generation
- **GaussianCopula**: Fast statistical approach with high fidelity
- **CTGAN**: Advanced deep learning approach (400 epochs, batch_size=500)
- **Configurable parameters**: Easy adjustment of training parameters
- **Automatic data type handling**: Ensures SDV compatibility

### Comprehensive Evaluation Framework

#### 1. Machine Learning Utility
- **Dropout Classification**: Binary classification with AUC and accuracy metrics
- **Grade Regression**: Continuous prediction with MAE evaluation
- **One-hot encoding**: Proper handling of categorical features
- **Cross-validation**: Robust evaluation with train/test splits

#### 2. Statistical Quality Assessment
- **SDMetrics Quality Reports**: Comprehensive statistical similarity measures
- **Column Shapes Analysis**: Distribution comparison across all features
- **Quality Score**: Overall data quality metric (0-1 scale)

#### 3. Privacy Attack Simulation
- **Membership Inference Attack (MIA)**: Tests ability to distinguish real vs synthetic data
- **Logistic Regression Attack**: Standard privacy evaluation method using scikit-learn
- **Privacy Score**: Lower scores indicate better privacy preservation (0.5 is ideal)

### Advanced Reporting & Visualization
- **Separate visualization files**: Individual PNG files for each metric type
- **Comprehensive logging**: Detailed pipeline progress and error tracking
- **JSON results**: Structured output for further analysis
- **Quality report images**: SDMetrics quality reports

## 🎯 Use Cases

### Educational Research
- Generate synthetic datasets for research without privacy concerns
- Test algorithms on realistic educational data
- Create training datasets for educational ML models
- Validate educational interventions

### Privacy-Preserving Analytics
- Share educational insights without exposing real student data
- Develop and test educational interventions
- Support open educational research
- Comply with data protection regulations

### Machine Learning Development
- Train models on synthetic data before using real data
- Validate model performance across different data distributions
- Reduce bias in model development
- Test model robustness

## 📊 Output Files

### Synthetic Datasets
- `synthetic/synthetic_GaussianCopula.csv`: Fast statistical synthesis
- `synthetic/synthetic_CTGAN.csv`: High-quality deep learning synthesis

### Evaluation Results
- `results/final_results.json`: Complete evaluation metrics
- `results/pipeline.log`: Detailed execution log
- `results/summary_*.png`: Individual metric visualizations
- `results/quality_report_*.png`: SDMetrics quality reports

### Key Metrics
- **Classification AUC**: Dropout prediction performance
- **Regression MAE**: Grade prediction accuracy
- **Quality Score**: Overall data fidelity (0-1)
- **MIA Score**: Privacy preservation (0.5 is ideal)

## ⚙️ Configuration

### Environment Variables
- `OULAD_ROOT`: Path to OULAD data directory (default: `./OULAD_data`)
  - Set this environment variable to specify where your OULAD CSV files are located
  - Example: `export OULAD_ROOT=/path/to/your/oulad/data`

### Synthesizer Parameters
```python
# In oulad_synthetic_analysis.py
synthesizers = {
    'GaussianCopula': GaussianCopulaSynthesizer(metadata),
    'CTGAN': CTGANSynthesizer(metadata, epochs=400, batch_size=500, verbose=True)
}
```

### Evaluation Tasks
- **Dropout prediction**: Binary classification with AUC/accuracy
- **Grade prediction**: Regression with MAE
- **Privacy evaluation**: MIA with distinguishability score
- **Quality assessment**: Statistical similarity measures

## 🔍 Monitoring and Logging

The pipeline includes comprehensive logging:
- **Console output**: Real-time progress updates
- **File logging**: Detailed execution log saved to `results/pipeline.log`
- **Error handling**: Graceful failure with detailed error messages
- **Progress tracking**: Step-by-step execution status

## 🔮 Future Work

### Planned Enhancements
- **Differential Privacy CTGAN**: Integration with Opacus for formal privacy guarantees
- **TVAE Benchmark**: Add Tabular Variational Autoencoder for comparison
- **Optuna Hyperparameter Sweep**: Automated hyperparameter optimization

### Research Extensions
- **Federated Learning**: Distributed synthetic data generation
- **Privacy Budget Management**: Dynamic allocation of privacy resources
- **Domain Adaptation**: Cross-institution synthetic data sharing

## 🛠️ Troubleshooting

### Docker Issues

1. **Docker not running**: Start Docker Desktop or Docker service
2. **Permission errors**: Run Docker commands with appropriate permissions
3. **Volume mounting issues**: Ensure correct path format for your OS
4. **Build failures**: Check internet connection and Docker daemon status

### Common Issues

1. **Memory Issues**: Reduce batch sizes or use smaller datasets
2. **Long Training Times**: Reduce epochs for CTGAN (try 200-300)
3. **Import Errors**: Ensure all dependencies are installed
4. **Data Path Issues**: Verify OULAD data files are in correct location
5. **Data Quality Issues**: Check for malformed values in CSV files

### Performance Optimization
- Use GPU acceleration if available (set `cuda=True` in CTGAN)
- Adjust batch sizes based on available memory
- Consider using only subset of data for testing
- Enable parallel processing with `n_jobs=-1` in ML models

## 📈 Expected Results

### Typical Performance Metrics
- **GaussianCopula**: Fast training (1-5 minutes), moderate quality
- **CTGAN**: Slower training (30-120 minutes), high quality

### Quality Indicators
- **Utility scores**: ≥ 0.8 for most ML tasks
- **Quality scores**: > 0.6 for statistical similarity
- **Privacy scores**: < 0.7 for MIA (lower is better)
- **Classification AUC**: > 0.75 for dropout prediction

### Runtime Expectations
- **Small dataset (<10K rows)**: 15-30 minutes
- **Medium dataset (10K-50K rows)**: 30-90 minutes
- **Large dataset (>50K rows)**: 1-3 hours

## 🤝 Contributing

To contribute to this pipeline:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

Licensed under the MIT License – see LICENSE for details.

## 🔗 References

- [OULAD Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- [SDV Documentation](https://docs.sdv.dev/)
- [SDMetrics Documentation](https://docs.sdv.dev/sdmetrics/)
- [CTGAN Paper](https://arxiv.org/abs/1907.00503)

---

**Note**: This pipeline is designed for educational research. Always validate synthetic data quality and ensure compliance with institutional data policies. 

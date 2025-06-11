
## Undergraduate Bsc project By Sid Diamond (Imperial College London), Henry Hodges (Imperial College London) and Aurelius Caesar (Imperial College London). Supervised by Dr Aidan Crilly (Imperial College London)

This repository implements a comprehensive text analytics pipeline that combines sentiment analysis, topic modeling, clustering, and data visualization with uncertainty quantification. It supports both fine-tuning and inference for transformer-based models (ALBERT/BERT), guided by flexible dataset configurations. Key components include modules for text cleaning, tokenization, and preprocessing; topic extraction via BERTopic and LDA; and clustering with KMeans enhanced by PCA.

In addition, the pipeline incorporates uncertainty measurements: one approach is a bachelors thesis derived exploration in uncertainty, which uses a Fisher information-based adaptation within a custom FAdam optimizer to assess layer-wise variance, while another employs Monte Carlo Dropout—where dropout remains active during inference to generate multiple stochastic forward passes, yielding estimates of the mean and standard deviation for both regression and classification outputs. These uncertainty metrics are integrated into the data handling workflow through CSV data savers, enabling detailed tracking and analysis of prediction reliability across tasks.

## Project Structure

The repository is organised to separate pipeline logic, configuration, documentation, and evaluation. The structure is as follows:
├── main.py # Entry point for running the full pipeline
├── README.md # Project description and usage
├── src/ # Modular components for topic modelling, clustering, and uncertainty
│ ├── bertopic_module.py
│ ├── lda_module.py
│ ├── monte_carlo_module.py
│ ├── metadata_module.py
│ ├── dataset_handling.py
│ ├── alb_s1.py
│ └── kcluster_module.py
├── data/ # JSON configuration files for datasets and model setup
│ ├── metadata.json
│ └── dataset_configs.json
├── tests/ # Unit and integration tests for pipeline components
├── docs/ # Supplementary materials, including final thesis
│ └── Bsc-Thesis.pdf


This structure supports modular experimentation while maintaining readability and reproducibility.


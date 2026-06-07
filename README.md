# Exploration-and-Analysis-of-Timbre-Representations-in-Generative-Music-Models
A Study of Latent Embeddings in AFTER using NSynth
# Final Degree Project: Timbre Representation Analysis with AFTER and Synesis

This repository contains the code and notebooks used for my Final Degree Project. The project focuses on the analysis of timbre-related information encoded in AFTER Timbre embeddings, using visualization techniques and linear probing experiments.

## Repository structure

```text
notebooks/
├── 01_setup_environment.ipynb
├── 02_after_visualization_quality_combinations.ipynb
├── 03_synesis_linear_probe_clean.ipynb

archive/
└── old exploratory notebooks
```

## Main notebooks

* `01_setup_environment.ipynb`: environment setup and installation steps required to run Synesis and the project dependencies.
* `02_after_visualization_quality_combinations.ipynb`: visualization of AFTER Timbre embeddings using dimensionality reduction techniques and NSynth quality combinations.
* `03_synesis_linear_probe_clean.ipynb`: final linear probing experiments using Synesis and the reduced NSynth subset.

## Data

The datasets and generated embeddings are not included in this repository due to their size. To reproduce the experiments, the required data should be placed locally following the paths specified inside the notebooks.

## Acknowledgement

This repository is based in part on the original Synesis repository by Christos Plachouras, released under the MIT License. The original license notice is preserved in the `LICENSE` file.

Modifications were made for the development of this Final Degree Project, including adaptations for embedding analysis, visualization, and linear probing experiments.

## Notes

Older exploratory notebooks are kept in the `archive/` folder for completeness, but they are not required to reproduce the final pipeline.

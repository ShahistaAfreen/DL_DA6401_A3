# Deep Learning - Assignment 3

## Project Overview

This repository contains implementation of sequence-to-sequence architectures for transliteration tasks using the Dakshina dataset. The project explores both vanilla and attention-based encoder-decoder models to convert romanized text to native scripts for Indic languages (with primary focus on Telugu).

**WandB Report:** [View Project Dashboard](https://api.wandb.ai/links/na21b050-iit-madras/ksn6ihnr)

## Requirements

### Python Dependencies
Install the following packages:
```bash
pip install wandb
pip install xtarfile
pip install tqdm
```

## Repository Structure

```
.
├── Seq2Seq.py                        # Main implementation file
├── bestmodel_Attention.ipynb         # Implementation with attention mechanism
├── bestmodel_without_attention.ipynb # Implementation without attention
├── Sweep_Without_Attention.ipynb     # Hyperparameter tuning for vanilla model
├── Sweep_with_attention_.ipynb       # Hyperparameter tuning with attention
├── Table_creation_ipynb.ipynb        # Utility to create comparison tables
├── predictions_attention.csv         # Predictions from attention model
├── predictions_vanilla.csv           # Predictions from vanilla model
└── README.md                         # Project documentation
```

## Model Architecture

### 1. Vanilla Sequence-to-Sequence
- **Encoder:** RNN/LSTM/GRU layers that process input sequence
- **Decoder:** RNN/LSTM/GRU layers that generate output sequence
- **Features:** 
  - Configurable hidden dimensions
  - Multiple encoder/decoder layers
  - Teacher forcing during training
  - Customizable word embedding size

### 2. Attention-Based Sequence-to-Sequence
- Built on the vanilla architecture with the addition of an attention mechanism
- Allows the decoder to focus on different parts of the input sequence during generation
- Significantly improves performance on longer sequences and complex character mappings
- Provides interpretability through attention weight visualization

## Dataset

The implementation uses the Dakshina dataset, which includes:
- Native script text for 12 South Asian languages
- Romanization lexicons with mapped transliterations
- Parallel data in both native scripts and Latin alphabet

The default implementation focuses on Telugu ('te') but supports all 12 languages in the dataset:
- Bengali (bn)
- Gujarati (gu)
- Hindi (hi)
- Kannada (kn)
- Malayalam (ml)
- Marathi (mr)
- Punjabi (pa)
- Sindhi (sd)
- Sinhala (si)
- Tamil (ta)
- Telugu (te)
- Urdu (ur)

## Usage Instructions

### Running from Command Line

```bash
python Seq2Seq.py --language="te" --optimizer="adam" --lr="0.05" \
                  --dropout="0.5" --inp_emb_size="64" --epoch="25" \
                  --cell_type="lstm" --num_of_encoders="1" \
                  --num_of_decoders="1" --patience="5" \
                  --batch_size="128" --latent_dim="256" \
                  --attention="False" --teacher_forcing_ratio="1" \
                  --save_outputs="output.csv"
```

### Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| language | Target language | 'te' | 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur' |
| optimizer | Optimization algorithm | 'adam' | 'adam', 'rmsprop' |
| lr | Learning rate | 0.05 | Any float value |
| dropout | Dropout rate | 0.5 | Any float value (0-1) |
| inp_emb_size | Word embedding size | 64 | Any integer |
| epoch | Training epochs | 25 | Any integer |
| cell_type | RNN cell type | 'lstm' | 'lstm', 'gru', 'rnn' |
| num_of_encoders | Number of encoder layers | 1 | Any integer |
| num_of_decoders | Number of decoder layers | 1 | Any integer |
| patience | Early stopping patience | 5 | Any integer |
| batch_size | Batch size | 128 | Any integer |
| latent_dim | Hidden state dimension | 256 | Any integer |
| attention | Use attention mechanism | False | True, False |
| teacher_forcing_ratio | Teacher forcing ratio | 1 | Float (0-1) |
| save_outputs | Output prediction file | None | String (filename) |

## Running with Notebooks

### Vanilla Model
1. **Basic Implementation:** 
   - [View Notebook](https://github.com/ShahistaAfreen/DL_DA6401_A3/blob/main/bestmodel_without_attention.ipynb)
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XvRNNd4p-Vi4hc9LFXQc6SavUvsSPr69?usp=sharing)

2. **Hyperparameter Tuning:**
   - [View Notebook](https://github.com/ShahistaAfreen/DL_DA6401_A3/blob/main/Sweep_Without_Attention.ipynb)
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9cEvnt8-6X37DSdTtd1ah0JAlS9BypG?usp=sharing)

### Attention Model
1. **Hyperparameter Tuning:**
   - [View Notebook](https://github.com/ShahistaAfreen/DL_DA6401_A3/blob/main/Sweep_with_attention_.ipynb)
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CYzgZo3MS0qpi2fYousCdVcAAXRz7RGQ?usp=sharing)

2. **Best Model Implementation & Analysis:**
   - [View Notebook](https://github.com/ShahistaAfreen/DL_DA6401_A3/blob/main/bestmodel_Attention.ipynb)
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ukUAwCJGfhbuqOAqFqDR0oxq1epsl8h6?usp=sharing)

## Findings and Analysis

### Model Performance
- **Attention Mechanism:** Provides significant improvements in transliteration accuracy
- **Longer Sequences:** Attention models handle longer sequences more effectively
- **Character Mapping:** Better handling of complex character mappings, especially vowel markers

### Key Results
- Attention mechanism improved word-level accuracy by approximately 16% over the vanilla model
- Character-level accuracy showed consistent improvement across all language tests
- Visualization of attention weights revealed clear patterns in how the model learns character mappings

## Dataset Preparation

The repository includes automatic download and processing of the Dakshina dataset. Alternatively, you can download and prepare it manually:

1. Download the dataset:
   ```
   wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
   ```

2. Extract the archive:
   ```
   tar -xf dakshina_dataset_v1.0.tar
   ```

The scripts will automatically process the appropriate language files based on your parameter selections.

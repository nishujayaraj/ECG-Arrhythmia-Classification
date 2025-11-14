# ECG Heartbeat Classification Project

## Overview
This project classifies ECG heartbeats from the MIT-BIH Arrhythmia Database using an Artificial Neural Network (ANN) implemented in `ANN_version.ipynb`. The `dataset.ipynb` notebook prepares the ECG data for classification. The project processes ECG signals, extracts segments around R-peaks, and trains an ANN to classify heartbeats into 15 valid types. The project is actively under development, and the GitHub repository will be updated over time. Refer to the `README.md` file in the repository for the latest project details and updates.

## Dataset
- **Source**: MIT-BIH Arrhythmia Database (accessed via `wfdb` from PhysioNet).
- **Records**: 48 ECG records (e.g., '100', '101', ..., '234').
- **Leads**: 12 standard ECG leads (I, II, III, aVF, aVR, aVL, V1–V6). Limb leads (I, II, III, aVR, aVL, aVF) and precordial leads (V1–V6) capture cardiac activation in 3D space.
- **Arrhythmia**: Irregular heart rhythm, potentially causing insufficient blood flow, leading to stroke, heart failure, or cardiac arrest.
- **Heartbeat Types**: 15 valid types (`A`: Atrial premature beat, `E`: Ventricular escape beat, `F`: Fusion beat, `J`: Junctional premature beat, `L`: Left bundle branch block beat, `N`: Normal sinus beat, `Q`: Unclassifiable beat, `R`: Right bundle branch block beat, `S`: Supraventricular ectopic beat, `V`: Ventricular ectopic beat, `a`: Aberrated atrial premature beat, `e`: Atrial escape beat, `f`: Fusion of ventricular and normal beat, `j`: Junctional escape beat, `x`: Non-conducted P-wave).
- **Data**:
  - Input: ECG signal segments (360 samples, 0.5 seconds at 360 Hz; or 300 samples for record '100') centered on R-peaks.
  - Labels: 15 valid heartbeat types.
  - Output shape: Full dataset (`X`: 112,544 × 360, `y`: 112,544); Record '100' (`X`: 2,271 × 300, `y`: 2,271).
- **Preprocessing**:
  - Loads ECG signals and annotations using `wfdb`.
  - Extracts first channel (`p_signal[:, 0]`).
  - Segments signals into windows around R-peaks.
  - Filters segments within signal bounds and optionally for valid labels.
  - Converts data to NumPy arrays or Pandas DataFrames.

## Methodology
### dataset.ipynb
1. **Data Loading**:
   - Loads 48 records and annotations from PhysioNet (`mitdb`).
   - Extracts first channel of ECG signal.
   - Segments into 360-sample windows (300 for record '100') centered on R-peaks.
2. **Visualization**:
   - Plots label distribution using `seaborn.countplot`, revealing class imbalance (e.g., `N`: 75,011, `S`: 2).
3. **Data Storage**:
   - Converts data for record '100' to Pandas DataFrames (`X_df`, `y_df`).
4. **Notes**:
   - 23 unique labels found, including non-valid types (e.g., `!`, `"`, `+`, `/`).
   - Commented code for filtering valid types.
   - Focuses on data preparation, not model training.

### ANN_version.ipynb
1. **Data Loading**:
   - Similar to `dataset.ipynb`, processes all records with 360-sample windows.
2. **Model Training**:
   - ANN architecture (TensorFlow/Keras):
     - Input layer: 360 features.
     - Hidden layers: Not specified (assumed dense).
     - Output layer: 15 classes (softmax).
   - Training:
     - Epochs: 50.
     - Batch size: Not specified (default assumed).
     - Optimizer: Not specified (e.g., Adam).
     - Loss: Categorical cross-entropy (inferred).
     - Metrics: Accuracy.
     - Data split: 80% training, 20% validation.
     - Training time: ~14-16 seconds/epoch on NVIDIA GeForce RTX 3060.
3. **Evaluation**:
   - Test accuracy: 99.34%.
   - Test loss: 0.0232.
   - Best validation accuracy: 99.51% (Epoch 47).
   - Best validation loss: 0.0191 (Epoch 47).
4. **Model Saving**: Not implemented.

## Requirements
- Python 3.x
- Libraries: `numpy`, `wfdb`, `matplotlib`, `seaborn`, `tensorflow`, `pandas`.
- Hardware: CUDA-enabled GPU (e.g., NVIDIA RTX 3060) for ANN training.
- Install:
  ```bash
  pip install numpy wfdb matplotlib seaborn tensorflow pandas
  ```
  For GPU support:
  ```bash
  pip install tensorflow[and-cuda]
  ```

## Installation
1. Clone repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies (see above).
3. Ensure CUDA/cuDNN for GPU acceleration.

## Usage
1. Run `dataset.ipynb` to prepare data:
   ```bash
   jupyter notebook dataset.ipynb
   ```
   - Loads and segments ECG data.
   - Visualizes label distribution.
   - Prepares DataFrames for record '100'.
2. Run `ANN_version.ipynb` to train model:
   ```bash
   jupyter notebook ANN_version.ipynb
   ```
   - Loads and preprocesses data.
   - Trains and evaluates ANN.

## Results
- **Test Performance**:
  - Accuracy: 99.34%.
  - Loss: 0.0232.
- **Training Performance**:
  - Best validation accuracy: 99.51% (Epoch 47).
  - Best validation loss: 0.0191 (Epoch 47).
- **Training Time**: ~14-16 seconds/epoch (50 epochs).

## Notes
- **Ongoing Development**: The project is actively being developed, with updates to the GitHub repository. Check `README.md` for the latest information.
- **Class Imbalance**: Significant imbalance (e.g., `N` dominates with 75,011 samples).
- **Commented Code**: Valid types filtering is commented out in `dataset.ipynb`.
- **GPU Usage**: NUMA node warnings from TensorFlow CUDA setup; no performance impact.
- **Model Saving**: Add `model.save('model.h5')` for persistence.
- **Single Record**: `dataset.ipynb` processes record '100' separately with 300-sample windows.

## Future Improvements
- Filter non-valid labels in `dataset.ipynb`.
- Explicitly define data splitting (e.g., `train_test_split`).
- Specify ANN architecture (layers, units, activations).
- Implement model saving.
- Use cross-validation for robustness.
- Tune hyperparameters (learning rate, batch size).
- Address class imbalance with SMOTE or class weights.
- Explore CNN or LSTM models for ECG classification.

## License
MIT License.

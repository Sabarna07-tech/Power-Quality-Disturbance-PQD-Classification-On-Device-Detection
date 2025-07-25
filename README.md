# Power Quality Disturbance Classification

This repository contains code and resources for a hybrid machine‑learning pipeline to detect and classify 17 types of power quality disturbances using one cycle (999 samples at 5 kHz) of voltage waveform data.

## Description

The project implements a two‑stage hybrid approach:

1. **Stage 1 – XPQRS-derived Features & SVM**

   * Computes up to four discrete derivatives of the raw waveform to amplify deviations.
   * Extracts simple energy and mobility features.
   * Classifies via a Quadratic SVM with fast decision time (\~35 ms).

2. **Stage 2 – 1‑D Residual TCN**

   * Processes the raw waveform with a stack of dilated 1‑D residual blocks.
   * Applies global average pooling and a dense head to refine low‑confidence cases.
   * Achieves > 99 % classification accuracy on held‑out test data.

The notebook also includes:

* Data loading and preprocessing for the XPQRS dataset.
* Model training, validation, and performance evaluation.
* Export of the deep model to ONNX (and optional TFLite/Flex) for edge deployment.

## Repository Structure

```
/               # root
├── data/       # scripts to download or sync XPQRS dataset
├── notebooks/  # Jupyter notebooks with EDA, training, conversion
│   └── power-quality-disturbance-pqd-classification.ipynb
├── src/        # Python modules
│   ├── features.py        # derivative & CWT feature extractors
│   ├── models.py          # Stage1 SVM & Stage2 TCN definitions
│   ├── train.py           # scripts to train SVM and deep model
│   ├── inference.py       # hybrid inference pipeline
│   └── convert.py         # ONNX/TFLite export utilities
├── stage2_saved_model/    # exported TensorFlow SavedModel
├── stage2.onnx            # exported ONNX model
├── stage2_flex.tflite     # optional TFLite with Flex ops
├── README.md              # this file
└── requirements.txt       # Python dependencies
```

## Installation

```bash
git clone https://github.com/Sabarna07-tech/Power-Quality-Disturbance-PQD-Classification-On-Device-Detection.git
cd power-quality-pqd
pip install -r requirements.txt
```

## Usage

1. **Data preparation**:

   ```bash
   python src/features.py --download-data
   ```

2. **Exploratory analysis & training**:
   Launch `notebooks/power-quality-disturbance-pqd-classification.ipynb` in Jupyter Lab.

3. **Command‑line training**:

   ```bash
   python src/train.py --stage1 --stage2
   ```

4. **Hybrid inference demo**:

   ```bash
   python src/inference.py --input sample.csv --mode hybrid
   ```

5. **Model export**:

   ```bash
   python src/convert.py --to onnx
   python src/convert.py --to tflite --flex
   ```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or contributions, please open an issue or contact the maintainer at [your.email@example.com](mailto:your.email@example.com).

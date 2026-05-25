Automated detection and grading of Diabetic Retinopathy across 5 severity classes using Transfer Learning on the APTOS 2019 Blindness Detection dataset. Comparative study of VGG16, ResNet50, EfficientNetB0, Xception and InceptionV3.


📌 Table of Contents

Overview
Dataset
Models
Results
Project Structure
How to Run
Visualizations
Limitations
Future Work
Author


🧠 Overview
Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. This project builds an automated deep learning pipeline to classify DR severity from retinal fundus images into 5 grades:
GradeLabelDescription0No DRHealthy retina1MildMicroaneurysms only2ModerateMore widespread vascular changes3SevereExtensive retinal abnormalities4Proliferative DRNeovascularization — most severe
Five pretrained CNN architectures were evaluated under identical experimental conditions to find the best performing model for this task.

📂 Dataset
APTOS 2019 Blindness Detection — https://www.kaggle.com/competitions/aptos2019-blindness-detection/code
AttributeDetailTotal Images3,662Image FormatPNG (RGB)Classes5 (DR Grades 0–4)Resized To224 × 224 pxTrain Split80% (~2,931 images)Validation Split20% (~731 images)Missing ValuesNone
Class Distribution
GradeLabelCountPercentage0No DR1,80549.3%1Mild37010.1%2Moderate99927.3%3Severe1935.3%4Proliferative DR2958.1%

⚠️ Dataset is significantly imbalanced — Class 0 dominates at 49.3%


🏗️ Models
All five models were loaded with ImageNet pretrained weights with frozen backbone layers. A unified custom classification head was attached to each:
GlobalAveragePooling2D
→ BatchNormalization
→ Dense(256, ReLU) + Dropout(0.4)
→ Dense(128, ReLU) + Dropout(0.3)
→ Dense(5, Softmax)
ModelParametersDepthKey InnovationVGG16138M16Stacked 3×3 convolutionsResNet5025.6M50Residual skip connectionsEfficientNetB05.3MVariableCompound scaling + SE blocksXception22.9M71Depthwise separable convolutionsInceptionV323.8M159Multi-scale Inception modules
Training Configuration
ParameterValueOptimizerAdam (lr = 1e-4)LossSparse Categorical Cross-EntropyMax Epochs5Batch Size32Early Stoppingpatience = 3 (val_accuracy)LR Reductionfactor = 0.5, patience = 2

📊 Results
Final Model Comparison
ModelAccuracyPrecisionRecallF1-ScoreROC-AUCVGG160.73460.68490.73460.67230.8555ResNet500.77700.75420.77700.75500.9090EfficientNetB00.76740.76470.76740.73140.8892Xception0.71960.71620.71960.66120.8950InceptionV30.72090.68110.72090.66820.8746
🏆 Best Model Per Metric
MetricBest ModelScoreAccuracyResNet500.7770PrecisionEfficientNetB00.7647RecallResNet500.7770F1-ScoreResNet500.7550ROC-AUCResNet500.9090

ResNet50 emerged as the best overall model, winning 4 out of 5 metrics. Its residual skip connections enable stable gradient flow and strong generalization on small datasets.


📁 Project Structure
diabetic-retinopathy-detection/
│
├── The_Final_of_Prl_.ipynb       # Main notebook (full pipeline)
├── README.md                      # This file
│
├── /content/aptos_data/           # Organized class folders (auto-generated)
│   ├── 0/                         # No DR images
│   ├── 1/                         # Mild DR images
│   ├── 2/                         # Moderate DR images
│   ├── 3/                         # Severe DR images
│   └── 4/                         # Proliferative DR images
│
└── /content/saved_models/         # Saved .keras model files
    ├── VGG16_final.keras
    ├── ResNet50_final.keras
    ├── EfficientNetB0_final.keras
    ├── Xception_final.keras
    └── InceptionV3_final.keras

    🚀 How to Run
1. Clone the repository
bashgit clone https://github.com/JoydipRupesh/Retinal_Disease_Classification.git
cd diabetic-retinopathy-detection
2. Download the dataset
Download the APTOS 2019 dataset from Kaggle and place the zip file in your Google Drive.
3. Open in Google Colab
Upload The_Final_of_Prl_.ipynb to Google Colab and run all cells sequentially.
4. Requirements
tensorflow >= 2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow

💡 Recommended: Use Google Colab with T4 GPU for faster training.


📈 Visualizations
The notebook generates the following visualizations:

Class distribution bar chart and pie chart
Sample retinal fundus images (2 per class)
Augmentation preview (9 variants)
Learning curves (accuracy and loss) for all 5 models
Confusion matrices (raw + normalized) for all 5 models
ROC-AUC curves (one-vs-rest) for all 5 models
Grouped bar chart comparing all models across all metrics
Radar chart for holistic performance comparison


⚠️ Limitations

Class imbalance was not explicitly handled (no class weighting or oversampling)
Only 5 training epochs — deeper models like InceptionV3 may need more
No fine-tuning of pretrained backbone layers was performed
Evaluation on validation split only — no test set inference
No interpretability visualization (Grad-CAM) included


🔮 Future Work

Add class-weighted cross-entropy or Focal Loss to handle imbalance
Implement two-phase training with backbone fine-tuning
Extend training to 20+ epochs with cosine annealing LR schedule
Add Grad-CAM heatmaps for clinical interpretability
Ensemble ResNet50 and EfficientNetB0 predictions
Deploy best model as a web app using Streamlit or Flask


👤 Author
Rupps

GitHub: JoydipRupesh
Dataset: Retinal_Disease_Classification


📜 License
This project is licensed under the MIT License — feel free to use and modify for educational purposes.

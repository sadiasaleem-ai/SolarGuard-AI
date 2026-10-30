# SolarGuard AI: Solar Panel Condition Classifier

SolarGuard AI is our 5th semester integrated project for Machine Learning, Deep Learning and Computer Vision. It looks at a photo of a solar panel and tells us what condition it is in: clean, dusty, covered in snow, bird droppings, electrical damage or physical damage.

Solar power is growing fast in Pakistan, and checking panels by hand on a roof is slow and risky. Our goal is to make this check automatic.

## Team

| Member | Area |
|--------|------|
| Noor   | Computer Vision |
| Sonia  | Machine Learning |
| Sadia  | Deep Learning |

## Links

- Colab notebook (Milestone 3, final comparison): https://colab.research.google.com/drive/1vdssafsEMqi8BQlHwlTPkwEFQCMMvCcR?usp=sharing

## Dataset

We use the [Solar Panel Images dataset on Kaggle](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images): 869 images in 6 classes. The dataset is not uploaded here because it is large. Please download it from the link above.

| Class | Images |
|-------|--------|
| Bird-drop | 191 |
| Clean | 193 |
| Dusty | 190 |
| Electrical-damage | 103 |
| Physical-Damage | 69 |
| Snow-Covered | 123 |

The classes are not balanced, so we also look at precision, recall and macro-F1 and use class weights.

## Milestone 2: dataset, preprocessing and first models

- Explored the data, resized all images to 224 x 224 (RGB) and made a stratified 70/15/15 split.
- Machine learning: HOG + HSV colour histogram features (2,276 per image) with Random Forest, SVM and KNN.
- Deep learning: a CNN built from scratch (3 convolution blocks, dropout, flip and rotation augmentation).
- Best result: Random Forest, 0.692 validation accuracy. CNN from scratch: 0.631.

## Milestone 3: model improvement and working prototype

What changed from Milestone 2, and why:

| Area | Milestone 2 | Milestone 3 | Why |
|------|-------------|-------------|-----|
| Split | 70/15/15 | 70/20/10 (608 / 174 / 87) | Instructor feedback; larger validation set for choosing models |
| Classical ML | Random Forest, SVM, KNN | Random Forest tuned (600 trees, balanced subsample) | Hand-made features cap these models near 0.69 |
| CNN from scratch | One setup, 20 epochs | Dropout on/off, 3/4/5 blocks, 20 and 100 epochs | Best 0.713 test, but it overfits (train 0.99, val 0.65); 869 images are too few |
| Transfer learning | Not used | MobileNetV2, EfficientNetB0, ResNet50 (frozen body, new top layer), early stopping, class weights | Pretrained networks already know edges and textures |
| Selection | Compared on validation | Ranked on validation, test only reported | Avoids choosing a model on the test set |
| Prototype | `predict_image(path)` | Gradio web app with suggested action | Easy to try for any user |

### Results

| Model | Val acc | Test acc | Test macro-F1 |
|-------|---------|----------|---------------|
| **Ensemble of 3 CNNs (final model)** | 0.822 | 0.724 | 0.757 |
| Ensemble of 3 CNNs + Random Forest | 0.816 | 0.713 | 0.732 |
| ResNet50 | 0.805 | 0.690 | 0.712 |
| MobileNetV2 | 0.759 | 0.678 | 0.698 |
| Random Forest (tuned) | 0.690 | 0.724 | 0.720 |
| EfficientNetB0 | 0.494 | 0.460 | 0.476 |

The final model averages the probabilities of the three pretrained networks. Validation accuracy passed our 0.80 target (0.822), but test accuracy is 0.724. The test set has only 87 images, so the results are noisy.

## Plan for Milestone 4

- Fix EfficientNetB0 (early stopping on validation accuracy) and re-test the ensemble
- Stronger augmentation and more images for the small classes
- Cross-validation for a less noisy estimate
- Final report, final presentation, working demo and viva

## Tools used

Python, OpenCV, NumPy, Pandas, scikit-learn, scikit-image, TensorFlow/Keras, Matplotlib, Seaborn, Gradio, Google Colab (T4 GPU)

## How to run

1. Download the dataset from Kaggle and upload the zip to your Google Drive.
2. Open `milestone3_final_comparison.ipynb` in Google Colab (or use the Colab link above).
3. Set the runtime to **T4 GPU** (Runtime > Change runtime type).
4. Change `ZIP_PATH` in the first cells to the path of your zip file.
5. Run all cells from top to bottom. The last cells start the Gradio app (its share link is temporary).

## Files in this repository

- `milestone2_solarguard_ai.ipynb`: Milestone 2 code (data, preprocessing, ML baselines, CNN from scratch)
- `milestone2_experiments.ipynb`: CNN experiments (dropout, 3/4/5 blocks, 20 and 100 epochs)
- `milestone3_final_comparison.ipynb`: Milestone 3 code (tuned Random Forest, transfer learning, ensemble, Gradio app)
- `README.md`: this file

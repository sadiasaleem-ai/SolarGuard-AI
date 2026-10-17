# SolarGuard AI: Solar Panel Condition Classifier

SolarGuard AI is our 5th semester integrated project for Machine Learning, Deep Learning and Computer Vision. It looks at a photo of a solar panel and tells us what condition it is in: clean, dusty, covered in snow, bird droppings, or damaged.

Solar power is growing fast in Pakistan, and checking panels by hand on a roof is slow and risky. Our goal is to make this check automatic.

## Team

| Member | Area | Work in Milestone 2 |
|---|---|---|
| Sadia | Computer Vision | Data exploration, image preprocessing, train/validation/test split |
| Sonia | Machine Learning | HOG and color features, Random Forest, SVM, KNN, evaluation |
| Noor | Deep Learning | CNN built from scratch, training, learning curves |

## Dataset

We used the [Solar Panel Images dataset on Kaggle](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images). It has 869 images in 6 classes.

| Class | Images |
|---|---|
| Bird-drop | 191 |
| Clean | 193 |
| Dusty | 190 |
| Electrical-damage | 103 |
| Physical-Damage | 69 |
| Snow-Covered | 123 |

The dataset is not uploaded here because it is large. Please download it from the link above.

Some classes (like Physical-Damage) have far fewer images than others, so we kept this in mind while training and judging the models.

## What we did in Milestone 2

**1. Exploring the data**
We counted the images in each class, plotted the class distribution, and looked at sample images. We also found 135 different image sizes, so every image needs to be resized.

**2. Preprocessing**
All images were converted to RGB and resized to 224 x 224. We then split the data into 70% training (608 images), 15% validation (130 images) and 15% test (131 images). The split keeps the same class proportions in every part.

**3. Machine Learning models**
For every image we extracted HOG features (shape and edges) and an HSV color histogram, giving 2,276 features per image. We trained Random Forest, SVM and KNN on these features. Random Forest and SVM use balanced class weights to handle the uneven classes.

**4. Deep Learning model**
We built a CNN from scratch with three convolution and pooling blocks, dropout, and a softmax output. It also has random flip and rotation layers for basic augmentation. It was trained for 20 epochs on a Colab GPU.

**5. Working pipeline**
The function `predict_image(path)` takes an image path, preprocesses the image, and returns the predicted class with a confidence score. On a test image of a clean panel, it predicted "Clean" with 96.5% confidence.

## Results (validation set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.692 | 0.695 |
| SVM | 0.585 | 0.595 |
| KNN | 0.338 | 0.284 |
| CNN (from scratch) | 0.631 | 0.582 |

Random Forest is our best model so far. The CNN does well on Snow-Covered panels but finds Electrical-damage and Physical-Damage harder, probably because there are fewer images of them.

## Problems we faced

- Images came in many different sizes and had to be resized to one size.
- The classes are not balanced, so accuracy alone can be misleading. We also looked at precision, recall and F1 for each class.
- The CNN trained from scratch on a small dataset does not reach very high accuracy.

## Plan for Milestone 3

- Transfer learning with a pretrained model (VGG16 or MobileNet)
- Better data augmentation and hyperparameter tuning
- Better handling of the small classes
- Final evaluation on the test set
- A Gradio web app where a user uploads a panel photo and gets the condition and a suggested action

## Tools used

Python, OpenCV, NumPy, Pandas, scikit-learn, scikit-image, TensorFlow/Keras, Matplotlib, Seaborn, Google Colab

## How to run

1. Download the dataset from Kaggle and keep it in a folder with one sub-folder per class.
2. Open `milestone2_solarguard_ai.ipynb` in Google Colab.
3. Change `ZIP_PATH` and `DATA_DIR` in the first cells to match where your dataset is saved.
4. Run all cells from top to bottom.

## Files in this repository

- `milestone2_solarguard_ai.ipynb`: the full Milestone 2 code
- `README.md`: this file

# Gender-Prediction-Using-Deep-Learning
# Gender Prediction Using Deep Learning

## Introduction
This project implements a deep learning-based gender prediction system that can classify male and female faces in real-time using a webcam feed. The model utilizes **WideResNet CNN architecture**, trained on labeled datasets, to perform accurate gender classification. The system has applications in **security, marketing, human-computer interaction, and real-time analytics**.

## Features
- **Deep Learning Model:** Utilizes a WideResNet convolutional neural network (CNN) for facial classification.
- **Real-Time Gender Prediction:** Live video feed processing for immediate classification.
- **Robust Training:** Dataset includes images with diverse lighting, angles, and occlusions.
- **Graphical User Interface (GUI):** Built using Tkinter for user-friendly interaction.
- **Performance Optimization:** Includes data augmentation techniques to enhance model generalization.

## Technologies Used
- **Python**
- **OpenCV** (for image processing & webcam feed handling)
- **TensorFlow & Keras** (for deep learning model implementation)
- **Tkinter** (for GUI interface)
- **Matplotlib & NumPy** (for visualizing training performance)

## Project Structure
```
Gender_Prediction/
│── dataset/                # Contains the gender-labeled face images
│   ├── man/                # Folder for male images
│   ├── woman/              # Folder for female images
│── models/                 # Pretrained and trained models
│   ├── gender_model.h5     # Trained deep learning model
│── src/                    # Source code for the project
│   ├── train_model.py      # Script for training the CNN model
│   ├── predict_gender.py   # Script for gender classification from images
│   ├── gui.py              # GUI interface script for real-time webcam feed
│── requirements.txt        # Python dependencies for running the project
│── README.md               # Documentation for the project
│── config.py               # Configuration file for model parameters
│── results/                # Stores evaluation results (confusion matrix, accuracy reports)
│   ├── test_results.txt    # Performance metrics of the trained model
│   ├── model_performance.png # Graphs of training accuracy/loss
│── LICENSE                 # License file (MIT, Apache, etc.)
```

## Installation & Setup
### 1. Clone the repository
```sh
git clone https://github.com/yourusername/Gender_Prediction.git
cd Gender_Prediction
```

### 2. Install dependencies
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### 3. Train the model (optional)
If you want to train from scratch:
```sh
python src/train_model.py
```

### 4. Run gender prediction
To predict gender from images:
```sh
python src/predict_gender.py --image path/to/image.jpg
```

To run the **real-time webcam GUI**:
```sh
python src/gui.py
```

## Model Training Details
- **Dataset:** The model was trained on the **CelebA** and **FairFace** datasets, containing over **200K labeled images**.
- **Preprocessing:** Images resized to **96x96 RGB**, normalized pixel values, and applied augmentation techniques.
- **Model Architecture:** Uses **WideResNet** with batch normalization, dropout, and softmax activation.
- **Optimizer:** **Adam**, binary cross-entropy loss function.
- **Training Metrics:** Achieved **96.2% validation accuracy** on diverse test datasets.

## Results & Performance
- **Accuracy:** 96.4% on independent test set.
- **Inference Speed:** 42 milliseconds per frame (real-time processing capability).
- **ROC Curve:** AUC score of **0.97**, indicating strong predictive ability.

## Future Improvements
- Enhance training with more real-world images.
- Optimize model to work efficiently on mobile devices.
- Improve robustness to occlusions (e.g., glasses, masks).

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.

## Acknowledgments
- CelebA and FairFace datasets for labeled facial images.
- Open-source deep learning frameworks (TensorFlow/Keras).

---
This repository provides an effective deep learning pipeline for **real-time gender classification**, integrating computer vision and AI techniques. Contributions and improvements are welcome!



# Skin Lesion Segmentation using U-Net

## Project Overview
This project focuses on automatic skin lesion segmentation using deep learning. The objective is to accurately identify lesion regions in dermoscopic images using a U-Net-based architecture.

The ISIC 2018 dataset is used to build a complete pipeline including data preprocessing, model training, evaluation, and visualization.

---

## Dataset
- Dataset: ISIC 2018 Skin Lesion Segmentation  
- Input: Dermoscopic images  
- Output: Binary masks (lesion vs background)

The dataset is split into:
- 80% Training  
- 20% Validation  

---

## Methodology

### Data Preprocessing
- Images resized to 256 × 256  
- Normalization applied  
- Data augmentation (training only):
  - Horizontal flip  
  - Rotation  

---

### Model Architecture
A U-Net model is implemented with:
- Encoder-decoder structure  
- Skip connections for spatial information preservation  
- Binary segmentation output  

---

### Loss Functions
The following loss functions were evaluated:
- Dice Loss  
- Binary Cross Entropy (BCE) Loss  

---

### Training
- Optimizer: Adam  
- Learning rate: 1e-3  
- Epochs: 3  
- Batch size: 8  

---

### Evaluation Metrics
- Dice Score  
- Intersection over Union (IoU)  

---

## Results

### Epoch-wise Performance (Dice Loss)

| Epoch | Dice | IoU |
|------|------|-----|
| 1 | 0.69 | 0.53 |
| 2 | 0.76 | 0.63 |
| 3 | 0.78 | 0.64 |

The results show consistent improvement as training progresses.

---

### Loss Function Comparison

| Loss Function | Dice | IoU |
|--------------|------|-----|
| Dice Loss    | 0.78 | 0.64 |
| BCE Loss     | 0.72 | 0.58 |

Dice Loss achieves better performance due to its ability to directly optimize overlap and handle class imbalance.

---

## Visualization

Predictions were compared across epochs:

Input Image | Ground Truth | Epoch 1 Prediction | Epoch 3 Prediction

Observations:
- Predictions improve with training  
- Boundaries become smoother and more accurate  
- Better alignment with ground truth masks  

---

## Project Structure
src/
├── dataset.py
├── transforms.py
├── models/unet.py
├── losses.py
├── train.py
├── evaluate.py

outputs/
├── checkpoints/
├── figures/

main.py
evaluate_checkpoints.py
compare_predictions.py


---

## How to Run

### Train the model
python main.py


### Evaluate checkpoints
python evaluate_checkpoints.py


### Visualize predictions
python compare_predictions.py


---

## Conclusion

This project demonstrates a complete deep learning pipeline for skin lesion segmentation. The U-Net model shows strong performance, and results indicate that Dice Loss is more suitable than BCE Loss for this task. Performance improves with additional training epochs.

---




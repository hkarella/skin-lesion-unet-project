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

An Attention U-Net model is implemented with:
- Encoder-decoder architecture  
- Skip connections integrated with attention gates  
- Attention mechanism to suppress irrelevant regions and highlight salient features  
- Enhanced feature propagation between encoder and decoder  
- Binary segmentation output 
---

### Loss Functions
The following loss functions were evaluated:
- Dice Loss  
- Binary Cross Entropy (BCE) Loss  

---

### Training
- Model: U-Net  
- Optimizer: Adam  
- Learning rate: 1e-3  
- Epochs: 3  
- Batch size: 8
  
- Model: Attention U-Net  
- Optimizer: Adam for efficient gradient optimization  
- Learning rate: 1e-3  
- Epochs: 10  
- Batch size: 8  

---

### Evaluation Metrics
- Dice Score  
- Intersection over Union (IoU)  

---

## Results
### Unet

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

### Attention Unet
## Epoch-wise Training Loss

| Epoch | Loss |
|------|------|
| 0 | 1.1061 |
| 1 | 0.8704 |
| 2 | 0.6584 |
| 3 | 0.5637 |
| 4 | 0.5327 |
| 5 | 0.5048 |
| 6 | 0.4734 |
| 7 | 0.4637 |
| 8 | 0.4551 |
| 9 | 0.4404 |

The training loss shows a consistent decrease across epochs, indicating stable learning and effective convergence of the model.

---

## Validation Performance

| Metric | Score |
|--------|------|
| Dice Score | 0.8268 |
| IoU Score  | 0.7094 |

The model demonstrates strong segmentation performance on the validation set, with good overlap between predicted masks and ground truth.

---

## Observations

For U-net
- Steady loss reduction confirms stable training  
- Dice score above 0.82 indicates strong segmentation quality  
- IoU above 0.70 shows reliable region overlap  
- Model generalizes well to unseen data

For Attention U-net
- The training loss shows a steady decrease from 1.1061 to 0.4404, indicating stable learning and effective convergence of the Attention U-Net model  
- The use of attention gates improves feature selection by focusing on relevant regions and suppressing irrelevant background information  
- A Dice score of 0.8268 reflects strong segmentation performance with accurate boundary prediction  
- An IoU score of 0.7094 indicates reliable overlap between predicted masks and ground truth  
- The model demonstrates good generalization capability on unseen validation data  
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


### How to run the Attention U-net

You can find the tranined model code for you to run and train it (attentionUnet_training.ipynb)

If you wish to run the model without training, you can download the pretrained Attention U-Net model here:
[Download Model](https://drive.google.com/file/d/1m31Y2i9oyqGIB2E8ajIoP3cnxf4oP0XC/view?usp=sharing) (best_model.pth)

Combine best_model.pth and attentionUnet.py file (which acts as main.py for this attentionUnet model) add in your sample
skin lesion image and run it. 

Sample output for the Attention U-net is presented in the output.png file.

---

## Conclusion

This project demonstrates a complete deep learning pipeline for skin lesion segmentation. The U-Net model shows strong performance, and results indicate that Dice Loss is more suitable than BCE Loss for this task. Performance improves with additional training epochs.

---




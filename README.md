# Signal-Object-Detection

Machine learning project developed for a Kaggle competition focused on detecting the number of objects present in noisy radio signal images.

The task is formulated as a **5-class image classification problem**, where each image belongs to a class representing the number of detected objects (1–5).

## Competition

The dataset contains:

- **15,500 training images**
- **5,500 test images**
- PNG images representing noisy radio signals
- 5 target classes

The competition is evaluated using **classification accuracy**. The final submission contains the image `id` and the predicted `label`.

## Approach

Several machine learning approaches were explored, starting with classical models and gradually moving towards convolutional neural networks.

### Models Tested

1. **Logistic Regression**
   - Grayscale images converted to feature vectors
   - StandardScaler
   - Used as a baseline

2. **KNN + PCA**
   - Grayscale pixel features
   - PCA with 100 components
   - `n_neighbors = 7`
   - Distance-based weighting
   - Validation accuracy: **21.51%**

3. **Random Forest**
   - Grayscale pixel features
   - Multiple combinations of `n_estimators` and `max_depth`
   - Best validation accuracy: **27.06%**

4. **Extra Trees + Handcrafted Features**
   - Image statistics
   - Intensity histograms
   - Row and column projections
   - Best validation accuracy: **48.00%**

5. **MLP + Handcrafted Features**
   - Combined grayscale, RGB and projection-based features
   - StandardScaler
   - Fully-connected neural network

6. **Hybrid CNN + Handcrafted Features**
   - CNN branch for image features
   - MLP branch for handcrafted features
   - Feature fusion before classification

7. **CNN + XGBoost**
   - CNN-generated embeddings
   - CNN probabilities
   - XGBoost trained on learned representations
   - Best validation accuracy: **65.40%**

8. **Improved CNN + Handcrafted Features + XGBoost**
   - Higher-resolution input
   - Improved CNN architecture
   - Auxiliary count prediction head
   - Best validation accuracy: **70.57%**

9. **Final Custom RGB CNN**
   - Residual convolutional blocks
   - Squeeze-and-Excitation blocks
   - DropPath and Dropout regularization
   - Auxiliary count prediction
   - EMA model
   - Fine-tuning
   - Test-Time Augmentation (TTA)

## Final Model

The final model processes RGB images resized to **192 × 84**.

The architecture contains:

- Convolutional stem
- Residual convolutional blocks
- Squeeze-and-Excitation blocks
- DropPath regularization
- Global average pooling
- Fully-connected layers
- Classification head for the 5 classes
- Auxiliary count prediction head

The training objective combines the main classification loss with two auxiliary losses:

```text
Loss = CrossEntropyLoss
       + 0.10 × CountLoss
       + 0.06 × ExpectedClassLoss

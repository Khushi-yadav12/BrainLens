# ACU-Net Model Flowcharts

This document outlines the detailed working process of the Attention-based Convolutional U-Net (ACU-Net) implemented for brain tumor segmentation.

## 1. High-Level System Workflow
This flowchart illustrates the end-to-end process of the application, from receiving a raw MRI scan to outputting the final tumor overlay.

```mermaid
graph TD
    A["Raw MRI Scan Input"] --> B["Pre-processing (Resize, Normalize)"]
    B --> C["ACU-Net Segmentation Model"]
    C --> D["Output Probability Map"]
    D --> E["Post-processing (Thresholding)"]
    E --> F["Final Tumor Mask / Overlay"]
```

## 2. ACU-Net Architecture Overview
The macro architecture of the ACU-Net. It follows a 'U' shape consisting of a contracting path (encoder) to capture context, and a symmetric expanding path (decoder) that enables precise localization using Attention Gates.

```mermaid
graph TD
    In["Input Image"] --> E1["Encoder Block 1"]
    E1 -->|Skip Connection| AG2["Attention Gate 2"]
    E1 --> P1["MaxPool (Downsample)"]
    
    P1 --> E2["Encoder Block 2"]
    E2 -->|Skip Connection| AG1["Attention Gate 1"]
    E2 --> P2["MaxPool (Downsample)"]
    
    P2 --> B["Bottleneck (Deepest Features)"]

    B --> U1["UpSample"]
    U1 --> C1["Concatenation"]
    
    AG1 -->|Filtered Features| C1
    U1 -->|Gating Signal| AG1
    
    C1 --> D1["Decoder Block 1"]

    D1 --> U2["UpSample"]
    U2 --> C2["Concatenation"]
    
    AG2 -->|Filtered Features| C2
    U2 -->|Gating Signal| AG2
    
    C2 --> D2["Decoder Block 2"]

    D2 --> Out["1x1 Conv (Sigmoid Activation)"]
    Out --> Mask["Predicted Tumor Mask"]
```

## 3. Attention Gate Mechanism
The attention mechanism allows the model to focus on target structures of varying shapes and sizes. It filters out irrelevant background features from the encoder's skip connections using contextual information from the deeper decoder layers (the gating signal).

```mermaid
graph LR
    G["Gating Signal (from deeper layer)"] --> G_Conv["1x1 Conv"]
    X["Skip Feature (from encoder)"] --> X_Conv["1x1 Conv"]
    
    G_Conv --> Add["Element-wise Addition"]
    X_Conv --> Add
    
    Add --> Relu["ReLU Activation"]
    Relu --> Psi["1x1 Conv"]
    Psi --> Sigm["Sigmoid (Attention Coefficients)"]
    
    Sigm --> Mult["Element-wise Multiplication"]
    X --> Mult
    
    Mult --> Out["Attention Modulated Features (To Decoder)"]
```

## 4. Feature Extraction & Reconstruction Blocks
A detailed look inside the building blocks of the encoder and decoder.

```mermaid
graph TD
    subgraph "Encoder Block"
        E_In["Input Features"] --> E_C1["3x3 Convolution"]
        E_C1 --> E_BN1["Batch Normalization"]
        E_BN1 --> E_R1["ReLU"]
        E_R1 --> E_C2["3x3 Convolution"]
        E_C2 --> E_BN2["Batch Normalization"]
        E_BN2 --> E_R2["ReLU"]
        E_R2 --> E_Out["To MaxPool / Attention"]
    end

    subgraph "Decoder Block"
        D_In["Concatenated Features"] --> D_C1["3x3 Convolution"]
        D_C1 --> D_BN1["Batch Normalization"]
        D_BN1 --> D_R1["ReLU"]
        D_R1 --> D_C2["3x3 Convolution"]
        D_C2 --> D_BN2["Batch Normalization"]
        D_BN2 --> D_R2["ReLU"]
        D_R2 --> D_Out["To Upsample / Final Output"]
    end
```

## 5. Training Pipeline & Loss Calculation
How the ACU-Net model is trained using a specialized combined loss function to handle severe class imbalance common in medical imaging (tumors make up a small portion of the brain scan).

```mermaid
graph TD
    Input["MRI Image Batch"] --> Forward["ACU-Net Forward Pass"]
    Forward --> Pred["Predicted Mask Probabilities"]
    GT["Ground Truth Masks"] --> LossCalc

    Pred -.-> LossCalc

    subgraph "Combined Loss Function Calculation"
        LossCalc --> CE["Cross-Entropy Loss (Pixel level accuracy)"]
        LossCalc --> Dice["Dice Loss (Spatial overlap/boundary)"]
        CE --> AddLoss["Combined Total Loss"]
        Dice --> AddLoss
    end

    AddLoss --> Backprop["Backpropagation"]
    Backprop --> Optimizer["Adam Optimizer Step"]
    Optimizer --> Weights["Update Model Weights"]
```

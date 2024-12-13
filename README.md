# On-device Multimodal Emotion Recognition on Neural Processing Unit (NPU)
**Optimizing AI for low latency and power consumption in real-time applications.**

---

## 🔍 Project Summary

As part of a government agency project, I led the development of an **on-device multimodal emotion recognition system** on NPUs (Neural Processing Units). The project focused on optimizing real-time AI applications for **high emotion classification accuracy**, **low latency**, and **power efficiency**, addressing constraints typical of edge systems, such as limited model size and computational resources.

### Objectives
1. **Enhancing emotion recognition performance** by leveraging multimodal data sources, including:
   - Heart rate (HR)
   - EEG
   - Speech
   - Images

2. **Implementing a scalable real-time system** by embedding models on NPUs to reduce latency and power consumption.

---

## 🛠 Project Workflow
### Overall Architecture
![figure1](https://github.com/user-attachments/assets/e22babde-a2ad-42d1-bf5e-509ebed0e3f7)

### Detailed Structures of Emotion Recognition Models
![figure2](https://github.com/user-attachments/assets/ed881ac7-39db-447f-a180-429580abd3cd)

### 1. Model Design and Optimization
- **Simplified Architectures**: Developed deep learning models using architectures like CNNs and dense layers to balance performance and complexity.
- **Hyperparameter Tuning**: Conducted ablation studies to fine-tune parameters such as optimizer type, number of epochs, batch size, and loss functions.
- **Multimodal Fusion**: Adopted a **score-based fusion method** to combine outputs from multiple models at the decision level, avoiding additional neural network complexity.

### 2. NPU Deployment
- Converted models into **ONNX format** and compiled them using the **MXQ compiler** for compatibility with Mobilint’s NPU chips.
- Applied **quantization techniques** (Max, Percentile, and Max-Percentile) to compress models, optimizing based on an efficiency metric combining:
  - Matrix: Accuracy-increase ratio x Compression ratio


## 📊 Optimization Methods

### Multimodal Fusion and Simplified Models
- Built individual models for HR, EEG, speech, and image data.
- Focused on reducing model parameters while maintaining relative performance using simple architectures like CNN and dense layers.
- Used **score-based fusion** to integrate outputs without additional network complexity.

### Quantization Techniques
- Converted models into NPU-compatible formats via **ONNX** and the **MXQ compiler**.
- Applied three quantization methods to determine the best compression:
  - **MAX**: Clipping ranges based on minimum and maximum values.
  - **Percentile**: Clipping ranges based on top percentile values.
  - **Max-Percentile**: Clipping ranges based on the top percentile of maximum values.

---

## ⚙️ Evaluation Metrics
### Emotion Classification Accuracy
- Achieved an impressive **99.68% accuracy**, ensuring reliable and robust emotion recognition in real-time applications.

### Latency
- Compared model size before and after compression.
- Achieved **1.47x reduction** in model size.

### Power Consumption
- Measured power usage with an outlet power meter.
- Found **3.12x reduction** in power consumption for NPU-based models compared to GPU-based models.

## 📝 Key Findings

- The system achieved significant improvements in **efficiency and scalability**, making it suitable for real-time AI applications.
- Successfully implemented at the **Korean Institute of Science and Technology** as part of a government initiative.
- Findings were presented at an academic conference, and a related paper is currently under review.
- Reinforced my passion for developing **efficient, real-world AI systems**.

## 🤔 Insights on Clipping Range for Quantization
- **MAX**: Activations clipped using minimum and maximum values.
- **Percentile**: Activations clipped using the top percentile of values.
- **Max-Percentile**: Activations clipped using the top percentile of maximum values.

---

## 🌟 Conclusion

This project demonstrated the viability of deploying **real-time AI systems on edge devices** by optimizing multimodal emotion recognition models for **low latency** and **power efficiency**. It solidified my passion for creating **practical and scalable AI solutions** for real-world applications.

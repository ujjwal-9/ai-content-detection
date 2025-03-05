# ai-content-detection
Detect AI content using weak small language models


## 1. Introduction

The objective of this pipeline is to detect AI-generated content by integrating discriminative and generative models with handcrafted linguistic features. This multi-faceted approach leverages the strengths of different architectures to capture both semantic nuances and statistical properties of text. The final classifier is trained on a rich feature set that distinguishes between human-written and AI-generated content.

---

## 2. Model Selection

### DistilBERT

- **Why DistilBERT?**
  - **Efficiency:** A distilled, lighter version of BERT that provides robust contextual representations with reduced computational overhead.
  - **Discriminative Capability:** Pre-trained and fine-tuned on classification tasks (e.g., sentiment analysis), DistilBERT yields discriminative logits that capture subtle semantic and syntactic patterns.

- **Role in the Pipeline:**
  - **Feature Extraction:** DistilBERT is used to obtain output logits from the input text. These logits serve as high-level features that summarize the semantic content.

### GPT‑2

- **Why GPT‑2?**
  - **Generative Strength:** GPT‑2, as an autoregressive language model, excels in modeling language generation and provides useful probabilistic metrics.
  - **Language Modeling Metrics:** It computes features such as overall perplexity, layer-wise log perplexity, and burstiness (variability in token probabilities), which can signal differences between natural human text and machine-generated text.

- **Role in the Pipeline:**
  - **Perplexity Metrics:** Overall perplexity is computed as the exponential of the loss (or using the raw loss as log perplexity) to indicate how “surprised” the model is by the text.
  - **Layer-wise Analysis:** Hidden states from various transformer layers (first, second, second-last, and last) are analyzed to derive layer-specific log perplexity and burstiness features.
  - **DetectGPT-Inspired Feature:** By perturbing the text (e.g., random character substitutions) and observing changes in log probabilities, a robustness measure is derived, which can help flag AI-generated patterns.

### Combining the Two Models

- **Complementary Strengths:**
  - **Discriminative (DistilBERT):** Captures deep semantic features that help in differentiating subtle nuances in language.
  - **Generative (GPT‑2):** Provides statistical metrics that reflect the language model’s probability distribution, enabling the detection of unnatural or synthetic text patterns.

- **Overall Benefit:**
  - The fusion of features from DistilBERT and GPT‑2 yields a more comprehensive representation of the input text. This combined feature set improves classifier performance by leveraging both contextual representations and generative cues.

---

## 3. Feature Extraction

### Linguistic Features

- **Metrics Extracted:**
  - **Word Count, Sentence Count:** Capture basic text structure.
  - **Average Word Length, Punctuation Count:** Provide insights into stylistic elements.

- **Purpose:**  
  These basic linguistic features complement high-dimensional neural features, offering additional human-interpretable signals.

### GPT‑2 Specific Features

- **Overall Perplexity:**
  - **Definition:** Exponential of the average negative log-likelihood; reflects how “surprised” GPT‑2 is by the text.
  - **Rationale:** Unnatural or repetitive patterns in AI-generated text may lead to distinct perplexity scores.

- **Layer-wise Log Perplexity:**
  - **Implementation:** Uses hidden states from selected transformer layers to compute cross-entropy loss.
  - **Rationale:** Raw loss values (log perplexity) from early or late layers provide a granular measure of the model’s prediction uncertainty without resulting in huge exponential values.

- **Burstiness:**
  - **Definition:** Standard deviation of token log probabilities (post-softmax) for the ground truth tokens.
  - **Rationale:** Higher burstiness suggests more variability in token confidence, potentially indicating machine-generated text.

- **DetectGPT-Inspired Feature:**
  - **Method:** Slightly perturb the text (e.g., random character substitution) and measure the change in GPT‑2 log probability.
  - **Rationale:** A smaller change in probability after perturbation can signal a more uniform and synthetic text pattern.

---

## 4. Processing Pipeline and Scalability

### Multi-GPU Support

- **Motivation:**
  - Deep learning feature extraction is computationally intensive. Multi-GPU processing allows parallel processing of data batches, significantly reducing runtime.
  
- **Implementation:**
  - **Batch Processing:** The dataset is divided into batches that are processed concurrently.
  - **ThreadPoolExecutor:** Used for parallel execution with GPUs assigned in a round-robin fashion.
  - **Fallback:** When GPUs are unavailable, the system defaults to CPU processing.

### Data Preprocessing and Robustness

- **Text Truncation:**
  - Very long texts are truncated to avoid CUDA memory issues.
  
- **Error Handling:**
  - Samples that cause errors are skipped to ensure the overall pipeline remains robust and efficient.

---

## 5. Classification and Evaluation

### Classifier Choice

- **XGBoost:**
  - **Rationale:** Handles high-dimensional, heterogeneous feature sets well and often outperforms simpler models in structured data scenarios.
  
- **Logistic Regression:**
  - **Rationale:** Provides a straightforward, interpretable alternative with lower computational overhead.

### Evaluation Metrics

- **Metrics Used:**
  - **Accuracy, F1 Score, Precision, Recall:** Standard measures to assess model performance.
  - **Confusion Matrix:** Visualized to analyze misclassifications and understand the distribution of prediction errors.

### Feature Importance Analysis

- **Purpose:**
  - Assessing which features (e.g., perplexity, burstiness, logits) are most influential can guide further feature refinement and model improvements.

---

## 6. Insights and Future Directions

- **Feature Refinement:**  
  - Further investigation into alternative metrics (e.g., entropy-based features, n-gram patterns) may enhance detection capabilities.
  
- **Model Fine-tuning:**
  - Fine-tuning DistilBERT or GPT‑2 on domain-specific datasets could yield more discriminative features.
  
- **Real-Time Processing:**
  - Optimizing the multi-GPU pipeline and exploring streaming data approaches can enable real-time AI content detection.
  
- **Ensemble Methods:**
  - Combining multiple classifiers or incorporating deep ensemble techniques could further improve robustness and accuracy.

---

## 7. Conclusion

This pipeline demonstrates a comprehensive approach to AI content detection by combining:
- **Discriminative features** from DistilBERT,
- **Generative metrics** from GPT‑2 (overall perplexity, layer-wise log perplexity, burstiness, and DetectGPT-inspired perturbation),
- **Handcrafted linguistic features.**

The synergy of these diverse features enables the classifier to effectively differentiate between human-written and AI-generated text. With robust multi-GPU support and detailed error handling, the system is designed for scalability and reliability in real-world applications.

Future work will focus on refining feature extraction techniques, exploring additional metrics, and optimizing model training for enhanced performance.
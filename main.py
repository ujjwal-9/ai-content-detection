import torch
import math
import numpy as np
import nltk
import random
import pandas as pd
import pickle
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from data_parser import GhostbusterDataset
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
import torch.nn.functional as F


def load_models(gpu_id=None):
    """
    Load and return the required models and tokenizers.

    Args:
        gpu_id: Specific GPU ID to use, or None for automatic selection

    Returns:
        Tuple of (tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device)
    """
    # Check if CUDA is available
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on device: {device}")

    # Model for extracting logits and linguistic features (e.g., DistilBERT fine-tuned on sentiment)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer_llm = AutoTokenizer.from_pretrained(model_name)
    model_llm = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device
    )

    # GPT-2 for DetectGPT-style features and perplexity (log probability estimates)
    gpt2_model_name = "gpt2"
    tokenizer_gpt = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    model_gpt = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
    model_gpt.eval()  # set to evaluation mode

    return tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device


def get_available_gpus():
    """
    Get the number of available GPUs.

    Returns:
        Number of available GPUs
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def process_batch_with_gpu(batch_data, gpu_id):
    """
    Process a batch of samples using a specific GPU.

    Args:
        batch_data: Tuple of (batch_df, batch_idx, total_batches)
        gpu_id: GPU ID to use

    Returns:
        Tuple of (features, labels, processed_count, skipped_count)
    """
    batch_df, batch_idx, total_batches = batch_data

    # Load models on the specified GPU
    tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device = load_models(gpu_id)

    features = []
    labels = []
    skipped_samples = 0

    print(
        f"GPU {gpu_id}: Processing batch {batch_idx+1}/{total_batches} with {len(batch_df)} samples..."
    )

    start_time = time.time()

    for index, row in batch_df.iterrows():
        try:
            text = row["text"]
            label = row["label"]

            # Skip empty texts
            if not text or len(text.strip()) == 0:
                skipped_samples += 1
                continue

            feature_vector = create_feature_vector(
                text, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device
            )
            features.append(feature_vector)
            labels.append(label)

            # Print progress occasionally
            if len(features) % 10 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = len(features) / elapsed if elapsed > 0 else 0
                print(
                    f"GPU {gpu_id}: Processed {len(features)}/{len(batch_df)} samples ({samples_per_sec:.2f} samples/sec)"
                )

        except Exception as e:
            skipped_samples += 1
            print(f"GPU {gpu_id}: Error processing sample {index}: {str(e)[:100]}...")
            continue

    elapsed = time.time() - start_time
    samples_per_sec = len(features) / elapsed if elapsed > 0 else 0
    print(
        f"GPU {gpu_id}: Completed batch {batch_idx+1}/{total_batches}. "
        f"Processed {len(features)} samples, skipped {skipped_samples} samples. "
        f"Speed: {samples_per_sec:.2f} samples/sec"
    )

    # Clear GPU memory
    del model_llm
    del model_gpt
    torch.cuda.empty_cache()

    return features, labels, len(features), skipped_samples


def extract_features_from_dataset_multi_gpu(dataset, num_gpus=None, batch_size=None):
    """
    Extract features from a dataset using multiple GPUs.

    Args:
        dataset: DataFrame containing text samples and labels
        num_gpus: Number of GPUs to use (None for all available)
        batch_size: Number of samples per batch (None for automatic)

    Returns:
        Features array and labels array
    """
    # Get available GPUs
    available_gpus = get_available_gpus()

    if available_gpus == 0:
        print("No GPUs available. Using CPU instead.")
        # Fall back to single-device processing
        tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device = load_models()
        return extract_features_from_dataset(
            dataset,
            tokenizer_llm,
            model_llm,
            tokenizer_gpt,
            model_gpt,
            device,
            batch_size,
        )

    # Determine number of GPUs to use
    if num_gpus is None or num_gpus > available_gpus:
        num_gpus = available_gpus

    print(f"Using {num_gpus} GPUs for parallel processing")

    # Determine batch size and create batches
    if batch_size is None:
        # Automatically determine batch size based on dataset size and number of GPUs
        batch_size = max(1, len(dataset) // (num_gpus * 2))

    # Create batches
    batches = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_df = dataset.iloc[start_idx:end_idx]
        batches.append((batch_df, batch_idx, total_batches))

    print(
        f"Split dataset into {len(batches)} batches of approximately {batch_size} samples each"
    )

    # Process batches in parallel using multiple GPUs
    all_features = []
    all_labels = []
    total_processed = 0
    total_skipped = 0

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        # Submit batch processing tasks
        future_to_batch = {}
        for i, batch_data in enumerate(batches):
            gpu_id = i % num_gpus  # Assign batches to GPUs in a round-robin fashion
            future = executor.submit(process_batch_with_gpu, batch_data, gpu_id)
            future_to_batch[future] = batch_data

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_data = future_to_batch[future]
            try:
                features, labels, processed_count, skipped_count = future.result()
                all_features.extend(features)
                all_labels.extend(labels)
                total_processed += processed_count
                total_skipped += skipped_count
            except Exception as e:
                print(f"Batch processing failed: {str(e)}")

    print(
        f"Multi-GPU processing complete. Processed {total_processed} samples, skipped {total_skipped} samples."
    )

    return np.array(all_features), np.array(all_labels)


def extract_logits(text, tokenizer, model, device):
    """
    Extract logits using the LLM (DistilBERT).

    Args:
        text: Input text
        tokenizer: Tokenizer for the model
        model: The model to extract logits from
        device: The device to run the model on

    Returns:
        Numpy array of logits
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.logits shape is (batch_size, num_labels)
    return outputs.logits.cpu().detach().numpy()[0]


def extract_linguistic_features(text):
    """
    Extract basic linguistic features.

    Args:
        text: Input text

    Returns:
        Numpy array of linguistic features
    """
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    punctuation_count = sum(1 for char in text if char in ".,;!?")
    return np.array([word_count, sentence_count, avg_word_length, punctuation_count])


def compute_log_prob(text, model, tokenizer, device):
    """
    Compute the log probability of a text using GPT-2.

    Args:
        text: Input text
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: The device to run the model on

    Returns:
        Log probability score
    """
    # Add truncation to handle long texts
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    # Multiply average loss by token count to get the total log probability
    total_log_prob = -outputs.loss.item() * input_ids.shape[1]
    return total_log_prob


def perturb_text(text, num_changes=1):
    """
    Simple perturbation: randomly change characters in words.

    Args:
        text: Input text
        num_changes: Number of changes to make

    Returns:
        Perturbed text
    """
    words = text.split()
    if not words:
        return text
    for _ in range(num_changes):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 1:
            char_idx = random.randint(0, len(word) - 1)
            random_char = random.choice("abcdefghijklmnopqrstuvwxyz")
            # Ensure a change is made
            if word[char_idx] == random_char:
                random_char = random.choice("abcdefghijklmnopqrstuvwxyz")
            new_word = word[:char_idx] + random_char + word[char_idx + 1 :]
            words[idx] = new_word
    return " ".join(words)


def process_long_text(text, tokenizer, max_length=1000):
    """
    Process long text by splitting it into chunks if needed.

    Args:
        text: Input text
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum length of each chunk in tokens

    Returns:
        List of text chunks
    """
    # Encode the text
    tokens = tokenizer.encode(text)

    # If the text is short enough, return it as is
    if len(tokens) <= max_length:
        return [text]

    # Split into chunks
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i : i + max_length]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def compute_detectgpt_feature(text, model, tokenizer, device, num_perturbations=10):
    """
    Compute a DetectGPT-inspired feature.

    Args:
        text: Input text
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: The device to run the model on
        num_perturbations: Number of perturbations to generate

    Returns:
        DetectGPT feature score
    """
    # Handle long texts by processing the first chunk only
    chunks = process_long_text(text, tokenizer)
    text_to_process = chunks[0]  # Use the first chunk

    original_score = compute_log_prob(text_to_process, model, tokenizer, device)
    diff_scores = []
    for _ in range(num_perturbations):
        perturbed_text = perturb_text(text_to_process, num_changes=1)
        perturbed_score = compute_log_prob(perturbed_text, model, tokenizer, device)
        diff_scores.append(abs(original_score - perturbed_score))
    return np.mean(diff_scores)


def compute_perplexity(text, model, tokenizer, device):
    """
    Compute perplexity of the text using GPT-2.

    Args:
        text: Input text
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: The device to run the model on

    Returns:
        Perplexity score
    """
    # Add truncation to handle long texts
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    # The loss is the average negative log likelihood per token;
    # perplexity is computed as the exponentiation of the loss.
    loss = outputs.loss.item()
    perplexity = math.exp(loss)
    return perplexity


# Compute layer-wise "log perplexity" for first and last transformer layers of GPT-2.
def compute_layer_wise_perplexity(text, model, tokenizer, device):
    # Tokenize with explicit truncation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple: embeddings, layer1, ..., last layer
    # Use the first transformer block (hidden_states[1]) and the last (hidden_states[-1])
    first_hidden = hidden_states[1]
    second_hidden = hidden_states[2]
    second_last_hidden = hidden_states[-2]
    last_hidden = hidden_states[-1]

    # Compute logits for each using the lm_head
    logits_first = model.lm_head(first_hidden)
    logits_second = model.lm_head(second_hidden)
    logits_second_last = model.lm_head(second_last_hidden)
    logits_last = model.lm_head(last_hidden)

    # Prepare shifted logits and labels for loss computation
    shift_logits_first = logits_first[:, :-1, :].contiguous()
    shift_logits_second = logits_second[:, :-1, :].contiguous()
    shift_logits_second_last = logits_second_last[:, :-1, :].contiguous()
    shift_logits_last = logits_last[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_first = F.cross_entropy(
        shift_logits_first.view(-1, shift_logits_first.size(-1)), shift_labels.view(-1)
    )
    loss_second = F.cross_entropy(
        shift_logits_second.view(-1, shift_logits_second.size(-1)),
        shift_labels.view(-1),
    )
    loss_second_last = F.cross_entropy(
        shift_logits_second_last.view(-1, shift_logits_second_last.size(-1)),
        shift_labels.view(-1),
    )
    loss_last = F.cross_entropy(
        shift_logits_last.view(-1, shift_logits_last.size(-1)), shift_labels.view(-1)
    )

    # Instead of computing perplexity (exp(loss)) which may be huge, we use the raw loss (log perplexity) as features.
    log_perplexity_first = loss_first.item()
    log_perplexity_second = loss_second.item()
    log_perplexity_second_last = loss_second_last.item()
    log_perplexity_last = loss_last.item()

    return (
        log_perplexity_first,
        log_perplexity_second,
        log_perplexity_second_last,
        log_perplexity_last,
    )


# Define burstiness as the standard deviation of token log probabilities.
def compute_layer_wise_burstiness(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    # Select first and last transformer blocks
    first_hidden = hidden_states[1]
    second_hidden = hidden_states[2]
    second_last_hidden = hidden_states[-2]
    last_hidden = hidden_states[-1]

    logits_first = model.lm_head(first_hidden)
    logits_second = model.lm_head(second_hidden)
    logits_second_last = model.lm_head(second_last_hidden)
    logits_last = model.lm_head(last_hidden)

    # Align logits with labels (shifted)
    shift_logits_first = logits_first[:, :-1, :].contiguous()
    shift_logits_second = logits_second[:, :-1, :].contiguous()
    shift_logits_second_last = logits_second_last[:, :-1, :].contiguous()
    shift_logits_last = logits_last[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute log probabilities using softmax
    log_probs_first = F.log_softmax(shift_logits_first, dim=-1)
    log_probs_second = F.log_softmax(shift_logits_second, dim=-1)
    log_probs_second_last = F.log_softmax(shift_logits_second_last, dim=-1)
    log_probs_last = F.log_softmax(shift_logits_last, dim=-1)

    # Gather the log probabilities for the actual (ground truth) tokens
    token_log_probs_first = log_probs_first.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs_second = log_probs_second.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs_second_last = log_probs_second_last.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probs_last = log_probs_last.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    burstiness_first = token_log_probs_first.std().item()
    burstiness_second = token_log_probs_second.std().item()
    burstiness_second_last = token_log_probs_second_last.std().item()
    burstiness_last = token_log_probs_last.std().item()

    return burstiness_first, burstiness_second, burstiness_second_last, burstiness_last


def create_feature_vector(
    text,
    tokenizer_llm,
    model_llm,
    tokenizer_gpt,
    model_gpt,
    device,
    num_detectgpt_perturbations=10,
):
    """
    Combine features into a single feature vector.

    Args:
        text: Input text
        tokenizer_llm: LLM tokenizer
        model_llm: LLM model
        tokenizer_gpt: GPT-2 tokenizer
        model_gpt: GPT-2 model
        device: The device to run the models on
        num_detectgpt_perturbations: Number of perturbations for DetectGPT

    Returns:
        Combined feature vector
    """
    try:
        # Truncate very long texts to avoid CUDA errors
        if len(text) > 10000:
            text = text[:10000]

        # LLM (DistilBERT) logits
        logits = extract_logits(text, tokenizer_llm, model_llm, device)

        # Linguistic features
        linguistic_features = extract_linguistic_features(text)

        # DetectGPT-inspired feature
        try:
            detectgpt_feature = compute_detectgpt_feature(
                text, model_gpt, tokenizer_gpt, device, num_detectgpt_perturbations
            )
        except Exception as e:
            print(f"Error computing DetectGPT feature: {str(e)[:100]}...")
            detectgpt_feature = 0.0

        # Perplexity feature using GPT-2
        # Handle long texts by processing the first chunk only
        chunks = process_long_text(text, tokenizer_gpt)
        try:
            perplexity_feature = compute_perplexity(
                chunks[0], model_gpt, tokenizer_gpt, device
            )
        except Exception as e:
            print(f"Error computing perplexity: {str(e)[:100]}...")
            perplexity_feature = 1000.0

        # Layer-wise perplexity features
        try:
            first_layer_ppl, second_layer_ppl, second_last_layer_ppl, last_layer_ppl = (
                compute_layer_wise_perplexity(
                    chunks[0], model_gpt, tokenizer_gpt, device
                )
            )
        except Exception as e:
            print(f"Error computing layer-wise perplexity: {str(e)[:100]}...")
            first_layer_ppl, second_layer_ppl, second_last_layer_ppl, last_layer_ppl = (
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            )

        # Layer-wise burstiness features
        try:
            (
                first_layer_burstiness,
                second_layer_burstiness,
                second_last_layer_burstiness,
                last_layer_burstiness,
            ) = compute_layer_wise_burstiness(
                chunks[0], model_gpt, tokenizer_gpt, device
            )
        except Exception as e:
            print(f"Error computing layer-wise burstiness: {str(e)[:100]}...")
            (
                first_layer_burstiness,
                second_layer_burstiness,
                second_last_layer_burstiness,
                last_layer_burstiness,
            ) = (0.0, 0.0, 0.0, 0.0)

        # Combine all features
        feature_vector = np.concatenate(
            [
                logits,
                linguistic_features,
                np.array(
                    [
                        detectgpt_feature,
                        perplexity_feature,
                        first_layer_ppl,
                        second_layer_ppl,
                        second_last_layer_ppl,
                        last_layer_ppl,
                        first_layer_burstiness,
                        second_layer_burstiness,
                        second_last_layer_burstiness,
                        last_layer_burstiness,
                    ]
                ),
            ]
        )

        return feature_vector
    except Exception as e:
        # If anything fails, return a default feature vector
        print(f"Error creating feature vector: {str(e)[:100]}...")
        # Create a default feature vector with the right dimensions
        logits_dim = 2  # Typical dimension for DistilBERT logits
        ling_features_dim = 8  # Typical dimension for linguistic features
        default_features = np.zeros(
            logits_dim + ling_features_dim + 4
        )  # +4 for detectgpt, perplexity, and 2 layer perplexities
        return default_features


def extract_features_from_dataset(
    dataset, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device, batch_size=None
):
    """
    Extract features from a dataset.

    Args:
        dataset: DataFrame containing text samples and labels
        tokenizer_llm: LLM tokenizer
        model_llm: LLM model
        tokenizer_gpt: GPT-2 tokenizer
        model_gpt: GPT-2 model
        device: The device to run the models on
        batch_size: Number of samples to process at once (None for all at once)

    Returns:
        Features array and labels array
    """
    features = []
    labels = []
    skipped_samples = 0

    # Process in batches if specified
    if batch_size:
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        print(
            f"Processing {len(dataset)} samples in {total_batches} batches of size {batch_size}"
        )

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset.iloc[start_idx:end_idx]

            print(
                f"Processing batch {batch_idx+1}/{total_batches} (samples {start_idx+1}-{end_idx})..."
            )

            # Process each sample in the batch
            for index, row in batch.iterrows():
                try:
                    text = row["text"]
                    label = row["label"]

                    # Skip empty texts
                    if not text or len(text.strip()) == 0:
                        skipped_samples += 1
                        continue

                    feature_vector = create_feature_vector(
                        text, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device
                    )
                    features.append(feature_vector)
                    labels.append(label)

                    # Print progress every 10 samples within a batch
                    if len(features) % 10 == 0:
                        print(f"Processed {len(features)} samples total...")

                except Exception as e:
                    skipped_samples += 1
                    print(f"Error processing sample {index}: {str(e)[:100]}...")
                    continue
    else:
        # Process all samples at once
        for index, row in dataset.iterrows():
            try:
                text = row["text"]
                label = row["label"]

                # Skip empty texts
                if not text or len(text.strip()) == 0:
                    skipped_samples += 1
                    continue

                feature_vector = create_feature_vector(
                    text, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device
                )
                features.append(feature_vector)
                labels.append(label)

                # Print progress every 100 samples
                if len(features) % 100 == 0:
                    print(f"Processed {len(features)} samples...")

            except Exception as e:
                skipped_samples += 1
                print(f"Error processing sample {index}: {str(e)[:100]}...")
                continue

    print(
        f"Completed feature extraction. Processed {len(features)} samples, skipped {skipped_samples} samples."
    )
    return np.array(features), np.array(labels)


def train_and_evaluate_model(
    features, labels, test_size=0.25, random_state=42, use_xgboost=True, xgb_params=None
):
    """
    Train and evaluate a model on the given features and labels.

    Args:
        features: Feature matrix
        labels: Target labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        use_xgboost: Whether to use XGBoost (True) or LogisticRegression (False)
        xgb_params: Dictionary of XGBoost parameters

    Returns:
        clf: Trained classifier
        metrics: Dictionary of evaluation metrics
        X_test: Test features
        y_test: Test labels
        y_pred: Predicted labels
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(
        f"Training set: {len(y_train)} samples ({np.sum(y_train == 1)} AI-generated, {np.sum(y_train == 0)} human-written)"
    )
    print(
        f"Test set: {len(y_test)} samples ({np.sum(y_test == 1)} AI-generated, {np.sum(y_test == 0)} human-written)"
    )
    print(f"Total samples: {len(y_train) + len(y_test)}")

    # Balance the test set to ensure equal representation of both classes
    ai_indices = np.where(y_test == 1)[0]
    human_indices = np.where(y_test == 0)[0]

    # Determine the target size (minimum of the two classes)
    target_size = min(len(ai_indices), len(human_indices))

    # Resample the larger class to match the smaller one
    if len(ai_indices) > len(human_indices):
        ai_indices = resample(
            ai_indices, n_samples=target_size, random_state=random_state
        )
    elif len(human_indices) > len(ai_indices):
        human_indices = resample(
            human_indices, n_samples=target_size, random_state=random_state
        )

    # Combine the balanced indices and extract the balanced test set
    balanced_indices = np.concatenate([ai_indices, human_indices])
    X_test = X_test[balanced_indices]
    y_test = y_test[balanced_indices]

    # Choose classifier based on parameter
    if use_xgboost:
        # Default XGBoost parameters if none provided
        if xgb_params is None:
            xgb_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "random_state": random_state,
            }

        # Create and train XGBoost classifier
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_train, y_train)

        # Get feature importances
        feature_importances = clf.feature_importances_
        print("\nFeature Importances:")
        for i, importance in enumerate(feature_importances):
            print(f"Feature {i}: {importance:.4f}")
    else:
        # Use LogisticRegression as fallback
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    return clf, metrics, X_test, y_test, y_pred


def print_dataset_stats(dataset):
    """
    Print statistics about the dataset.

    Args:
        dataset: GhostbusterDataset instance
    """
    stats = dataset.get_dataset_stats()
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")

    print("\nSamples by dataset:")
    for dataset, count in stats["by_dataset"].items():
        print(f"{dataset}: {count}")

    print("\nSamples by source:")
    for source, count in stats["by_source"].items():
        print(f"{source}: {count}")


def save_model(clf, feature_info, output_dir="models"):
    """
    Save the trained model and feature information for later use.

    Args:
        clf: Trained classifier
        feature_info: Dictionary containing feature information
        output_dir: Directory to save the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(output_dir, "ai_content_detector.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save feature information
    feature_info_path = os.path.join(output_dir, "feature_info.pkl")
    with open(feature_info_path, "wb") as f:
        pickle.dump(feature_info, f)

    print(f"\nModel saved to {model_path}")
    print(f"Feature information saved to {feature_info_path}")


def load_model(model_dir="models"):
    """
    Load the trained model and feature information.

    Args:
        model_dir: Directory where the model is saved

    Returns:
        Trained classifier and feature information
    """
    model_path = os.path.join(model_dir, "ai_content_detector.pkl")
    feature_info_path = os.path.join(model_dir, "feature_info.pkl")

    # Load the model
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Load feature information
    with open(feature_info_path, "rb") as f:
        feature_info = pickle.load(f)

    return clf, feature_info


def predict_text(text, clf, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device):
    """
    Make a prediction on a single text sample.

    Args:
        text: Text to classify
        clf: Trained classifier
        tokenizer_llm: LLM tokenizer
        model_llm: LLM model
        tokenizer_gpt: GPT-2 tokenizer
        model_gpt: GPT-2 model
        device: The device to run the models on

    Returns:
        Prediction (0 for human, 1 for AI) and confidence score
    """
    # Extract features
    feature_vector = create_feature_vector(
        text, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device
    )

    # Make prediction
    prediction = clf.predict([feature_vector])[0]

    # Get confidence score
    confidence = clf.predict_proba([feature_vector])[0][prediction]

    return prediction, confidence


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot and optionally save a confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure and axes
    plt.figure(figsize=(8, 6))

    # Plot confusion matrix as heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
    )

    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the AI content detection pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Content Detection Tool")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force training a new model even if one exists",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Text to predict (if provided, will only run prediction)",
    )
    parser.add_argument(
        "--predict-file", type=str, help="Path to a file containing text to predict"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use for training (for testing purposes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of samples to process at once (for memory constraints)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force using CPU even if CUDA is available"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable multi-GPU processing even if multiple GPUs are available",
    )
    parser.add_argument(
        "--plot-cm",
        action="store_true",
        help="Plot confusion matrix after model evaluation",
    )
    parser.add_argument(
        "--save-cm",
        type=str,
        default=None,
        help="Path to save confusion matrix plot (e.g., 'plots/confusion_matrix.png')",
    )
    parser.add_argument(
        "--use-logistic",
        action="store_true",
        help="Use LogisticRegression instead of XGBoost (default is XGBoost)",
    )
    parser.add_argument(
        "--xgb-depth",
        type=int,
        default=6,
        help="XGBoost max_depth parameter (default: 6)",
    )
    parser.add_argument(
        "--xgb-lr",
        type=float,
        default=0.1,
        help="XGBoost learning_rate parameter (default: 0.1)",
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=100,
        help="XGBoost n_estimators parameter (default: 100)",
    )
    args = parser.parse_args()

    # Ensure NLTK resources are downloaded
    nltk.download("punkt", quiet=True)

    # Force CPU if requested
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Forcing CPU usage as requested")

    # For prediction, we only need to load models once
    if args.predict or args.predict_file:
        # Load models
        print("Loading models...")
        tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device = load_models()

        # Check if model exists
        model_path = os.path.join("models", "ai_content_detector.pkl")
        if not os.path.exists(model_path):
            print(
                "Error: No trained model found. Please run the script without --predict first to train a model."
            )
            return

        # Load the model
        clf, feature_info = load_model()

        # Get text to predict
        if args.predict:
            text = args.predict
        else:
            with open(args.predict_file, "r", encoding="utf-8") as f:
                text = f.read()

        # Make prediction
        prediction, confidence = predict_text(
            text, clf, tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device
        )

        # Print result
        print("\nPrediction Results:")
        print(
            f"Text classified as: {'AI-generated' if prediction == 1 else 'Human-written'}"
        )
        print(f"Confidence: {confidence:.4f}")
        return

    # Load dataset
    print("Loading dataset...")
    datasets = GhostbusterDataset()

    if args.train or not os.path.exists(
        os.path.join("models", "ai_content_detector.pkl")
    ):
        # Print dataset statistics
        print_dataset_stats(datasets)

        # Load all samples from different datasets
        wp_human_samples = datasets.get_samples(dataset_type="wp", source_type="human")
        essays_human_samples = datasets.get_samples(
            dataset_type="essay", source_type="human"
        )
        reuters_human_samples = datasets.get_samples(
            dataset_type="reuter", source_type="human"
        )
        wp_gpt_samples = datasets.get_samples(dataset_type="wp", source_type="gpt")
        essays_gpt_samples = datasets.get_samples(
            dataset_type="essay", source_type="gpt"
        )
        # reuters_gpt_samples = datasets.get_samples(
        #     dataset_type="reuter", source_type="gpt"
        # )

        print(f"\nNumber of human WP samples: {len(wp_human_samples)}")
        print(f"Number of human essays samples: {len(essays_human_samples)}")
        print(f"Number of human Reuters samples: {len(reuters_human_samples)}")
        print(f"Number of GPT WP samples: {len(wp_gpt_samples)}")
        print(f"Number of GPT essays samples: {len(essays_gpt_samples)}")
        # print(f"Number of GPT Reuters samples: {len(reuters_gpt_samples)}")

        print("\nTraining new model...")
        # Prepare dataset for feature extraction - include all sample types
        human_data = [
            {"text": sample.text, "label": 0}
            for sample in wp_human_samples
            + essays_human_samples
            + reuters_human_samples
        ]
        gpt_data = [
            {"text": sample.text, "label": 1}
            for sample in wp_gpt_samples + essays_gpt_samples
        ]

        # Limit dataset size if specified
        if args.max_samples:
            max_human = min(len(human_data), args.max_samples // 2)
            max_gpt = min(len(gpt_data), args.max_samples - max_human)
            human_data = human_data[:max_human]
            gpt_data = gpt_data[:max_gpt]
            print(f"\nLimiting dataset to {max_human + max_gpt} samples for testing")

        all_data = human_data + gpt_data
        df = pd.DataFrame(all_data)

        print(f"\nPrepared dataset with {len(df)} samples")
        print(f"Human samples: {len(human_data)}")
        print(f"GPT samples: {len(gpt_data)}")

        # Extract features
        print("Extracting features...")
        start_time = time.time()

        # Check if we should use multi-GPU processing
        if not args.no_multi_gpu and not args.cpu and get_available_gpus() > 1:
            # Use multi-GPU processing
            features, labels = extract_features_from_dataset_multi_gpu(
                df, num_gpus=args.gpus, batch_size=args.batch_size
            )
        else:
            # Use single device processing
            tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device = load_models()
            features, labels = extract_features_from_dataset(
                df,
                tokenizer_llm,
                model_llm,
                tokenizer_gpt,
                model_gpt,
                device,
                batch_size=args.batch_size,
            )

        elapsed = time.time() - start_time
        print(f"Feature extraction completed in {elapsed:.2f} seconds")

        # Train and evaluate model
        print("Training model...")

        # Configure XGBoost parameters if using XGBoost
        xgb_params = None
        if not args.use_logistic:
            xgb_params = {
                "max_depth": args.xgb_depth,
                "learning_rate": args.xgb_lr,
                "n_estimators": args.xgb_estimators,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "random_state": 42,
            }
            print(
                f"Using XGBoost with parameters: max_depth={args.xgb_depth}, "
                f"learning_rate={args.xgb_lr}, n_estimators={args.xgb_estimators}"
            )
        else:
            print("Using LogisticRegression classifier")

        clf, metrics, X_test, y_test, y_pred = train_and_evaluate_model(
            features, labels, use_xgboost=not args.use_logistic, xgb_params=xgb_params
        )

        # Print evaluation metrics
        print("\nModel Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

        # Plot confusion matrix if requested
        if args.plot_cm or args.save_cm:
            print("\nGenerating confusion matrix...")
            plot_confusion_matrix(y_test, y_pred, save_path=args.save_cm)

        # Save the model and feature information
        feature_info = {
            "model_llm_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "model_gpt_name": "gpt2",
            "feature_names": [
                "logits",
                "linguistic_features",
                "detectgpt",
                "perplexity",
                "first_layer_perplexity",
                "last_layer_perplexity",
            ],
            "num_features": features.shape[1],
        }
        save_model(clf, feature_info)
    else:
        # Load models for example predictions
        tokenizer_llm, model_llm, tokenizer_gpt, model_gpt, device = load_models()

        print("\nLoading existing model...")
        clf, feature_info = load_model()

        # Example: Load specific samples for testing
        wp_human_samples = datasets.get_samples(dataset_type="wp", source_type="human")
        wp_gpt_samples = datasets.get_samples(dataset_type="wp", source_type="gpt")

        # Example: Make a prediction on a sample text
        if wp_human_samples and wp_gpt_samples:
            # Test on one human sample and one AI sample
            human_sample = wp_human_samples[0].text
            ai_sample = wp_gpt_samples[0].text

            human_pred, human_conf = predict_text(
                human_sample,
                clf,
                tokenizer_llm,
                model_llm,
                tokenizer_gpt,
                model_gpt,
                device,
            )
            ai_pred, ai_conf = predict_text(
                ai_sample,
                clf,
                tokenizer_llm,
                model_llm,
                tokenizer_gpt,
                model_gpt,
                device,
            )

            print("\nExample predictions:")
            print(
                f"Human sample predicted as: {'AI-generated' if human_pred == 1 else 'Human-written'} (confidence: {human_conf:.4f})"
            )
            print(
                f"AI sample predicted as: {'AI-generated' if ai_pred == 1 else 'Human-written'} (confidence: {ai_conf:.4f})"
            )


if __name__ == "__main__":
    main()

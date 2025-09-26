import os
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#from nnsight import LanguageModel


def load_model_and_vectors(model_name: str, model=None, compute_features=True, normalize_features=True) -> Dict[str, torch.Tensor]:

    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    # go into directory of this file
    vector_path = f"../train-steering-vectors/results/vars/mean_vectors_{model_id}.pt"
    if os.path.exists(vector_path):
        mean_vectors_dict = torch.load(vector_path)

        if compute_features:
            # Compute feature vectors by subtracting overall mean
            feature_vectors = {}
            feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']

            for label in ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]:

                if label != 'overall':
                    feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']

                if normalize_features:
                    for label in feature_vectors:
                        for layer in range(model.config.num_hidden_layers):
                            feature_vectors[label][layer] = feature_vectors[label][layer] * (feature_vectors["overall"][layer].norm() / feature_vectors[label][layer].norm())

        return feature_vectors

    else:
        raise FileNotFoundError(f"Steering vectors not found at: {vector_path}")
        mean_vectors_dict = {}
        feature_vectors = {}
        return feature_vectors

def custom_generate_steering(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steering_config, steer_positive=False):
    """
    Generate text while removing or adding projections of specific features.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors containing steering_vector_set
        steer_positive: If True, steer towards the label, if False steer away
    """
    model_layers = model.model.layers
    print("custom_generate_steering")

    with model.generate(
        {
            "input_ids": input_ids,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if feature_vectors is not None:
            vector_layer = steering_config[label]["vector_layer"]
            pos_layers = steering_config[label]["pos_layers"]
            neg_layers = steering_config[label]["neg_layers"]
            coefficient = steering_config[label]["pos_coefficient"] if steer_positive else steering_config[label]["neg_coefficient"]


            if steer_positive:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in pos_layers:
                    model.model.layers[layer_idx].output[0][:, :] += coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
            else:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in neg_layers:
                    model.model.layers[layer_idx].output[0][:, :] -= coefficient * feature_vector.unsqueeze(0).unsqueeze(0)

        outputs = model.generator.output.save()

    return outputs


steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "backtracking": {"vector_layer": 17, "pos_layers": [17], "neg_layers": [17], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 15, "pos_layers": [15], "neg_layers": [15], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "backtracking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "backtracking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 24, "pos_layers": [24], "neg_layers": [24], "pos_coefficient": 1, "neg_coefficient": 1},
    }

}

from transformers import ViTForImageClassification, BeitForImageClassification, AutoFeatureExtractor


# Transformer Models

def create_transformer_model(model_name, num_classes=4):
    if model_name == 'ViT':
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    elif model_name == 'Beit':
        model = BeitForImageClassification.from_pretrained(
            "microsoft/beit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError("Invalid model name. Choose 'ViT' or 'Beit'.")
    return model

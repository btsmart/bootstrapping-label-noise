import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


logger = logging.getLogger(__name__)

# ---------------
# --- Modules ---
# ---------------


class SemanticModule(nn.Module):
    """Basic module that accepts a feature-encoding and a noisy label vector, and
    combines the two to form a prediction"""

    def __init__(self, config, encoding_size=128, label_size=10, num_classes=10):
        super(SemanticModule, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, config.model.feature_encoding_size),
            nn.Dropout(p=config.model.dropout, inplace=False),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, config.model.label_encoding_size),
            nn.Dropout(p=config.model.dropout, inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(
                config.model.feature_encoding_size + config.model.label_encoding_size,
                config.model.hidden_layer_size,
            ),
            nn.Dropout(p=config.model.dropout, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(config.model.hidden_layer_size),
            nn.Linear(config.model.hidden_layer_size, num_classes),
        )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        # @NOTE: A skip connection can be used here like in alternate methods
        # out = labels + self.classifier(out)
        out = self.classifier(out)
        return out


# --------------
# --- Models ---
# --------------


class RegularModel(nn.Module):
    """A simple model that uses a linear layer to predict the true label based on
    image features. Note that the noisy label of samples is ignored during prediction.
    It is here to provide a consistent interface with the semantic model"""

    def __init__(self, config, encoder, encoding_size=128, num_classes=10):
        super(RegularModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(encoding_size, num_classes),)

    def forward(self, x, y=None):
        encoding = self.encoder(x)
        out = self.classifier(encoding)
        return out


class SemanticModel(nn.Module):
    """A model that attempts to combine an image and its noisy labels to make a
    prediction of the true label"""

    def __init__(
        self, config, encoder, encoding_size=128, num_classes=10, label_size=10
    ):
        super(SemanticModel, self).__init__()
        self.encoding_size = encoding_size
        self.encoder = encoder
        self.semantic_module = SemanticModule(
            config, encoding_size, label_size, num_classes
        )

    def forward(self, x, y):
        encoding = self.encoder(x)
        out = self.semantic_module(encoding, y)
        return out


# --------------
# --- ResNet ---
# --------------


def get_preact_resnet_encoder(config, load_pretrained):
    """Load a pretrained resnet checkpoint"""

    import models.preact_resnet_cifar

    data = models.preact_resnet_cifar.PreActResNet18()
    encoder, encoding_size = data["backbone"], data["dim"]

    if load_pretrained:
        checkpoint_path = os.path.join(
            config.current_dir,
            "models",
            "simclr_pretrained",
            f"{config.data.dataset}_{config.model.base_model_type}.pth",
        )
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
        sd = {}
        for key in checkpoint["model"]:
            new_key = key.replace("encoder.", "")
            new_key = new_key.replace("backbone.", "")
            sd[new_key] = checkpoint["model"][key]
        encoder.load_state_dict(sd, strict=False)

    return encoder, encoding_size


def get_vgg_encoder(config, load_pretrained):
    """Load a pretrained VGG checkpoint"""

    import models.vgg

    data = models.vgg.get_vgg19_bn()
    encoder, encoding_size = data["backbone"], data["dim"]

    if load_pretrained:
        checkpoint_path = os.path.join(
            config.current_dir,
            "models",
            "simclr_pretrained",
            f"{config.data.dataset}_{config.model.base_model_type}.pth",
        )
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
        sd = {}
        for key in checkpoint["model"]:
            new_key = key.replace("encoder.", "")
            new_key = new_key.replace("backbone.", "")
            sd[new_key] = checkpoint["model"][key]
        encoder.load_state_dict(sd, strict=False)

    return encoder, encoding_size


def get_inception_resnet_v2_encoder(config, load_pretrained):
    """Load a pretrained Inception Resnet V2 checkpoint"""

    import models.InceptionResNetV2

    data = models.InceptionResNetV2.network()
    encoder, encoding_size = data["backbone"], data["dim"]

    if load_pretrained:
        checkpoint_path = os.path.join(
            config.current_dir,
            "models",
            "mocov2_pretrained",
            f"{config.data.dataset}.pth.tar",
        )
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
        sd = {}
        for key in checkpoint["model"]:
            new_key = key.replace("backbone.", "")
            sd[new_key] = checkpoint["model"][key]
        encoder.load_state_dict(sd, strict=False)

    return encoder, encoding_size


def get_encoder(config, load_pretrained=False):

    if config.model.base_model_type == "preact_resnet18":
        logger.info(f"Creating PreAct ResNet model: {config.model.base_model_type}")
        encoder, encoding_size = get_preact_resnet_encoder(config, load_pretrained)

    elif config.model.base_model_type == "vgg19":
        logger.info(f"Creating VGG19 model: {config.model.base_model_type}")
        encoder, encoding_size = get_vgg_encoder(config, load_pretrained)

    elif config.model.base_model_type == "inception_resnet_v2":
        logger.info(
            f"Creating Inception Resnet v2 Model: {config.model.base_model_type}"
        )
        encoder, encoding_size = get_inception_resnet_v2_encoder(
            config, load_pretrained
        )

    else:
        return NotImplementedError(
            f"Model type {config.modell.base_model_type} not implemented"
        )

    return encoder, encoding_size


def create_model(
    config,
    semantic_type: str,
    num_classes: int,
    label_size: int = 10,
    load_pretrained: bool = True,
):
    """Create a model given the encoding type, noisy label integration type and other
    hyperparameters."""

    encoder, encoding_size = get_encoder(config, load_pretrained)

    logger.info(
        f"Creating a {semantic_type} model with "
        f"num_classes={num_classes}"
        f", encoding_size={encoding_size}"
        f" and label_size={label_size}"
    )

    # A `normal` model is an encoder followed by a linear classification head
    if semantic_type == "normal":
        model = RegularModel(config, encoder, encoding_size, num_classes)

    # A `semantic` model concatenates the feature encoding with the noisy label before
    # using a fully connected MLP classification head
    elif semantic_type == "semantic":
        model = SemanticModel(config, encoder, encoding_size, num_classes, label_size)

    # Other types of models (eg. attention and mixture of experts)
    elif semantic_type.startswith("semantic"):
        from models.ablation_models import CustomSemanticModel

        model = CustomSemanticModel(
            config, encoder, semantic_type, encoding_size, num_classes, label_size
        )

    total_parameters = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: {:.2f}M".format(total_parameters / 1e6))

    return model

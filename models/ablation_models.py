import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from models.models import get_encoder, RegularModel

# A list of other model constructions used for ablation studies

logger = logging.getLogger(__name__)

class SemanticModule1(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule1, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, feature_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, label_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size,),
            nn.Dropout(p=dropout, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Linear(hidden_layer_size, num_classes),
        )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        out = self.classifier(out)
        return out


class SemanticModule2(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule2, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, feature_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, label_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size),
            nn.Dropout(p=dropout, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Linear(hidden_layer_size, hidden_layer_size // 2),
            nn.Dropout(p=dropout, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size // 2),
            nn.Linear(hidden_layer_size // 2, num_classes),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size,),
        #     nn.Dropout(p=dropout, inplace=False),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_layer_size),
        #     nn.Linear(hidden_layer_size, num_classes),
        # )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        out = self.classifier(out)
        return out


class SemanticModule3(nn.Module):
    def __init__(
        self,
        config,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule3, self).__init__()
        # self.classifier = nn.Sequential(
        #     # nn.Linear(encoding_size + label_size, num_classes),
        #     MaskedLinear(encoding_size, label_size),
        #     # nn.Dropout(p=dropout, inplace=False),
        #     nn.ReLU(),
        #     # nn.BatchNorm1d(num_classes),
        #     nn.Linear(num_classes, num_classes),
        # )
        self.classifier = nn.Sequential(
            MaskedLinear2(encoding_size, label_size),
            nn.ReLU(),
            nn.Linear(encoding_size, num_classes),
        )
        self.classifier[0].set_mask()

    def forward(self, encoding, labels):
        out = torch.cat((encoding, labels), dim=1)
        out = self.classifier(out)
        return out


class SemanticModule4(nn.Module):
    def __init__(
        self, config, encoding_size=128, label_size=10, num_classes=10,
    ):
        super(SemanticModule4, self).__init__()
        self.classifier = nn.Sequential(
            MaskedLinear2(encoding_size, label_size),
            nn.ReLU(),
            nn.Linear(encoding_size * label_size, num_classes),
        )
        self.classifier[0].set_mask()

    def forward(self, encoding, labels):
        out = torch.cat((encoding, labels), dim=1)
        out = self.classifier(out)
        return out


class MaskedLinear2(nn.Module):
    def __init__(self, left_size, right_size):
        super(MaskedLinear2, self).__init__()

        # Calculate the mask
        self.mask = torch.zeros(left_size * right_size, left_size + right_size)
        for i in range(left_size):
            for j in range(right_size):
                self.mask[i * right_size + j, i] = 1
                self.mask[i * right_size + j, left_size + j] = 1

        # Create the pruned linear layer
        self.classifier = nn.Linear(left_size + right_size, left_size * right_size)

    def set_mask(self):
        self.classifier = prune.custom_from_mask(
            self.classifier, name="weight", mask=self.mask
        )

    def remove_mask(self):
        self.classifier = prune.remove(self.classifier, "weight")

    def forward(self, x):
        out = self.classifier(x)
        return out


class SemanticModuleMOE(nn.Module):
    def __init__(
        self, config, encoding_size=128, label_size=10, num_classes=10,
    ):
        super(SemanticModuleMOE, self).__init__()
        self.classifiers = nn.ModuleList()
        self.label_size = label_size
        for i in range(self.label_size + 1):
            self.classifiers.append(nn.Linear(encoding_size, num_classes))

    def forward(self, encoding, labels):

        outputs = []
        for i, classifier in enumerate(self.classifiers):
            outputs.append(classifier(encoding))

        null_weight = 1.0 - torch.sum(labels, dim=1)
        output = null_weight[:, None] * outputs[self.label_size]
        for i in range(0, self.label_size):
            output += labels[:, i][:, None] * outputs[i]

        return output


class SemanticModule5(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule5, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, feature_encoding_size),
            nn.BatchNorm1d(feature_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, label_encoding_size),
            nn.BatchNorm1d(label_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size,),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_layer_size, num_classes),
        )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        out = self.classifier(out)
        return out


class SemanticModule6(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule6, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, feature_encoding_size),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, label_encoding_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size,),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes),
        )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        out = self.classifier(out)
        return out


class SemanticModule7(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule7, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, feature_encoding_size),
            nn.BatchNorm1d(feature_encoding_size),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, label_encoding_size),
            nn.BatchNorm1d(label_encoding_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size,),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes),
        )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        out = self.classifier(out)
        return out


class SemanticModule8(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModule8, self).__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(encoding_size, feature_encoding_size),
        )
        self.label_projector = nn.Sequential(
            nn.Linear(label_size, label_encoding_size),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_encoding_size + label_encoding_size),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(feature_encoding_size + label_encoding_size, hidden_layer_size,),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_layer_size, num_classes),
        )

    def forward(self, encoding, labels):
        feature_embedding = self.feature_projector(encoding)
        label_embedding = self.label_projector(labels)
        out = torch.cat((feature_embedding, label_embedding), dim=1)
        out = self.classifier(out)
        return out


class SemanticModuleAttention(nn.Module):
    def __init__(
        self,
        config,
        feature_encoding_size=128,
        label_encoding_size=128,
        hidden_layer_size=128,
        dropout=0.2,
        encoding_size=128,
        label_size=10,
        num_classes=10,
    ):
        super(SemanticModuleAttention, self).__init__()

        self.key_size = 16
        self.value_size = 16
        self.sequence_length = 128

        self.target_sequence_length = 1

        self.query_projector = nn.Sequential(nn.Linear(label_size, self.key_size),)
        self.key_projector = nn.Sequential(
            nn.Linear(encoding_size, self.key_size * self.sequence_length),
        )
        self.value_projector = nn.Sequential(
            nn.Linear(encoding_size, self.value_size * self.sequence_length),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.value_size),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(self.value_size, num_classes),
        )

    # From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    def scaled_dot_product(self, q, k, v, mask=None):

        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, encoding, labels):

        batch_size = encoding.size(0)

        query = self.query_projector(labels)
        query = query.view(batch_size, self.target_sequence_length, self.key_size)

        key = self.key_projector(encoding)
        key = key.view(batch_size, self.sequence_length, self.key_size)

        value = self.value_projector(encoding)
        value = value.view(batch_size, self.sequence_length, self.value_size)

        out, _ = self.scaled_dot_product(query, key, value)
        out = out.view(batch_size, self.value_size)

        out = self.classifier(out)

        return out


class CustomSemanticModel(nn.Module):
    """A model that attempts to combine an image and its noisy labels to make a
    prediction of the true label"""

    def __init__(
        self,
        config,
        encoder,
        semantic_type,
        encoding_size=128,
        num_classes=10,
        label_size=10,
    ):
        super(CustomSemanticModel, self).__init__()
        self.encoding_size = encoding_size
        # self.num_classes = num_classes
        # self.label_size = label_size
        self.encoder = encoder
        if semantic_type == "semantic":
            self.semantic_module = SemanticModule1(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_2":
            self.semantic_module = SemanticModule2(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_3":
            self.semantic_module = SemanticModule3(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_4":
            self.semantic_module = SemanticModule4(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_5":
            self.semantic_module = SemanticModule5(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_6":
            self.semantic_module = SemanticModule6(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_7":
            self.semantic_module = SemanticModule7(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_8":
            self.semantic_module = SemanticModule8(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_attention":
            self.semantic_module = SemanticModuleAttention(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        elif semantic_type == "semantic_moe":
            self.semantic_module = SemanticModuleMOE(
                config,
                encoding_size=encoding_size,
                label_size=label_size,
                num_classes=num_classes,
            )
        else:
            logger.info(f"Semantic Type: {semantic_type} is not valid")

    def forward(self, x, y):
        encoding = self.encoder(x)
        out = self.semantic_module(encoding, y)
        return out


def create_model(
    config,
    semantic_type: str,
    num_classes: int,
    label_size: int = 10,
    load_pretrained: bool = True,
):

    encoder, encoding_size = get_encoder(config, load_pretrained)

    logger.info(
        f"Creating a {semantic_type} model with "
        f"num_classes={num_classes}"
        f", encoding_size={encoding_size}"
        f" and label_size={label_size}"
    )

    if semantic_type == "normal":
        model = RegularModel(config, encoder, encoding_size, num_classes)
    elif semantic_type.startswith("semantic"):
        model = CustomSemanticModel(
            config, encoder, semantic_type, encoding_size, num_classes, label_size
        )
    total_parameters = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: {:.2f}M".format(total_parameters / 1e6))

    return model

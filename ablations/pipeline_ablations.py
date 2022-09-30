import logging
import os
import time

import easydict

import main
from workspace import get_config, create_workspace_from_config

logger = logging.getLogger(__name__)


def cifar10_symmetric_ablation(current_dir: str, gpu: str):
    """Run all the CIFAR10 symmetric and asymmetric noise experiments"""

    experiments = (
        # ("asym-0.4", "semantic", "noise_generation"),
        ("asym-0.4", "semantic", "noise_weighted"),
        ("sym-0.2", "semantic", "noise_weighted"),
        ("sym-0.5", "semantic", "noise_weighted"),
        ("sym-0.8", "semantic", "noise_weighted"),
        ("sym-0.9", "semantic", "noise_weighted"),
        ("sym-0.0", "semantic", "noise_weighted"),
        ("asym-0.4", "normal", "class_weighted"),
        ("sym-0.2", "normal", "class_weighted"),
        ("sym-0.5", "normal", "class_weighted"),
        ("sym-0.8", "normal", "class_weighted"),
        ("sym-0.9", "normal", "class_weighted"),
        ("sym-0.0", "normal", "class_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/cifar10_sym.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}_{balancing_mode}"
        )

        config.gpus = gpu
        config.data.noise_type = noise_type
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def cifar10_pmd_ablation(current_dir: str, gpu: str):
    """Run all the CIFAR10 polynomial margin diminishing noise experiments"""

    experiments = (
        ("pmd-1-0.35", "normal", "class_weighted"),
        ("pmd-1-0.35", "semantic", "noise_weighted"),
        ("pmd-2-0.35", "normal", "class_weighted"),
        ("pmd-2-0.35", "semantic", "noise_weighted"),
        ("pmd-3-0.35", "normal", "class_weighted"),
        ("pmd-3-0.35", "semantic", "noise_weighted"),
        ("pmd-1-0.70", "normal", "class_weighted"),
        ("pmd-1-0.70", "semantic", "noise_weighted"),
        ("pmd-2-0.70", "normal", "class_weighted"),
        ("pmd-2-0.70", "semantic", "noise_weighted"),
        ("pmd-3-0.70", "normal", "class_weighted"),
        ("pmd-3-0.70", "semantic", "noise_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/cifar10_pmd.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}_{balancing_mode}"
        )

        config.gpus = gpu
        config.data.noise_type = noise_type
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def cifar10_rog_ablation(current_dir: str, gpu: str):
    """Run all the CIFAR10 ROG noise experiments"""

    experiments = (
        ("semantic_densenet", "normal", "class_weighted"),
        ("semantic_densenet", "semantic", "noise_weighted"),
        ("semantic_resnet", "normal", "class_weighted"),
        ("semantic_resnet", "semantic", "noise_weighted"),
        ("semantic_vgg", "normal", "class_weighted"),
        ("semantic_vgg", "semantic", "noise_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/cifar10_rog.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}_{balancing_mode}"
        )

        config.gpus = gpu
        config.data.noise_type = noise_type
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def cifar100_symmetric_ablation(current_dir: str, gpu: str):
    """Run all the CIFAR100 symmetric noise experiments"""

    experiments = (
        # ("sym-0.2", "semantic", "noise_generation"),
        ("sym-0.2", "semantic", "noise_weighted"),
        ("sym-0.5", "semantic", "noise_weighted"),
        ("sym-0.8", "semantic", "noise_weighted"),
        ("sym-0.9", "semantic", "noise_weighted"),
        ("sym-0.2", "normal", "class_weighted"),
        ("sym-0.5", "normal", "class_weighted"),
        ("sym-0.8", "normal", "class_weighted"),
        ("sym-0.9", "normal", "class_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/cifar100_sym.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}_{balancing_mode}"
        )

        config.gpus = gpu
        config.data.noise_type = noise_type
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def cifar100_pmd_ablation(current_dir: str, gpu: str):
    """Run all the CIFAR100 polynomial margin diminishing noise experiments"""

    experiments = (
        ("pmd-1-0.35", "normal", "class_weighted"),
        ("pmd-1-0.35", "semantic", "noise_weighted"),
        ("pmd-2-0.35", "normal", "class_weighted"),
        ("pmd-2-0.35", "semantic", "noise_weighted"),
        ("pmd-3-0.35", "normal", "class_weighted"),
        ("pmd-3-0.35", "semantic", "noise_weighted"),
        ("pmd-1-0.70", "normal", "class_weighted"),
        ("pmd-1-0.70", "semantic", "noise_weighted"),
        ("pmd-2-0.70", "normal", "class_weighted"),
        ("pmd-2-0.70", "semantic", "noise_weighted"),
        ("pmd-3-0.70", "normal", "class_weighted"),
        ("pmd-3-0.70", "semantic", "noise_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/cifar100_pmd.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}_{balancing_mode}"
        )

        config.gpus = gpu
        config.data.noise_type = noise_type
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def cifar100_rog_ablation(current_dir: str, gpu: str):
    """Run all the CIFAR100 ROG noise experiments"""

    experiments = (
        ("semantic_densenet", "normal", "class_weighted"),
        ("semantic_densenet", "semantic", "noise_weighted"),
        ("semantic_resnet", "normal", "class_weighted"),
        ("semantic_resnet", "semantic", "noise_weighted"),
        ("semantic_vgg", "normal", "class_weighted"),
        ("semantic_vgg", "semantic", "noise_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/cifar100_rog.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}_{balancing_mode}"
        )

        config.gpus = gpu
        config.data.noise_type = noise_type
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def animal10n_ablation(current_dir: str, gpu: str):
    """Run all the Animal10N experiments"""

    experiments = (("semantic", "noise_weighted"), ("normal", "class_weighted"))

    # Fetch the base config
    config_file_name = "configs/ablations/animal10n.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(parent_save_dir, f"{semantic_type}")

        config.gpus = gpu
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def webvision_ablation(current_dir: str, gpu: str):
    """Run all the Webvision experiments"""

    experiments = (
        ("normal", "class_weighted"),
        ("semantic", "noise_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/webvision.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(parent_save_dir, f"{semantic_type}")

        config.gpus = gpu
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def red_blue_ablation(current_dir: str, gpu: str):
    """Run all the Red/Blue Controlled Web Noise experiments"""

    experiments = (
        ("0.2", "semantic", "noise_weighted"),
        ("0.4", "semantic", "noise_weighted"),
        ("0.6", "semantic", "noise_weighted"),
        ("0.8", "semantic", "noise_weighted"),
        ("0.2", "normal", "class_weighted"),
        ("0.4", "normal", "class_weighted"),
        ("0.6", "normal", "class_weighted"),
        ("0.8", "normal", "class_weighted"),
    )

    # Fetch the base config
    config_file_name = "configs/ablations/red_blue.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for noise_type, semantic_type, balancing_mode in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{noise_type}_{semantic_type}"
        )

        config.gpus = gpu
        config.data.noise_type = f"labels_{noise_type}"
        config.data.dataset = f"red_blue_{noise_type}"
        config.ds_partition.balancing_mode = balancing_mode
        config.model.semantic_type = semantic_type

        config = create_workspace_from_config(config)

        main.main(config)


def null_label_ablation(current_dir: str, gpu: str):
    """Run experiments with different types of `null` label, to test whether using zero
    labels to represent null labels is the best choice"""

    experiments = ("zeros", "ones", "average")

    # Fetch the base config
    config_file_name = "configs/ablations/cifar10_sym.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for null_label_type in experiments:

        config.result_dir = os.path.join(parent_save_dir, f"{null_label_type}")

        config.gpus = gpu
        config.null_label_type = null_label_type

        config = create_workspace_from_config(config)

        main.main(config)


def model_ablation(current_dir: str, gpu: str):
    """Experiment with using concatenation, mixture of experts and scaled dot product
    attention models"""

    experiments = ("semantic_moe", "semantic_attention", "semantic")

    # Fetch the base config
    config_file_name = "configs/ablations/cifar10_sym.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for model_type in experiments:

        config.result_dir = os.path.join(parent_save_dir, f"{model_type}")

        config.gpus = gpu
        config.model.semantic_type = model_type

        config = create_workspace_from_config(config)

        main.main(config)


def label_ablation(current_dir: str, gpu: str):
    """Run a test on whether generating labels for testing is preferable to using null
    labels"""

    config_file_name = "configs/ablations/cifar10_sym.yaml"

    # Fetch the base config
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")

    # Tweak the config, create the workspace and run the experiment
    config.gpus = gpu
    config = create_workspace_from_config(config)
    main.main(config)


def null_nls_for_pseudo_generation_ablation(current_dir: str, gpu: str):
    """Run tests on whether dropping null labels for pseudolabel generation during SSL
    improves final accuracy"""

    experiments = (False, True)

    # Fetch the base config
    config_file_name = "configs/ablations/cifar10_sym.yaml"
    config = get_config(current_dir, config_file_name)
    config = easydict.EasyDict(config)
    config.current_dir = current_dir
    logger.info(f"Using config from {config_file_name}")
    parent_save_dir = time.strftime(config.result_dir, time.localtime())

    # For each experiment, tweak the config, create the workspace and run the experiment
    for null_nls_for_pseudo_generation in experiments:

        config.result_dir = os.path.join(
            parent_save_dir, f"{null_nls_for_pseudo_generation}"
        )

        config.gpus = gpu
        config.ssl.null_nls_for_pseudo_generation = null_nls_for_pseudo_generation

        config = create_workspace_from_config(config)

        main.main(config)

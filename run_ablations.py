import os
import sys

import ablations.pipeline_ablations as pipeline_ablations

if __name__ == "__main__":

    # Load the config and create the results folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gpus = sys.argv[2]

    if sys.argv[1] == 'setup_cifar100_datasets':
        import datasets.cifar100.generate_datasets as generate_cifar100_datasets
        generate_cifar100_datasets.main()

    elif sys.argv[1] == 'setup_animal10n_datasets':
        import datasets.animal10n.generate_datasets as generate_animal10n_datasets
        generate_animal10n_datasets.main()

    elif sys.argv[1] == 'setup_red_blue_datasets':
        import datasets.red_blue.generate_datasets as generate_red_blue_datasets
        generate_red_blue_datasets.main()

    elif sys.argv[1] == 'setup_webvision_dataset':
        import datasets.webvision.generate_datasets as generate_webvision_datasets
        generate_webvision_datasets.main()

    elif sys.argv[1] == 'cifar10_sym':
        pipeline_ablations.cifar10_symmetric_ablation(current_dir, gpus)

    elif sys.argv[1] == 'cifar10_rog':
        pipeline_ablations.cifar10_rog_ablation(current_dir, gpus)

    elif sys.argv[1] == 'cifar10_pmd':
        pipeline_ablations.cifar10_pmd_ablation(current_dir, gpus)

    elif sys.argv[1] == 'cifar100_sym':
        pipeline_ablations.cifar100_symmetric_ablation(current_dir, gpus)
    
    elif sys.argv[1] == 'cifar100_rog':
        pipeline_ablations.cifar100_rog_ablation(current_dir, gpus)

    elif sys.argv[1] == 'cifar100_pmd':
        pipeline_ablations.cifar100_pmd_ablation(current_dir, gpus)

    elif sys.argv[1] == 'animal10n':
        pipeline_ablations.animal10n_ablation(current_dir, gpus)

    elif sys.argv[1] == 'webvision':
        pipeline_ablations.webvision_ablation(current_dir, gpus)

    elif sys.argv[1] == 'red_blue':
        pipeline_ablations.red_blue_ablation(current_dir, gpus)
    
    elif sys.argv[1] == 'null_label_ablation':
        pipeline_ablations.null_label_ablation(current_dir, gpus)

    elif sys.argv[1] == 'model_ablation':
        pipeline_ablations.model_ablation(current_dir, gpus)

    elif sys.argv[1] == "label_ablation":
        pipeline_ablations.label_ablation(current_dir, gpus)

    elif sys.argv[1] == "null_nls_for_pseudo_generation":
        pipeline_ablations.null_nls_for_pseudo_generation_ablation(current_dir, gpus)

    else:
        print(f"Ablation name '{sys.argv[1]}' not recognised")
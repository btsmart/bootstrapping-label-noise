import datasets.cifar10.cifar10 as cifar10
import datasets.cifar100.cifar100 as cifar100
import datasets.animal10n.animal10n as animal10n
import datasets.webvision.webvision as webvision
import datasets.red_blue.red_blue as red_blue
import datasets.ilsvrc2012.ilsvrc2012 as ilsvrc2012


def get_class_name(ds_name, class_idx):

    if ds_name == "cifar10":
        return cifar10.cifar10_class_name(class_idx)

    elif ds_name == "cifar100":
        return cifar100.cifar100_class_name(class_idx)

    elif ds_name == "animal10n":
        return animal10n.animal10n_class_name(class_idx)

    elif ds_name == "webvision":
        return webvision.webvision_class_name(class_idx)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_class_name(class_idx, noise)

    else:
        raise RuntimeError(f"Tried to fetch class name for unknown dataset: {ds_name}")


def load_dataset(ds_name, ds_type):

    if ds_name == "cifar10":
        return cifar10.cifar10_load_dataset(ds_type)

    elif ds_name == "cifar100":
        return cifar100.cifar100_load_dataset(ds_type)

    elif ds_name == "animal10n":
        return animal10n.animal10n_load_dataset(ds_type)

    elif ds_name == "webvision":
        return webvision.webvision_load_dataset(ds_type)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_load_dataset(ds_type, noise)

    else:
        raise RuntimeError(f"Tried to load unknown dataset: {ds_name}")


def get_bootstrapping_datasets(ds_name, ds, train_aug):

    if ds_name == "cifar10":
        return cifar10.cifar10_bootstrapping_datasets(ds, train_aug)

    elif ds_name == "cifar100":
        return cifar100.cifar100_bootstrapping_datasets(ds, train_aug)

    elif ds_name == "animal10n":
        return animal10n.animal10n_bootstrapping_datasets(ds, train_aug)

    elif ds_name == "webvision":
        return webvision.webvision_bootstrapping_datasets(ds, train_aug)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_bootstrapping_datasets(ds, train_aug, noise)

    else:
        raise RuntimeError(
            f"Tried to fetch Bootstrapping Datasets for unknown dataset: {ds_name}"
        )


def get_split_dataset_datasets(ds_name, ds, eval_aug):

    if ds_name == "cifar10":
        return cifar10.cifar10_split_dataset_datasets(ds, eval_aug)

    elif ds_name == "cifar100":
        return cifar100.cifar100_split_dataset_datasets(ds, eval_aug)

    elif ds_name == "animal10n":
        return animal10n.animal10n_split_dataset_datasets(ds, eval_aug)

    elif ds_name == "webvision":
        return webvision.webvision_split_dataset_datasets(ds, eval_aug)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_split_dataset_datasets(ds, eval_aug, noise)

    else:
        raise RuntimeError(
            f"Tried to fetch Split Dataset Datasets for unknown dataset: {ds_name}"
        )


def get_test_aug_datasets(ds_name):

    if ds_name == "cifar10":
        return cifar10.cifar10_test_aug_datasets()

    elif ds_name == "cifar100":
        return cifar100.cifar100_test_aug_datasets()

    elif ds_name == "animal10n":
        return animal10n.animal10n_test_aug_datasets()

    elif ds_name == "webvision":
        return webvision.webvision_test_aug_datasets()

    elif ds_name == "ilsvrc2012":
        return ilsvrc2012.imagenet_test_aug_datasets()

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_test_aug_datasets(noise)

    else:
        raise RuntimeError(
            f"Tried to fetch Test Aug Dataset Datasets for unknown dataset: {ds_name}"
        )


def get_final_datasets(ds_name, ds):

    if ds_name == "cifar10":
        return cifar10.cifar10_final_datasets(ds)

    elif ds_name == "cifar100":
        return cifar100.cifar100_final_datasets(ds)

    elif ds_name == "animal10n":
        return animal10n.animal10n_final_datasets(ds)

    elif ds_name == "webvision":
        return webvision.webvision_final_datasets(ds)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_final_datasets(ds, noise)

    else:
        raise RuntimeError(
            f"Tried to fetch Final Datasets for unknown dataset: {ds_name}"
        )


def get_ssl_datasets(ds_name, clean_ds, noisy_ds):

    if ds_name == "cifar10":
        return cifar10.cifar10_ssl_datasets(clean_ds, noisy_ds)

    elif ds_name == "cifar100":
        return cifar100.cifar100_ssl_datasets(clean_ds, noisy_ds)

    elif ds_name == "animal10n":
        return animal10n.animal10n_ssl_datasets(clean_ds, noisy_ds)

    elif ds_name == "webvision":
        return webvision.webvision_ssl_datasets(clean_ds, noisy_ds)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_ssl_datasets(clean_ds, noisy_ds, noise)

    else:
        raise RuntimeError(
            f"Tried to fetch SSL datasets for unknown dataset: {ds_name}"
        )


def get_pretraining_dataset(ds_name, ds):

    if ds_name == "cifar10":
        return cifar10.cifar10_pretraining_dataset(ds)

    elif ds_name == "cifar100":
        return cifar100.cifar100_pretraining_dataset(ds)

    elif ds_name == "animal10n":
        return animal10n.animal10n_pretraining_dataset(ds)

    elif ds_name == "webvision":
        return webvision.webvision_pretraining_dataset(ds)

    elif ds_name.startswith("red_blue"):
        noise = ds_name[9:]
        return red_blue.red_blue_pretraining_dataset(ds, noise)

    else:
        raise RuntimeError(
            f"Tried to fetch Pretraining datasets for unknown dataset: {ds_name}"
        )

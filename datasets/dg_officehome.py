
import os.path as osp
import glob
import random

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing



@DATASET_REGISTRY.register()
class DGOfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - 4 domains: Art, Clipart, Product, Real world.
        - 65 categories.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
        - Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_x = self._read_data_test(cfg.DATASET.SOURCE_DOMAINS, "train")
        val = self._read_data_test(cfg.DATASET.SOURCE_DOMAINS, "val")
        test = self._read_data_test(cfg.DATASET.TARGET_DOMAINS, "all")
        super().__init__(train_x=train_x, val=val, test=test)

    def _read_data_train(self, input_domains, split, num_labeled):
        items_x, items_u = [], []
        num_labeled_per_class = None
        num_domains = len(input_domains)

        for domain, dname in enumerate(input_domains):
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path, sort=True)

            if num_labeled_per_class is None:
                num_labeled_per_class = num_labeled / (num_domains * len(folders))

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))
                assert len(impaths) >= num_labeled_per_class
                random.shuffle(impaths)

                for i, impath in enumerate(impaths):
                    item = Datum(impath=impath, label=label, domain=domain)
                    if (i + 1) <= num_labeled_per_class:
                        items_x.append(item)
                    else:
                        items_u.append(item)

        return items_x, items_u

    def _read_data_test(self, input_domains, split):
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory, sort=True)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(self.dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items

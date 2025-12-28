from dataset.index_anno import AnnoIndexedDataset
from dataset.index_src import SrcIndexedDataset

dataset_registry = {
    "annoindexed": AnnoIndexedDataset,
    "srcindexed": SrcIndexedDataset,
}

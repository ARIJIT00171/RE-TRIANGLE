from dataset_utils.index_anno import AnnoIndexedDataset
from dataset_utils.index_src import SrcIndexedDataset

dataset_registry = {
    "annoindexed": AnnoIndexedDataset,
    "srcindexed": SrcIndexedDataset,
}


from tabulate import tabulate
from datasets.metacatalog import MetadataCatalog, DatasetCatalog


def check_all_register_dataset():
    table_header = ['Dataset Name', "MetaData"]

    dataset_items = DatasetCatalog.list()

    total_registered_dataset_with_meta_data = []
    for dataset_it in dataset_items:
        total_registered_dataset_with_meta_data.append((dataset_it, MetadataCatalog.get(dataset_it)))

    print(tabulate(total_registered_dataset_with_meta_data, table_header, tablefmt='grid'))

if __name__ == '__main__':
    check_all_register_dataset()
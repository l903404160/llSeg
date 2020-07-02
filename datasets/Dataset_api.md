# How to use costum dataset in this framework:

## Step 1: Register your dataset in `DatasetCatalog` and `MetadataCatalog`:
1. Find the corresponding folder for your tasks. (e.g. `CityScapes` is registered in `segmentation` )
2. Specify the properties for your dataset like `stuff_classes` and `thing_classes`. Then add above properties into the metadata of your dataset in `/datasets/<task>/<task>_builtin_meta.py`
    > Tips: You can add any necessary properties into the metadata of your dataset. For example, `Detectron2` also registers the color of each class in the metadata.  
3. Add registration function in `/datasets/<task>/register.py`. 
    > Tips: Here, the registration function is just a map function. The main purpose of it is to separate the registration and `Dataset` object generation. 
4. (The final step) Complete the final registration in the `/datasets/<task>/<task>_builtin.py`

## Step 2: Complete the function of data loading
The data loading function is required to return a `List` which contains all data pair. Each item of the `List` should be formated by the following format:
```python
data_dict = [
        {
        'file_name': image_file,
        'sem_seg_file_name': label_file,
        'height': h,
        'width': w,
        # ... 
    }
]
``` 
A sample loading function is placed in `/datasets/segmentation/cityscapes.py`

## Step 3: Test the corresponding dataset is registered or not.

Because the registration and data loading are separated in two stream. Thus we are able to test the two part respectively.

### Test part 1: Data Loading Function
1. Check the function is return the correct type of data or not
2. Check the necessary information of your dataset is loaded in each dict of the returned variable or not

### Test part 2: Dataset Registration
1. Check the corresponding dataset is registered in the `DatasetCatalog` and `MetadataCatalog` or not.
2. Check the registered dataset is able to return the same variable of the `Data Loading Function` or not.
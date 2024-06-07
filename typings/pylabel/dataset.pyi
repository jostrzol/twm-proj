import pandas as pd

"""
This type stub file was generated by pyright.
"""

"""The dataset is the primary object that you will interactive with when using PyLabel.
All other modules are sub-modules of the dataset object. 
"""

class Dataset:
    def __init__(self, df) -> None: ...
    def ReindexCatIds(self, cat_id_index=...):  # -> None:
        """
        Reindex the values of the cat_id column so that that they start from an int (usually 0 or 1) and
        then increment the cat_ids to index + number of categories.
        It's useful if the cat_ids are not continuous, especially for dataset subsets,
        or combined multiple datasets. Some models like Yolo require starting from 0 and others
        like Detectron require starting from 1.

        Args:
            cat_id_index (int): Defaults to 0.
                The cat ids will increment sequentially the cat_index value. For example if there are 10
                classes then the cat_ids will be a range from 0-9.

        Example:
            >>> dataset.analyze.class_ids
                [1,2,4,5,6,7,8,9,11,12]
            >>> dataset.ReindexCatIds(cat_id_index) = 0
            >>> dataset.analyze.class_ids
                [0,1,2,3,4,5,6,7,8,9]
        """
        ...
    df: pd.DataFrame

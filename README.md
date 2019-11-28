## classification-utils

Helper functions for text classification with BERT.

### Data preparation

Read data and standardize the format with `./run python prepare.py DATA_ID`

Multiple labels are split, sorted and represented with a one-hot encoding.
The processed data is stored as _.pkl_ file.

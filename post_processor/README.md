The post-processing scripts in this repository are organized into folders based on the Document AI version they are
compatible with.

Short descriptions of the scripts in each folder are provided below.

- `post_processor_1.py`: Merges tokens and creates a flat JSON output.
- `post_processor_2.py`: Merges tokens, extracts line items from a supply-chain document.
- `post_processor_3.py`: Modified version of post_processor_2.py with template method for entity extraction.
- `post_processor_4.py`: Outputs all document information in flat JSON, convertible to CSV.
- `post_processor_5.py`: Extracts image snippets of entities, provides top-n answers per class.
- `post_processor_6.py`: Offers empty answers for unidentified model classes in supply-chain documents.
- `post_processor_7.py`: Offers empty answers for unidentified model classes in general documents.
- `post_processor_8.py`: Accepts template dicts for post-processing.
- `post_processor_9.py`: Returns redacted image.
- `post_processor_10.py`: For ocr-only pipeine, outputs all texts.
- `post_processor_11.py`: Add signature detection result to the final output



Parameters in `post_process_predictions` function that can be adjust to fit your own use cases

- model_preds: pandas DataFrame
        A DataFrame containing the model predictions, with columns for the document ID, page ID, coordinates,
        text content, label, and OCR confidence.
- top_n_preds: list of str
        A list of the predicted label names for the model.
- token_merge_type: str, optional (default='MIXED_MERGE')
        The type of token merging to use, either 'MIXED_MERGE' or 'NO_MERGE'.
- token_merge_xdist_regular: float, optional (default=1.0)
        The distance threshold for merging tokens on the x-axis, between 0.0 and 1.0.
- label_merge_x_regular: str or None, optional (default=None)
        The label names to merge on the x-axis, either 'ALL' or a string of labels separated by '|'.
- token_merge_xydist_regular: float, optional (default=1.0)
        The distance threshold for merging tokens on the x- and y-axes, between 0.0 and 1.0.
- label_merge_xy_regular: str or None, optional (default=None)
        The label names to merge on the x- and y-axes, either 'ALL' or a string of labels separated by '|'.
- token_merge_xdist_wide: float, optional (default=1.5)
        The distance threshold for wide token merging on the x-axis, between 1.0 and 10.0.
- label_merge_x_wide: str or None, optional (default=None)
        The label names to merge for wide token merging on the x-axis, either 'ALL' or a string of labels separated by '|'.
- output_labels: str or list, optional (default='INCLUDE_O')
        The type of output labels to include. Options are 'INCLUDE_O' (include all labels), 'EXCLUDE_O' (exclude "O" labels),
        or a list of label names to include.
- parse_line_items: bool, optional (default=False)
        Whether to parse line items from the input data.
- line_item_completeness: float, optional (default=0.6)
        The completeness threshold for parsing line items, between 0.0 and 1.0.
- try_templates: bool, optional (default=False)
        Whether to try extracting data using templates.
- templates_dict_dir: str, optional (default='')
        The directory path to the template configuration files.
- templates_input_dir: str, optional (default='./')
        The directory path to the input files for templates.
- templates_use_model_preds_mapping: dict, option (default={})
        A dictionary of similar labels to be used when `use_model_preds=True`. This is useful for properly parsing name/address fields. For example, if the templates specify the `receiver_address` is in a particular location, but `use_model_preds=True` and the model predicts the text in that location as the `billing_address` we will convert the `billing_address` text to be the `receiver_address`.
- use_camelot_tables: bool, optional (default=False)
        Whether to use Camelot tables for extracting line items.
- images_dir_camelot: str, optional (default='')
        The directory path to the images for Camelot tables.

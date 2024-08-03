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

For v0.7 and later versions:

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
 
 
 
For v0.6, the function named `post_process_via_predictions`

- templates_dict_dir: folder name under argus_contrib
- try_templates: try templates method or not
- input_dir: absolute  directory oof input pdfs
- via_predictions: prediction via.json from the model
- probabilities: probabilities top_n.json from the model
- token_merge_type: how to merge tokens (along x-y axis) of same label.
    Options are "x", "xy", "mixed" and None. default: "mixed"
- token_merge_threshold_x: threshold for token merging distance in horizontal direction.
    tokens would be merged if their distance is less than [mean_token_height * this value]. range: [0, 1]  default: 0.33
- token_merge_threshold_xy: threshold for token merging distance in both horizontal and vertical directions.
    tokens would be merged if their distance is less than [mean_token_height * this value]. range: [0, 1]  default: 0.5
- labels_to_merge_x: regex of labels to be merged only in horizontal direction. default: "", which means merge
    all nearby same label tokens in x-direction.
- labels_to_merge_x_long_range: regex of labels to be merged only in horizontal direction that are
    far apart from each other (max dist: 1*median char height). default: "phone|fax",
- labels_to_merge_xy: regex of labels to be merged in both horizontal and vertical directions.
    if label contains any part of this string, it will be merged. default: "address|name|comment|note".
    The value "" means merge all nearby same label tokens in both x and y-direction.
- parse_line_items: whether to parse line items or not. default: False
- output_labels: what labels to output. Options are "FULL", "BD", "BSMH". FULL means output all training labels.
    default: "FULL"
- line_th: range 0.0-1.0. default:0.6
- output_cleaning_method: what cleaning method to use. Options are "BD" and "NONE". default: "NONE"
- try_templates: Try using the template method.
- template_dict_dir: Directory containing the template config files.
- images_dir: directory containing the images of the pages
- use_camelot_tables: Use camelot to find tables and parse line items

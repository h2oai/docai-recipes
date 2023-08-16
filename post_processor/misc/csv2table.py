import os
import pandas as pd


def tag_line_items(df, configs):
    """
    Labels rows in the dataframe based on their association with anchor items.

    This function processes the input dataframe to identify line items based on
    the provided configurations. The function checks the presence of the minimum
    number of items on the same line as specified in the configuration.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing prediction results.
      Expected columns in the dataframe include 'label', 'page_id', 'ymin', and 'ymax',
      among others.

    - configs (list of dict): A list of configurations where each configuration is a dictionary
      containing the following keys:
      - 'item_classes' (list of str): Labels that should be present on the same line.
      - 'anchor_class' (str): The label used to identify and anchor a line.
      - 'table_id' (str): Identifier for the type of table or section.
      - 'min_items' (int): Minimum number of 'item_classes' labels that should be present on the
        same line to be considered valid.
      (Optional configurations can be added as needed.)

    Returns:
    - pandas.DataFrame: A modified dataframe with additional columns 'line' and 'table_id'
      indicating the line number and table type for each row, respectively.

    Example:
    df = pd.DataFrame({
        'label': ['payment', 'deduction', 'payment_type', 'deduction_type'],
        'page_id': [1, 1, 1, 1],
        'ymin': [10, 20, 10, 20],
        'ymax': [15, 25, 15, 25]
     })
    configs = [
        {
            'item_classes': ['payment_type', 'payment'],
            'anchor_class': 'payment',
            'table_id': 'payable',
            'min_items': 2,
        }
     ]
    tagged_df = tag_line_items(df, configs)
    """

    if 'line' not in df.columns:
        df['line'] = None
    if 'table_id' not in df.columns:
        df['table_id'] = None

    for config in configs:
        line_id = 1
        anchors = df[df['label'].isin([config['anchor_class']])]

        # Iterate through each unique page_id
        for page in df['page_id'].unique():
            page_anchors = anchors[anchors['page_id'] == page]

            for _, anchor_row in page_anchors.iterrows():
                # Identify the row based on ymin of the anchor
                same_row_indices = df[(df['page_id'] == page) & (df['ymin'] == anchor_row['ymin'])].index

                # Check if at least min_line_items are found
                line_items_found = sum(df.loc[same_row_indices, 'label'].isin(config['item_classes']))

                if line_items_found >= config['min_items']:
                    df.loc[same_row_indices, 'line'] = line_id
                    df.loc[same_row_indices, 'table_id'] = config['table_id']
                    line_id += 1

    return df


def process_csv_files(directory_path):
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(directory_path, csv_file))

        configurations = [
            {
                'item_classes': ['payment_type', 'payment'],
                'anchor_class': 'payment',
                'table_id': 'payment',
                'min_items': 2,
            },
            {
                'item_classes': ['deduction_type', 'deduction'],
                'anchor_class': 'deduction',
                'table_id': 'deduction',
                'min_items': 2,
            },
        ]

        tagged_df = tag_line_items(df, configurations)
        print(f"Processed contents of {csv_file}:")
        print(tagged_df)


process_csv_files('data/preds_csv')

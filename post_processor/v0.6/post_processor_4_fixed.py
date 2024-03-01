import os
import gc
import psutil
import traceback
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, Any

from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors.post_processor_supply_chain import PostProcessor as PostProcessorSupplyChain


def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_rss = memory_info.rss / (1024**2)  # Convert bytes to MB
    memory_usage_vms = memory_info.vms / (1024**2)  # Convert bytes to MB
    print(
        f"Current Memory Usage: RSS = {memory_usage_rss:.2f} MB, VMS = {memory_usage_vms:.2f} MB"
    )


def patched_via2df(via_predictions, probabilities):
    model_dir = os.environ.get("DOCAI_LABEL_MODEL_DIR", "microsoft/layoutlm-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    entity_data = []
    prob_data = []

    base_columns = [
        "doc_id",
        "page_id",
        "x",
        "y",
        "width",
        "height",
        "text",
        "label",
        "ocr_confidence",
    ]
    df_all_data = pd.DataFrame(columns=base_columns)

    class_names = probabilities.get("class_names", [])
    pp.log.info(f"Class names: {class_names}")

    for key, value in tqdm(via_predictions.get("_via_img_metadata", {}).items()):
        filename = value.get("filename")
        doc_id, _page_num = filename.rsplit("+", maxsplit=1)
        page_num = _page_num[:-4]
        page_id = f"{doc_id}+{page_num}"
        regions = value.get("regions", [])
        pp.log.info(f"Processing page: {page_id}")
        pp.log.info(f"New Regions: {len(regions)}")

        try:
            # Make a matrix of probabilities from top_n
            top_n = probabilities.get("top_n", {}).get(page_id, {})
            for class_ids, probs in zip(
                    top_n.get("class_ids", []), top_n.get("probability", [])
            ):
                row = [0] * len(class_names)
                for i, prob in zip(class_ids, probs):
                    row[i] = prob
                prob_data.append(row)

            pp.log.info(f"Probabilities matrix: {len(prob_data)}")

            # Extract entities
            for region in regions:
                shape_attr = region.get("shape_attributes", {})
                x, y, width, height = (
                    shape_attr.get("x"),
                    shape_attr.get("y"),
                    shape_attr.get("width"),
                    shape_attr.get("height"),
                )
                region_attr = region.get("region_attributes", {})
                text, label, ocr_conf = (
                    region_attr.get("text"),
                    region_attr.get("label", ""),
                    region_attr.get("confidence", 0),
                )

                token_count = len(tokenizer.tokenize(text))
                # Skip empty tokens
                if token_count == 0:
                    continue

                entity_data.append(
                    {
                        "doc_id": doc_id,
                        "page_id": page_num,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "text": text,
                        "label": label,
                        "ocr_confidence": ocr_conf,
                    }
                )
            pp.log.info(f"Entities: {len(entity_data)}")

        except Exception as e:
            traceback.print_exc()
            pp.log.info("Error in via2df, skipping page...")
            pp.log.info(e)
            continue

    # Concatenate entity and probability data
    if prob_data:
        try:
            pp.log.info("Concatenating lists into dataframe...")
            df_prob = pd.DataFrame(prob_data, columns=class_names)
            df_entity = pd.DataFrame(entity_data, columns=base_columns)
            df_all_data = pd.concat([df_entity, df_prob], axis=1)
            pp.log.info("Dataframes concatenated successfully.")
            pp.log.info(f"Dataframe shape: {df_all_data.shape}")
        except Exception as e:
            pp.log.info("Error in entity-probability merging, returning empty dataframe...")
            pp.log.info(e)

    pp.log.info(f"Dataframe top 10: {df_all_data.head(10)}")
    return df_all_data, class_names


pp.via2df = patched_via2df


class PostProcessor(PostProcessorSupplyChain):
    """Represents a last step in pipeline process that receives all pipeline intermediate
    results and translates them into a final json structure that will be returned to user.
    """

    def get_pages(self) -> Dict[int, Any]:
        return {}

    def get_entities(self):

        log_memory_usage()
        print("deleting all images")
        del self.images
        gc.collect()
        print("all images are deleted")
        log_memory_usage()

        if not self.has_labelling_model:
            return []

        docs = pp.post_process_predictions(
            model_preds=self.label_via_predictions,
            top_n_preds=self.label_top_n,
            token_merge_type="MIXED_MERGE",
            token_merge_xdist_regular=1.0,
            label_merge_x_regular="ALL",
            token_merge_xydist_regular=1.0,
            label_merge_xy_regular="address",
            token_merge_xdist_wide=1.5,
            label_merge_x_wide="phone|fax",
            output_labels="INCLUDE_O",  # Use INCLUDE_O, then remove unlabeled tokens later
            verbose=True,
        )

        df_list = []
        for doc in docs:
            predictions = docs[doc]
            predictions = predictions.round(decimals=4)
            for idx, row in predictions.iterrows():
                # Remove unlabeled tokens
                if row["label"] == "":
                    continue
                df_list.append(row.to_dict())
        pp.log.info(f"Labeled Entities: {len(df_list)}")
        return df_list

    '''
    Converting the dictionary to a dataframe
    import pandas as pd
    import json

    f = open('result.json')
    dict_data = json.load(f)
    df = pd.DataFrame(dict_data['entities'])
    df.to_csv('result.csv')
    '''


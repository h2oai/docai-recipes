"""
This script works with DccAI v0.5.*
It provides potential redaction of specific image regions based on given labels and/or text patterns.
"""
import re
import json
import uuid
import base64
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from argus_contrib.utils import post_process as pp
# from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor, BaseEntity

"""
User input: labels/text patterns to be redacted.
Valid values:
labels_to_redact = None
labels_to_redact = ['billing_name', 'billing_address']  # Must be model class labels
text_patterns_to_redact = None
text_patterns_to_redact = [r'[A-Za-z]{3}\d{6}', r'\d{3}-\d{2}-\d{4}', r'\b\d{11}\b', r'\b\d{11}\b', r"\b\d{3}-\d{3}-\d{4}\b"]
"""

labels_to_redact = None
text_patterns_to_redact = [r"^\d{3} \d{3} \d{3}$", r'\b\d{9}\b']   # regex pattern: 111 222 333, 111222333


class CustomEntity(BaseEntity):
    image: str


class PostProcessor(BasePostProcessor):
    """
    A post-processor class for processing intermediate results and
    translating them into a final JSON structure for user output.
    """

    def client_resolution(self) -> None:
        return None

    def argus_resolution(self) -> int:
        return self.ARGUS_DPI

    def has_text_tokens(self, via_predictions: dict) -> bool:
        if not via_predictions:
            return False

        text_values = []
        img_metadata = via_predictions.get('_via_img_metadata', {})
        for key, value in img_metadata.items():
            regions = value.get('regions', [])
            for region in regions:
                text = region['region_attributes'].get('text', None)
                if text is not None:
                    text_values.append(str(text))
        joined_text = "".join(text_values)
        return len(joined_text) > 0

    @staticmethod
    def redact_region(image: np.array, xmin: int, ymin: int, xmax: int, ymax: int) -> np.array:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
        return image

    def get_filtered_predictions(self, doc_predictions) -> pd.DataFrame:
        doc_predictions['text'] = doc_predictions['text'].str.strip()

        # Filter by labels
        label_filtered = doc_predictions[
            doc_predictions['label'].isin(labels_to_redact)] if labels_to_redact else pd.DataFrame()

        # Filter by text patterns
        pattern_filtered = pd.concat([
            doc_predictions[doc_predictions['text'].str.contains(pattern, regex=True)]
            for pattern in text_patterns_to_redact
        ]).drop_duplicates() if text_patterns_to_redact else pd.DataFrame()

        return pd.concat([label_filtered, pattern_filtered]).drop_duplicates()

    def get_entities(self) -> List[CustomEntity]:

        if not self.has_labelling_model or not self.has_text_tokens(self.label_via_predictions):
            return []

        docs = pp.post_process_via_predictions(input_dir=self.input_dir,
                                               via_predictions=self.label_via_predictions,
                                               probabilities=self.label_top_n,
                                               token_merge_type='mixed',
                                               labels_to_merge_x="",
                                               labels_to_merge_x_long_range="",
                                               labels_to_merge_xy="address",
                                               output_labels='ALL',  # 'ALL_TOKENS including O tokens
                                               token_merge_threshold_x=0.5,
                                               token_merge_threshold_xy=0.33,
                                               try_templates=False,
                                               )

        redacted_images_per_page = {}
        redacted_text_per_page = {}

        for doc in docs:
            docs[doc]["id"] = docs[doc]["label"].apply(lambda row: str(uuid.uuid4()))

            redacted_images_per_page = {
                f"{doc}+{row['page_id']}{self.img_extension}": self.images[
                    f"{doc}+{row['page_id']}{self.img_extension}"].copy()
                for doc in docs for _, row in docs[doc].iterrows()
            }

            filtered_predictions = self.get_filtered_predictions(docs[doc])

            for _, row in filtered_predictions.iterrows():
                filename = f"{doc}+{row['page_id']}{self.img_extension}"
                self.redact_region(
                    redacted_images_per_page[filename],
                    int(row["xmin"]), int(row["ymin"]),
                    int(row["xmax"]), int(row["ymax"])
                )

                page_id = row['page_id']
                text_to_redact = row['text']

                # Concatenate redacted text for each page
                if page_id in redacted_text_per_page:
                    redacted_text_per_page[page_id] += " | " + text_to_redact
                else:
                    redacted_text_per_page[page_id] = text_to_redact

        redacted_data_bundles = []
        for filename, redacted_image in redacted_images_per_page.items():
            _, img_encoded = cv2.imencode(".png", redacted_image)
            page_index = filename.split('+')[1].split(self.img_extension)[0]

            # Get the concatenated redacted text for this page
            concatenated_redacted_text = redacted_text_per_page.get(page_index, "")

            data_bundle: CustomEntity = {
                "pageIndex": page_index,
                "redactedData": concatenated_redacted_text, 
                "image": base64.b64encode(img_encoded).decode("ascii")
            }
            redacted_data_bundles.append(data_bundle)

        return redacted_data_bundles

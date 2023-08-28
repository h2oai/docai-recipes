"""
This script works with DccAI v0.6.*
It provides potential redaction of specific image regions based on given labels and/or text patterns.
"""
import re
import uuid
import base64
import cv2
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor

"""
User input: labels/text patterns to be redacted.
Valid values:
labels_to_redact = None
labels_to_redact = ['billing_name', 'billing_address']  # Must be model class labels
text_patterns_to_redact = None
text_patterns_to_redact = [r'[A-Za-z]{3}\d{6}', r'\d{3}-\d{2}-\d{4}', r'\b\d{11}\b']
"""

labels_to_redact = None
text_patterns_to_redact = [r"^\d{3} \d{3} \d{3}$", r'\b\d{9}\b']  # regex pattern: 111 222 333, 111222333


class PostProcessor(BasePostProcessor):
    """
    A post-processor class for processing intermediate results and
    translating them into a final JSON structure for user output.
    """

    def client_resolution(self) -> None:
        return None

    def argus_resolution(self) -> int:
        return self.ARGUS_DPI
    @staticmethod
    def has_text_tokens(via_predictions: dict) -> bool:
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

    def get_pages(self) -> Dict[int, Any]:
        return super().get_pages()

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

    def get_entities(self) -> List[dict]:

        if not self.has_labelling_model or not self.has_text_tokens(self.label_via_predictions):
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
            output_labels="INCLUDE_O",
            verbose=True,
        )

        redacted_images_per_page = {}
        redacted_data_bundles = []

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

        for filename, redacted_image in redacted_images_per_page.items():
            _, img_encoded = cv2.imencode(".png", redacted_image)
            data_bundle = {
                "pageIndex": filename.split('+')[1].split(self.img_extension)[0],
                "image": base64.b64encode(img_encoded).decode("ascii")
            }
            redacted_data_bundles.append(data_bundle)

        with open('redacted_data.json', 'w') as json_file:
            json.dump(redacted_data_bundles, json_file, indent=4)

        return redacted_data_bundles

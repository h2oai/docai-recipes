from typing import Dict, Any
import re
import uuid
import base64
from typing import List, TypedDict
import pandas as pd
import cv2
from h2o_docai_scorer.post_processors.post_processor_supply_chain import PostProcessor as PostProcessorSupplyChain
from argus_contrib.utils import post_process as pp


labels_to_redact = ['address', 'name', 'phone', 'fax', 'email', 'website', 'other']


class PostProcessor(PostProcessorSupplyChain):
    """Represents a last step in pipeline process that receives all pipeline intermediate
    results and translates them into a final json structure that will be returned to user.
    """

    def has_text_tokens(self, via_predictions):
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

    def redact_region(self, image, xmin, ymin, xmax, ymax):
        """Redact a given region in the image using OpenCV."""
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
        return image

    def get_entities(self):
        if not self.has_labelling_model:
            return []

        if not self.has_text_tokens(self.label_via_predictions):
            return []

        merging_results = pp.post_process_via_predictions(
            input_dir='',
            via_predictions=self.label_via_predictions,
            probabilities=self.label_top_n,
            token_merge_type='mixed',  # "x", "xy", mixed
            labels_to_merge_x="",
            labels_to_merge_xy="address|name",
            output_labels='ALL_TOKENS',
            output_cleaning_method='NONE',
            token_merge_threshold_x=0.5,
            token_merge_threshold_xy=0.33,
            parse_line_items=False,
            try_templates=False,
        )

        redacted_images_per_page = {}
        for doc in merging_results:
            predictions = merging_results[doc]
            predictions = predictions.round(decimals=4)

            # Filter the predictions based on the provided labels
            filtered_predictions = predictions[predictions['label'].isin(labels_to_redact)]

            for _, row in filtered_predictions.iterrows():
                page_id = row["page_id"]
                filename = f"{doc}+{page_id}{self.img_extension}"

                if filename not in redacted_images_per_page:
                    redacted_images_per_page[filename] = self.images[filename].copy()

                redacted_images_per_page[filename] = self.redact_region(
                    redacted_images_per_page[filename],
                    int(row["xmin"]),
                    int(row["ymin"]),
                    int(row["xmax"]),
                    int(row["ymax"]),
                )

        redacted_data_bundles = []
        for filename, redacted_image in redacted_images_per_page.items():
            _, img_encoded = cv2.imencode(".png", redacted_image)
            img = bytes(img_encoded.flatten())

            data_bundle = {
                "pageIndex": filename.split('+')[1].split(self.img_extension)[0],
                "image": base64.b64encode(img).decode("ascii"),
            }
            redacted_data_bundles.append(data_bundle)

        return redacted_data_bundles

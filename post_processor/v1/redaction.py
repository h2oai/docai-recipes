from typing import Dict, Any
import re
import uuid
import base64
from typing import List, TypedDict
import pandas as pd
import cv2
from h2o_docai_scorer.post_processors.post_processor_supply_chain import PostProcessor as PostProcessorSupplyChain
from argus_contrib.utils import post_process as pp


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

        df_list = []
        for doc in merging_results:
            predictions = merging_results[doc]
            predictions = predictions.round(decimals=4)
            for idx, row in predictions.iterrows():
                # df_list.append(row.to_dict())
                df_list.append(self.get_entity(doc, row))
        return df_list

    def get_entity(self, doc, filtered_row):
        filename = f"{doc}+{filtered_row['page_id']}{self.img_extension}"
        sliced_img = cv2.imencode(
            ".png",
            self.images[filename][
                int(filtered_row["ymin"]) : int(filtered_row["ymax"]),
                int(filtered_row["xmin"]) : int(filtered_row["xmax"]),
            ],
        )[1]
        img = bytes(sliced_img.flatten())

        data_bundle = {
            "pageIndex": filtered_row["page_id"],
            "image": base64.b64encode(img).decode("ascii"),
        }

        return data_bundle

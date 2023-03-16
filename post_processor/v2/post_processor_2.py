from typing import List
import uuid
import pandas as pd

from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor, BaseEntity


class SupplyChainEntity(BaseEntity):
    text: str
    label: str
    labelConfidence: float
    ocrConfidence: float
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    entityId: str
    lineId: int


class PostProcessor(BasePostProcessor):
    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self) -> List[SupplyChainEntity]:
        if not self.has_labelling_model:
            return []

        docs = pp.post_process_predictions(
            model_preds=self.label_via_predictions,
            top_n_preds=self.label_top_n,
            # token merging options
            token_merge_type="MIXED_MERGE",
            token_merge_xdist_regular=1.0,
            label_merge_x_regular="name|address|description",
            token_merge_xydist_regular=1.0,
            label_merge_xy_regular="name|address",
            token_merge_xdist_wide=1.5,
            label_merge_x_wide="phone|fax|total|net|due|date|number|invoice|po|order",
            output_labels="EXCLUDE_O",
            # line-item parsing options
            parse_line_items=True,
            line_item_completeness=0.6,
            # template options
            try_templates=False,
            templates_dict_dir="",
            templates_input_dir=self.input_dir,
            use_camelot_tables=False,
            images_dir_camelot="",
            verbose=True,
        )

        for doc in docs:
            docs[doc]["id"] = docs[doc]["label"].apply(lambda row: str(uuid.uuid4()))

        if hasattr(self.extra_params, "labelingThreshold"):
            labeling_threshold = self.extra_params["labelingThreshold"]
        else:
            labeling_threshold = 0.5  # default labeling threshold

        df_list = []
        # only one array - assuming there will be only one document provided
        for doc in docs:
            predictions = docs[doc]
            predictions_filtered = []
            for label in self.label_top_n["class_names"]:
                pred_df = predictions[predictions.label == label]
                pred_df = pred_df[pred_df["probability"] > labeling_threshold]
                predictions_filtered.append(pred_df)
            predictions_filtered = pd.concat(predictions_filtered)

            for idx, row in predictions_filtered.iterrows():
                df_list.append(self.get_entity(row))
        return df_list

    def get_entity(self, filtered_row) -> SupplyChainEntity:
        filtered_label = self.remove_non_ascii(filtered_row["label"])
        filtered_text = self.remove_non_ascii(filtered_row["text"])
        data_bundle: SupplyChainEntity = {
            "pageIndex": filtered_row["page_id"],
            "text": filtered_text,
            "label": filtered_label,
            "labelConfidence": round(float(filtered_row["probability"]), 3),
            "ocrConfidence": round(float(filtered_row.get("ocr_confidence", 1.0)), 3), # in templates, ocr_confidence may not be available
            "lineId": filtered_row["line"],
            "xmin": (filtered_row["xmin"]),
            "ymin": (filtered_row["ymin"]),
            "xmax": (filtered_row["xmax"]),
            "ymax": (filtered_row["ymax"]),
            "entityId": filtered_row["id"],
        }

        return data_bundle

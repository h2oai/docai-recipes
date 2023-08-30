from typing import List

import uuid
import pandas as pd

from argus_contrib.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor, SupplyChainEntity


class PostProcessor(BasePostProcessor):

    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self) -> List[SupplyChainEntity]:
        if not self.has_labelling_model:
            return []

        merging_results = pp.post_process_via_predictions(input_dir=self.input_dir,
                                                          via_predictions=self.label_via_predictions,
                                                          probabilities=self.label_top_n,
                                                          token_merge_type='mixed',  # "x", "xy", mixed
                                                          labels_to_merge_x="",
                                                          labels_to_merge_x_long_range="",
                                                          labels_to_merge_xy="address|name",
                                                          parse_line_items=True,
                                                          output_labels='FULL',
                                                          output_cleaning_method='NONE',
                                                          token_merge_threshold_x=0.5,
                                                          token_merge_threshold_xy=0.33,
                                                          try_templates=False,
                                                          templates_dict_dir=""
                                                          )

        for doc in merging_results:
            merging_results[doc]['id'] = merging_results[doc]['label'].apply(lambda row: str(uuid.uuid4()))

        if hasattr(self.extra_params, 'labelingThreshold'):
            labeling_threshold = self.extra_params["labelingThreshold"]
        else:
            labeling_threshold = 0.5  # default labeling threshold

        df_list = []
        for doc in merging_results:
            predictions = merging_results[doc]
            predictions_filtered = []
            for label in self.label_top_n['class_names']:
                pred_df = predictions[predictions.label == label]
                pred_df = pred_df[pred_df['probability'] > labeling_threshold]

                if pred_df.empty:
                    pred_df = pd.DataFrame(columns=predictions.columns)
                    pred_df.loc[0, "label"] = label
                    pred_df.loc[0, "probability"] = 0.0
                    pred_df.loc[0, "ocr_confidence"] = 0.0
                    pred_df = pred_df.fillna('')

                predictions_filtered.append(pred_df)
            predictions_filtered = pd.concat(predictions_filtered)

            for idx, row in predictions_filtered.iterrows():
                df_list.append(self.get_entity(doc, row))
        return df_list

    def get_entity(self, doc, filtered_row) -> SupplyChainEntity:
        filtered_label = self.remove_non_ascii(filtered_row['label'])
        filtered_text = self.remove_non_ascii(filtered_row['text'])
        data_bundle: SupplyChainEntity = {
            'pageIndex': filtered_row['page_id'],
            filtered_label: filtered_text,
            'lineId': filtered_row['line'],
            "labelConfidence": round(float(filtered_row["probability"]), 3),
            "ocrConfidence": round(float(filtered_row.get("ocr_confidence", 1.0)), 3),
            'imageCoordinates': {
                'xmin': (filtered_row['xmin']),
                'ymin': (filtered_row['ymin']),
                'xmax': (filtered_row['xmax']),
                'ymax': (filtered_row['ymax'])
            },
        }
        return data_bundle

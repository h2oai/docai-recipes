from typing import List
import uuid
import pandas as pd
from argus_contrib.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor, GenericEntity


class PostProcessor(BasePostProcessor):

    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self) -> List[GenericEntity]:
        if not self.has_labelling_model:
            return []

        docs = pp.post_process_via_predictions(input_dir=self.input_dir,
                                               via_predictions=self.label_via_predictions,
                                               probabilities=self.label_top_n,
                                               token_merge_type='mixed',
                                               labels_to_merge_x="",
                                               labels_to_merge_x_long_range="",
                                               labels_to_merge_xy="address",
                                               parse_line_items=False,
                                               output_labels='FULL',
                                               output_cleaning_method='NONE',
                                               token_merge_threshold_x=0.5,
                                               token_merge_threshold_xy=0.33,
                                               try_templates=False,
                                               templates_dict_dir=""
                                               )

        for doc in docs:
            docs[doc]['id'] = docs[doc]['label'].apply(lambda row: str(uuid.uuid4()))

        if hasattr(self.extra_params, 'labelingThreshold'):
            labeling_threshold = self.extra_params["labelingThreshold"]
        else:
            labeling_threshold = 0.5  # default labeling threshold

        df_list = []
        for doc in docs:
            predictions = docs[doc]
            predictions_filtered = []
            for label in self.label_top_n['class_names']:
                pred_df = predictions[predictions.label == label]
                pred_df = pred_df[pred_df['probability'] > labeling_threshold]
                predictions_filtered.append(pred_df)
            predictions_filtered = pd.concat(predictions_filtered)

            for idx, row in predictions_filtered.iterrows():
                df_list.append(self.get_entity(doc, row))
        return df_list

    def get_entity(self, doc, filtered_row) -> GenericEntity:
        filtered_label = self.remove_non_ascii(filtered_row['label'])
        filtered_text = self.remove_non_ascii(filtered_row['text'])
        data_bundle: GenericEntity = {
            'pageIndex': filtered_row['page_id'],
            'text': filtered_text,
            'label': filtered_label,
            "labelConfidence": round(float(filtered_row["probability"]), 3),
            "ocrConfidence": round(float(filtered_row.get("ocr_confidence", 1.0)), 3),
            'xmin': (filtered_row['xmin']),
            'ymin': (filtered_row['ymin']),
            'xmax': (filtered_row['xmax']),
            'ymax': (filtered_row['ymax']),
            'entityId': filtered_row['id']
        }

        return data_bundle

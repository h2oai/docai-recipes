import uuid
import pandas as pd
from typing import List
from argus.data_model import Document
from argus.processors.post_processors.base_post_processor import BasePostProcessor, BaseEntity
from argus.processors.post_processors.utils.utility import doc_to_df
from argus.processors.post_processors.utils import post_process as pp


# These are what in the output
class GenericEntity(BaseEntity):
    text: str
    label: str
    labelConfidence: float
    ocrConfidence: float
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    entityId: str


class PostProcessor(BasePostProcessor):

    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self, doc: Document, doc_id: str) -> List[GenericEntity]:

        if not self.has_labelling_model:
            return []

        df_doc = doc_to_df(doc, doc_id, self.token_label_names)

        docs = pp.post_process_predictions(model_preds=df_doc,
                                           top_n_preds=self.class_names,
                                           token_merge_type='MIXED_MERGE',
                                           token_merge_xdist_regular=1.0,
                                           label_merge_x_regular='ALL',
                                           token_merge_xydist_regular=1.0,
                                           label_merge_xy_regular='address',
                                           token_merge_xdist_wide=1.5,
                                           label_merge_x_wide='phone|fax',
                                           output_labels='INCLUDE_O',
                                           parse_line_items=False,
                                           line_item_completeness=0.6,
                                           try_templates=False,
                                           templates_dict_dir='',
                                           templates_input_dir='./',
                                           use_camelot_tables=False,
                                           images_dir_camelot='',
                                           verbose=True)

        for doc in docs:
            docs[doc]['id'] = docs[doc]['label'].apply(lambda row: str(uuid.uuid4()))

        labeling_threshold = self.labeling_threshold

        df_list = []
        # only one array - assuming there will be only one document provided
        for doc in docs:
            predictions = docs[doc]
            predictions_filtered = []
            for label in self.class_names:
                pred_df = predictions[predictions.label == label]
                pred_df = pred_df[pred_df['probability'] > labeling_threshold]
                predictions_filtered.append(pred_df)
            predictions_filtered = pd.concat(predictions_filtered)

            for idx, row in predictions_filtered.iterrows():
                df_list.append(self.get_entity(row))
        return df_list

    def get_entity(self, filtered_row) -> GenericEntity:
        filtered_label = self.remove_non_ascii(filtered_row['label'])
        filtered_text = self.remove_non_ascii(filtered_row['text'])
        data_bundle: GenericEntity = {
            'pageIndex': filtered_row['page_id'],
            'text': filtered_text,
            'label': filtered_label,
            'labelConfidence': round(filtered_row['probability'], 3),
            'ocrConfidence': filtered_row['ocr_confidence'],
            'xmin': (filtered_row['xmin']),
            'ymin': (filtered_row['ymin']),
            'xmax': (filtered_row['xmax']),
            'ymax': (filtered_row['ymax']),
            'entityId': filtered_row['id']
        }

        return data_bundle
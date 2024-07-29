# This is the post processor for DocAI 0.7 and later versions.It is used to extract detected signatures to the final output


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
                
        # append the signautres
        signatures = self.get_signatures(doc)

        # prep the final output list
        df_list = []

        # only process if the model has labelling model
        if self.has_labelling_model:
            
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
                                            output_labels='EXCLUDE_O',
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
                

        if len(signatures) > 0:
            df_list += signatures
        
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
    
    def get_signatures(self, doc) -> List[GenericEntity]:
        print('Getting signatures...')
        
        cols = ['page_id', 'xmin', 'ymin', 'xmax', 'ymax', 'text', 'label', 'probability','ocr_confidence','id']
        sig_list = []
        signature_bbox = []
        for page_id, page in doc.pages.items():
            for box in page.boxes:
                if box.type == 'signature':
                    [[x1, y1], [x2, y2]] = box.shape.bounding_box()
                    # width = x2 - x1
                    # height = y2 - y1                    
                    signature_bbox.append([page_id, x1, y1, x2, y2, box.text, box.type, box.attributes['confidence'], 1, 100])
                    
        if len(signature_bbox) > 0:
            sig_df = pd.DataFrame(signature_bbox, columns=cols)

            for idx, row in sig_df.iterrows():
                sig_list.append(self.get_entity(row))
        
        return sig_list
        
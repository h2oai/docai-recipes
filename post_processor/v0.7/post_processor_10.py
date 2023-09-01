from typing import List
from argus.data_model import Document
from argus.processors.post_processors.base_post_processor import BasePostProcessor, BaseEntity
from argus.processors.post_processors.utils.utility import doc_to_df_ocr_only
from argus.processors.post_processors.utils import post_process as pp


class OcrTokens(BaseEntity):
    text: str
    ocrConfidence: float
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class PostProcessor(BasePostProcessor):
    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self, doc: Document, doc_id: str) -> List[OcrTokens]:
        df_doc = doc_to_df_ocr_only(doc, doc_id)

        df_list = []
        for _, row in df_doc.iterrows():
            df_list.append(self.get_entity(row))
        return df_list

    def get_entity(self, filtered_row) -> OcrTokens:
        filtered_text = self.remove_non_ascii(filtered_row['text'])
        data_bundle: OcrTokens = {
            'pageIndex': filtered_row['page_id'],
            'text': filtered_text,
            'ocrConfidence': filtered_row['ocr_confidence'],
            'xmin': round(float(filtered_row['xmin']), 3),
            'ymin': round(float(filtered_row['ymin']), 3),
            'xmax': round(float(filtered_row['xmax']), 3),
            'ymax': round(float(filtered_row['ymax']), 3),
        }

        return data_bundle

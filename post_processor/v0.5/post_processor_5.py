import base64
from typing import Any, Dict, List

import cv2
import uuid
import pandas as pd

from argus_contrib.ucsf import read_document_ucsf
from h2o_docai_scorer.post_processors import BasePostProcessor, RootResult, BaseEntity


class PostProcessor(BasePostProcessor):

    def __init__(
            self,
            input_dir: str,
            temp_working_dir: str,
            label_via_predictions: Dict[str, Any],
            label_top_n,
            class_predictions,
            class_top_n,
            images,
            img_extension,
            document_guid,
            extra_params,
            page_failures,
            has_labelling_model: bool,
            has_classification_model: bool):
        super().__init__(
            input_dir,
            temp_working_dir,
            label_via_predictions,
            label_top_n,
            class_predictions,
            class_top_n,
            images,
            img_extension,
            document_guid,
            extra_params,
            page_failures,
            has_labelling_model,
            has_classification_model)

        self.df_list = None
        self.merging_results = None
        self.class_names = None
        self.CLIENT_DPI = 200

    def client_resolution(self):
        return self.CLIENT_DPI

    def argus_resolution(self):
        return self.ARGUS_DPI

    def process(self) -> RootResult:
        if self.has_labelling_model:
            label_df = pd.DataFrame(self.label_top_n['class_names'])
            self.log.info('Merging prediction results')
            self.class_names = self.label_top_n['class_names']

            self.df_list, self.merging_results = self.process_labels(
                via_preds=self.label_via_predictions,
                label_df=label_df,
                label_top_n=self.label_top_n,
                images=self.images,
                extra_params=self.extra_params,
                img_ext=self.img_extension)
        else:
            self.merging_results = {}
            self.class_names = []

        root = super().process()

        probabilities = self.prepare_probabilities({
            'class_names': self.class_names,
            'merging_results': self.merging_results
        })
        root.update(probabilities)

        return root

    def get_entities(self) -> List[BaseEntity]:
        if self.df_list is not None:
            return self.df_list
        else:
            return {}

    @staticmethod
    def get_labeling_threshold(extra_params):
        if hasattr(extra_params, 'labelingThreshold'):
            labeling_threshold = extra_params["labelingThreshold"]
        else:
            labeling_threshold = 0.5  # default labeling threshold
        return labeling_threshold

    def prepare_probabilities(self, value):
        merging_results = value['merging_results']
        class_names = value['class_names']
        labeling_threshold = self.get_labeling_threshold(value)
        classes_probabilities_all_documents = {'entityConfidences': []}
        for doc in merging_results:
            predictions = merging_results[doc]
            class_probabilities_final = []
            for class_name in class_names:
                class_probabilities = []
                for entity_value, class_probability, id in zip(predictions['text'], predictions[class_name],
                                                               predictions['id']):
                    if class_probability >= labeling_threshold:
                        class_probabilities.append(
                            {'entityValue': entity_value, 'entityConfidence': class_probability, 'entityId': id})
                if len(class_probabilities) != 0:
                    class_probabilities_final.append(
                        {'entityClass': class_name, 'topEntities': class_probabilities})
            # assuming only one document is contained in merging_results
            classes_probabilities_all_documents['entityConfidences'] = class_probabilities_final
        return classes_probabilities_all_documents

    def process_labels(self, via_preds, label_df, label_top_n, images, extra_params, img_ext):
        merging_results = read_document_ucsf.post_process_via_predictions(via_preds, label_df, label_top_n,
                                                                          output_probability_mode=read_document_ucsf.
                                                                          OutputProbabilityMode.OUTPUT_BOTH)

        for doc in merging_results:
            merging_results[doc]['id'] = merging_results[doc]['label'].apply(lambda row: str(uuid.uuid4()))

        labeling_threshold = self.get_labeling_threshold(extra_params)

        df_list = []
        # only one array - assuming there will be only one document provided
        for doc in merging_results:
            predictions = merging_results[doc]
            predictions_filtered = pd.DataFrame()

            for label in label_top_n['class_names']:
                pred_df = predictions[predictions.label == label]
                pred_df = pred_df[pred_df['probability'] > labeling_threshold]
                predictions_filtered = predictions_filtered.append(pred_df)

            for idx, row in predictions_filtered.iterrows():
                filename = doc + '+' + row['page_id'] + img_ext
                sliced_img = cv2.imencode('.png', images[filename][int(row['bottom']):int(row['top']),
                                                  int(row['left']):int(row['right'])])[1]
                # imencode wraps image in one more array level
                if len(sliced_img.shape) > 1:
                    img = bytes(i[0] for i in sliced_img)
                else:
                    img = bytes(sliced_img)
                # f = open(os.path.join(os.path.dirname(__file__), 'temp.png'), 'wb')
                # f.write(img)
                # f.close()
                filtered_label = self.remove_non_ascii(row['label'])
                filtered_text = self.remove_non_ascii(row['text'])
                contact_group = None
                if "PatientContact" in filtered_label:
                    if "2" in filtered_label:
                        contact_group = "Contact 2"
                    else:
                        contact_group = "Contact 1"
                    filtered_label = filtered_label.replace("PatientContact2", "PatientContact")
                data_bundle = {'pageIndex': row['page_id'],
                               'id': row['id'],
                               'label': filtered_label,
                               'value': filtered_text,
                               'labelConfidence': row['probability'],
                               'imageCoordinates': {'bx': (row['left'] * self.CLIENT_DPI / self.ARGUS_DPI),
                                                    'by': (row['top'] * self.CLIENT_DPI / self.ARGUS_DPI),
                                                    'ex': (row['right'] * self.CLIENT_DPI / self.ARGUS_DPI),
                                                    'ey': (row['bottom'] * self.CLIENT_DPI / self.ARGUS_DPI)},
                               'image': base64.b64encode(img).decode('ascii')
                               }
                if contact_group is not None:
                    data_bundle["group"] = contact_group
                df_list.append(data_bundle)
        return df_list, merging_results

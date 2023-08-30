import re
import uuid
import base64
from typing import List, TypedDict
import pandas as pd
import cv2

from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor, RootResult, BaseEntity


class ImageCoordinates(TypedDict):
    bx: int
    by: int
    ex: int
    ey: int


class CustomEntity(BaseEntity):
    id: str
    value: str
    label: str
    labelConfidence: float
    imageCoordinates: ImageCoordinates
    group: str
    image: str


class PostProcessor(BasePostProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_list = None
        self.merging_results = None
        self.class_names = None
        self.CLIENT_DPI = 200

    def client_resolution(self):
        return self.CLIENT_DPI

    def argus_resolution(self):
        return self.ARGUS_DPI

    @staticmethod
    def get_labeling_threshold(extra_params):
        if hasattr(extra_params, "labelingThreshold"):
            labeling_threshold = extra_params["labelingThreshold"]
        else:
            labeling_threshold = 0.5  # default labeling threshold
        return labeling_threshold

    def prepare_probabilities(self, value):
        merging_results = value["merging_results"]
        class_names = value["class_names"]
        labeling_threshold = self.get_labeling_threshold(value)
        classes_probabilities_all_documents = {"entityConfidences": []}
        for doc in merging_results:
            predictions = merging_results[doc]
            class_probabilities_final = []
            for class_name in class_names:
                class_probabilities = []
                for entity_value, class_probability, id in zip(
                        predictions["text"], predictions[class_name], predictions["id"]
                ):
                    if class_probability >= labeling_threshold:
                        class_probabilities.append(
                            {
                                "entityValue": entity_value,
                                "entityConfidence": class_probability,
                                "entityId": id,
                            }
                        )
                if len(class_probabilities) != 0:
                    class_probabilities_final.append(
                        {"entityClass": class_name, "topEntities": class_probabilities}
                    )
            # assuming only one document is contained in merging_results
            classes_probabilities_all_documents[
                "entityConfidences"
            ] = class_probabilities_final
        return classes_probabilities_all_documents

    def process(self) -> RootResult:
        if self.has_labelling_model:
            self.log.info("Merging prediction results")
            self.class_names = self.label_top_n["class_names"]
            self.df_list, self.merging_results = self.process_labels()
        else:
            self.merging_results = {}
            self.class_names = []

        root = super().process()

        probabilities = self.prepare_probabilities(
            {"class_names": self.class_names, "merging_results": self.merging_results}
        )
        root.update(probabilities)

        return root

    def clean_preds(self, df):
        dfs = []
        address_pattern = re.compile(
            "Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St|PO|Box",
            flags=re.IGNORECASE,
        )
        digit_pattern = re.compile(r"(\d+[.\d]*)")

        for label in self.label_top_n["class_names"]:
            print("label: ", label)
            pred_df = df[df.label == label]

            if "address" in label:
                pred_df["text"] = [
                    re.sub(r"\W+", " ", text) for text in pred_df["text"]
                ]
                pred_df = pred_df[pred_df["text"].str.contains(address_pattern)]
            elif "name" in label:
                pred_df["text"] = pred_df["text"].str.replace(digit_pattern, " ")
            elif "Phone" in label or "Fax" in label:
                pred_df = pred_df[pred_df["text"].str.count("\d") > 5]

            dfs.append(pred_df)

        dfs = pd.concat(dfs).reset_index(drop=True)

        return dfs

    def process_labels(self):
        if not self.has_labelling_model:
            return []

        docs = pp.post_process_predictions(
            model_preds=self.label_via_predictions,
            top_n_preds=self.label_top_n,
            token_merge_type="MIXED_MERGE",
            token_merge_xdist_regular=1.0,
            label_merge_x_regular="ALL",
            token_merge_xydist_regular=1.0,
            label_merge_xy_regular="name|address|comment",
            token_merge_xdist_wide=1.5,
            label_merge_x_wide="phone|fax",
            output_labels="EXCLUDE_O",
            verbose=True,
        )

        for doc in docs:
            docs[doc]["id"] = docs[doc]["label"].apply(lambda row: str(uuid.uuid4()))

        labeling_threshold = self.get_labeling_threshold(self.extra_params)

        df_list = []
        # only one array - assuming there will be only one document provided
        for doc in docs:
            predictions = docs[doc]
            predictions_filtered = []
            for label in self.label_top_n[
                "class_names"
            ]:  # class name "O" is present in the list, but not in predictions
                pred_df = predictions[predictions.label == label]
                pred_df = pred_df[pred_df["probability"] > labeling_threshold]
                predictions_filtered.append(pred_df)

            predictions_filtered = pd.concat(predictions_filtered).reset_index(
                drop=True
            )
            predictions_filtered = self.clean_preds(predictions_filtered)

            for idx, row in predictions_filtered.iterrows():
                df_list.append(self.get_entity(doc, row))
        return df_list, docs

    def get_entity(self, doc, row) -> CustomEntity:
        filename = f"{doc}+{row['page_id']}{self.img_extension}"
        sliced_img = cv2.imencode(
            ".png",
            self.images[filename][
            int(row["ymin"]): int(row["ymax"]), int(row["xmin"]): int(row["xmax"])
            ],
        )[1]
        img = bytes(sliced_img.flatten())

        filtered_label = self.remove_non_ascii(row["label"])
        filtered_text = self.remove_non_ascii(row["text"])
        contact_group = None

        if "PatientContact" in filtered_label:
            contact_group = "Contact 2" if "2" in filtered_label else "Contact 1"
            filtered_label = filtered_label.replace("PatientContact2", "PatientContact")

        data_bundle: CustomEntity = {
            "pageIndex": row["page_id"],
            "id": row["id"],
            "label": filtered_label,
            "value": filtered_text,
            "labelConfidence": round(float(row["probability"]), 3),
            "imageCoordinates": {
                "bx": int(row["xmin"] * self.CLIENT_DPI / self.ARGUS_DPI),
                "by": int(row["ymin"] * self.CLIENT_DPI / self.ARGUS_DPI),
                "ex": int(row["xmax"] * self.CLIENT_DPI / self.ARGUS_DPI),
                "ey": int(row["ymax"] * self.CLIENT_DPI / self.ARGUS_DPI),
            },
            "image": base64.b64encode(img).decode("ascii"),
        }

        if contact_group is not None:
            data_bundle["group"] = contact_group

        return data_bundle

    def get_entities(self) -> List[CustomEntity]:
        if self.df_list is not None:
            return self.df_list
        else:
            return {}

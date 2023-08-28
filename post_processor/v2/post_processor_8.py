# Works with DocAI 0.6.2


from typing import TypedDict, List
import uuid
import pandas as pd

from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors import BasePostProcessor, BaseEntity


class ImageCoordinates(TypedDict):
    xmin: int
    xmax: int
    ymin: int
    ymax: int


class SupplyChainEntity(BaseEntity):
    lineId: int
    labelConfidence: float
    ocrConfidence: float
    imageCoordinates: ImageCoordinates


class PostProcessor(BasePostProcessor):
    def client_resolution(self):
        return None

    def argus_resolution(self):
        return self.ARGUS_DPI

    def get_entities(self) -> List[SupplyChainEntity]:
        if not self.has_labelling_model:
            return []

        docs = post_process_predictions(
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
            try_templates=True,
            template_dicts=template_dict_list,  # template_dict_list is a global variable declared in this script 
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
            filtered_label: filtered_text,
            "lineId": filtered_row["line"],
            "labelConfidence": round(float(filtered_row["probability"]), 3),
            "ocrConfidence": round(
                float(filtered_row.get("ocr_confidence", 1.0)), 3
            ),  # in templates, ocr_confidence may not be available
            "imageCoordinates": {
                "xmin": (filtered_row["xmin"]),
                "ymin": (filtered_row["ymin"]),
                "xmax": (filtered_row["xmax"]),
                "ymax": (filtered_row["ymax"]),
            },
        }

        return data_bundle


#####################################################################################
# Template code
#####################################################################################
import os, time, re
from glob import glob
import numpy as np
import traceback
from typing import Dict, Union, TypedDict, List, Any, Hashable

from argus import ocr as argus_ocr
from argus.serializers import BaseSerializer
from argus.filters import Sorter
from argus.data_model import BoundingBox, Polygon

from argus.processors.post_processors.utils.box_process import box_process
from argus.processors.post_processors.utils.text_merge import merge_tokens

from argus_contrib.templates.processor import dict2csv, get_header_item, get_line_items
from argus_contrib.templates.utils import (
    ngram,
    get_text_in_search_box,
    merge_boxes,
    get_anchor_boxes,
)


def bb_intersection_over_union(box1, box2):
    # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_rel_bbox(bbox, page_width, page_height):
    bbox = bbox.reshape(4)
    rel_bbox = [
        b / x for b, x in zip(bbox, [page_width, page_height, page_width, page_height])
    ]
    return rel_bbox


class Box(BoundingBox):
    @staticmethod
    def from_rel_bbox(rel_bbox, w, h, text=""):
        abs_bbox = [b * x for b, x in zip(rel_bbox, [w, h, w, h])]
        bbox = BoundingBox(text=text, shape=Polygon.from_edges(abs_bbox))
        box = Box(bbox, w, h)
        return box

    @staticmethod
    def from_abs_bbox(abs_bbox, w, h):
        bbox = BoundingBox(text="", shape=Polygon.from_edges(abs_bbox))
        box = Box(bbox, w, h)
        return box

    def __init__(
        self,
        instance,
        page_width,
        page_height,
        text_index=0,
        text_index_rev=-1,
        page_rev=0,
    ):
        # del instance.a
        del instance.entities
        instance_attrs = vars(instance)
        super().__init__(**instance_attrs)
        self.rel_bbox = get_rel_bbox(
            instance.shape.bounding_box(), page_width, page_height
        )
        self.abs_bbox = instance.shape.bounding_box().reshape(4)
        self.text_index = text_index
        self.text_index_rev = text_index_rev
        self.tokens = [self.text]
        self.token_rel_bboxes = [self.rel_bbox]
        self.anchor_candidate = True
        self.page_rev = page_rev

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __add__(self, other):
        self.tokens.append(other.text)
        self.token_rel_bboxes.append(other.rel_bbox)

        new = self
        new.text = " ".join([new.text, other.text])
        bb1 = new.shape.bounding_box()
        bb2 = other.shape.bounding_box()
        left = min(bb1[0][0], bb2[0][0])
        top = min(bb1[0][1], bb2[0][1])
        right = max(bb1[1][0], bb2[1][0])
        bottom = max(bb1[1][1], bb2[1][1])
        new.abs_bbox = [left, top, right, bottom]
        new.shape = new.shape.from_edges([left, top, right, bottom])
        rel_left = min(new.rel_bbox[0], other.rel_bbox[0])
        rel_top = min(new.rel_bbox[1], other.rel_bbox[1])
        rel_right = max(new.rel_bbox[2], other.rel_bbox[2])
        rel_bottom = max(new.rel_bbox[3], other.rel_bbox[3])
        new.rel_bbox = [rel_left, rel_top, rel_right, rel_bottom]
        return new

    @property
    def height(self):
        return self.abs_bbox[3] - self.abs_bbox[1]

    @property
    def width(self):
        return self.abs_bbox[2] - self.abs_bbox[0]

    @property
    def rel_height(self):
        return self.rel_bbox[3] - self.rel_bbox[1]

    @property
    def rel_width(self):
        return self.rel_bbox[2] - self.rel_bbox[0]

    @property
    def area(self):
        return self.height * self.width

    @property
    def rel_area(self):
        return self.rel_height * self.rel_width

    def set_box_from_rel_coords(self, rel_bbox, w, h):
        self.rel_bbox = rel_bbox
        self.abs_bbox = [b * x for b, x in zip(rel_bbox, [w, h, w, h])]
        self.shape = Polygon.from_edges(self.abs_bbox)

    def bb_intersection_over_union(self, other):
        return bb_intersection_over_union(self.rel_bbox, other.rel_bbox)

    def bb_percentage_within(self, other):
        # Calculates the percentage of box1 (self) that is within box2 (other)
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.rel_bbox[0], other.rel_bbox[0])
        yA = max(self.rel_bbox[1], other.rel_bbox[1])
        xB = min(self.rel_bbox[2], other.rel_bbox[2])
        yB = min(self.rel_bbox[3], other.rel_bbox[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (self.rel_bbox[2] - self.rel_bbox[0]) * (
            self.rel_bbox[3] - self.rel_bbox[1]
        )

        perc = interArea / float(boxAArea)
        # return the intersection over union value
        return perc


def template_router(pages, template_dicts):
    # for template_json in template_jsons:
    for template_dict in template_dicts:
        try:
            # #############################################################3
            # with open(template_json, "r") as f:
            #     template_dict = json.load(f)
            # #############################################################3
            for page in pages:
                w, h = page.size
                # For each header item check if the anchor exists
                router_anchors = []
                for header_label, header_anchors in template_dict[
                    "header_items"
                ].items():
                    try:
                        anchor_1 = header_anchors["anchor_1"]
                        if "router_coordinates" in anchor_1:
                            router_anchors.append(False)
                            search_box = Box.from_rel_bbox(
                                anchor_1["router_coordinates"], w, h
                            )
                            output = get_text_in_search_box(
                                search_box, page.boxes, threshold=0.3
                            )
                            if output:
                                match_box = merge_boxes(output)
                                if re.findall(
                                    re.compile(re.escape(anchor_1["regex"])),
                                    match_box.text,
                                ):
                                    router_anchors[-1] = True
                                else:
                                    print(anchor_1["regex"], match_box.text)
                            else:
                                print(
                                    f"No router match found for {header_label}. Looking for '{anchor_1['regex']}'."
                                )
                    except Exception as e:
                        print(
                            f"Error during template router while try to match with {template_dict} label"
                            f" {header_label}"
                        )
                        print(e)

                if router_anchors and (np.mean(router_anchors) > 0.5):
                    print(f"Match found: {template_dict}")
                    return template_dict
        except Exception as e:
            print(f"Error during template router with {template_dict}")
            print(e)


def process_templates(fname, template_dicts, template_dict=None):
    path, bn = os.path.split(fname)
    doc_id = os.path.splitext(bn)[0]
    img_path = "./images"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    page_xmls, pages, pdf_images, fnames = argus_ocr.PdfTextExtract(
        ser=None, input_dir=path, images_dir=img_path
    )._get_data_to_process_for_fname(bn, doc_id)
    doc_dict = {}
    pages_list = []
    for pnum, (page_xml, page, image_fname) in enumerate(
        zip(page_xmls, pages, [os.path.join(img_path, f) for f in fnames])
    ):
        try:
            pdf_text_extract = argus_ocr.PdfTextExtract(BaseSerializer, input_dir=path)
            argus_page = pdf_text_extract.process_one_page(
                pnum, image_fname, page, pdf_images, fname, page_xml
            )
            d, words = pdf_text_extract._get_page_tree(page_xml)
            w, h = argus_page.size

            # Add relative page bbox to argus page boxes
            argus_page.boxes = [Box(box, w, h) for box in argus_page.boxes]
            argus_page.boxes = [
                argus_page.boxes[i] for i in Sorter.box_sort(argus_page)
            ]

            pages_list.append(argus_page)
        except argus_ocr.PdfTextExtractException as e:
            pass

    if not template_dict:
        # Check for matching template
        template_dict = template_router(pages_list, template_dicts)

    if template_dict:
        # get header items
        header_item_dict = {}
        use_model_preds = []
        for header_label, header_config in template_dict["header_items"].items():
            header_item_dict[header_label] = get_header_item(
                template_dict, pages_list, header_label
            )
            if header_config.get("use_model_preds", False):
                use_model_preds.append(header_label)

        # get line items from table
        line_items = get_line_items(template_dict, pages_list)
        doc_dict = {"header_items": header_item_dict, "table": line_items}

        return doc_dict, use_model_preds
    else:
        print("No template found.")
        return doc_dict, []


#####################################################################################
# Template Dicts
#####################################################################################

arthrex = {
    "header_items": {
        "customer_purchase_order": {
            "anchor_1": {
                "regex": "No.",
                "ignore_case": False,
                "selection_index": 2,
                "router_coordinates": [0.61, 0.25, 0.64, 0.27],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "top", 0.0018],
                    ["anchor_1", "right", 0.0929],
                    ["anchor_1", "bottom", -0.0005],
                ]
            ],
        },
        "customer_po_date": {
            "anchor_1": {
                "regex": "Acknowledgement",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.58, 0.08, 0.91, 0.11],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.064],
                    ["anchor_1", "bottom", 0.0544],
                    ["anchor_1", "right", -0.1917],
                    ["anchor_1", "bottom", 0.0638],
                ]
            ],
        },
        "order_number": {
            "anchor_1": {
                "regex": "No.",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.58, 0.19, 0.61, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "top", 0.0018],
                    ["anchor_1", "right", 0.1065],
                    ["anchor_1", "bottom", -0.0005],
                ]
            ],
        },
        "invoice_date": {
            "anchor_1": {
                "regex": "Order Acknowledgement",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0],
                    [None, 0],
                    ["anchor_1", "right", -0.08],
                    [None, 0.1],
                ]
            ],
        },
        "billing_name": {
            "anchor_1": {
                "regex": "Sold-To",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0],
                    ["anchor_1", "bottom", 0.0],
                    ["anchor_1", "right", 0.14],
                    ["anchor_1", "bottom", 0.03],
                ]
            ],
        },
        "billing_address": {
            "anchor_1": {
                "regex": "Sold-To",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0],
                    ["anchor_1", "bottom", 0.03],
                    ["anchor_1", "right", 0.14],
                    ["anchor_1", "bottom", 0.1],
                ]
            ],
        },
        "supplier_name": {
            "anchor_1": {
                "regex": "No.",
                "ignore_case": False,
                "selection_index": 3,
                "router_coordinates": [0.02, 0.48, 0.06, 0.51],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0045],
                    ["anchor_1", "bottom", 0.3768],
                    ["anchor_1", "right", 0.015],
                    ["anchor_1", "bottom", 0.3862],
                ]
            ],
        },
        "supplier_address": {
            "anchor_1": {
                "regex": "Customer Service:",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.05],
                    ["anchor_1", "top", -0.02],
                    ["anchor_1", "right", 0.2],
                    ["anchor_1", "top", 0.0],
                ]
            ],
        },
        "receiver_name": {
            "use_model_preds": True,
            "anchor_1": {
                "regex": "Ship-To",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0],
                    ["anchor_1", "bottom", 0.0],
                    ["anchor_1", "right", 0.14],
                    ["anchor_1", "bottom", 0.02],
                ]
            ],
        },
        "receiver_address": {
            "use_model_preds": True,
            "anchor_1": {
                "regex": "Ship-To",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0],
                    ["anchor_1", "bottom", 0.02],
                    ["anchor_1", "right", 0.14],
                    ["anchor_1", "bottom", 0.1],
                ]
            ],
        },
        "total_amount": {
            "anchor_1": {
                "regex": "Total :",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0],
                    ["anchor_1", "top", 0.0],
                    [None, 1],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "Comments:",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                },
                {
                    "regex": "Total :",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": True,
                },
            ],
            "headers_exist_on_additional_pages": True,
            "line_anchor": {
                "column_name": "Ext. Price",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.75,
                "right_coordinate": 1,
            },
            "remove_line_regex": "",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.093],
                            ["line_anchor", "top", 0],
                            [None, 0.20],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": True,
                    "search_areas": [
                        [
                            [None, 0.2233],
                            ["line_anchor", "top", 0],
                            [None, 0.4112],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.42],
                            ["line_anchor", "top", 0],
                            [None, 0.48],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.491],
                            ["line_anchor", "top", 0],
                            [None, 0.5187],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.544],
                            ["line_anchor", "top", 0],
                            [None, 0.67],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.7874],
                            ["line_anchor", "top", 0],
                            [None, 0.8383],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
            },
        },
        {
            "table_end": [
                {
                    "regex": "Total :",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": True,
                },
                {
                    "regex": "Arthrex:",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": True,
                },
            ],
            "headers_exist_on_additional_pages": True,
            "line_anchor": {
                "column_name": "Ext. Price",
                "selection_index": 1,
                "vertical_alignment": "top",
                "left_coordinate": 0.75,
                "right_coordinate": 1,
            },
            "remove_line_regex": "",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.08],
                            ["line_anchor", "top", 0],
                            [None, 0.18],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": True,
                    "search_areas": [
                        [
                            [None, 0.2233],
                            ["line_anchor", "top", 0],
                            [None, 0.38],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_quantity_bo": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.38],
                            ["line_anchor", "top", 0],
                            [None, 0.42],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.42],
                            ["line_anchor", "top", 0],
                            [None, 0.49],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.52],
                            ["line_anchor", "top", 0],
                            [None, 0.564],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.7874],
                            ["line_anchor", "top", 0],
                            [None, 0.8383],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
            },
        },
    ],
}

lake_court = {
    "header_items": {
        "header_quantity_bo": {
            "anchor_1": {
                "regex": "Terms",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.62, 0.32, 0.69, 0.36],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0353],
                    ["anchor_1", "bottom", 0.0463],
                    ["anchor_1", "right", 0.0145],
                    ["anchor_1", "bottom", 0.0576],
                ]
            ],
        },
        "customer_purchase_order": {
            "anchor_1": {
                "regex": "Confirm",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.02, 0.29, 0.08, 0.31],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0],
                    ["anchor_1", "bottom", 0.0341],
                    ["anchor_1", "right", 0.0353],
                    ["anchor_1", "bottom", 0.0545],
                ]
            ],
        },
        "customer_po_date": {
            "anchor_1": {
                "regex": "Number:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.64, 0.1, 0.71, 0.12],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "bottom", 0.0036],
                    ["anchor_1", "right", 0.1002],
                    ["anchor_1", "bottom", 0.0151],
                ]
            ],
        },
        "order_number": {
            "anchor_1": {
                "regex": "Acknowledgement",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.67, 0.06, 0.82, 0.08],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0653],
                    ["anchor_1", "bottom", 0.0236],
                    ["anchor_1", "right", -0.0275],
                    ["anchor_1", "bottom", 0.0351],
                ]
            ],
        },
        "billing_name": {
            "anchor_1": {
                "regex": "Sold To:",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0],
                    ["anchor_1", "bottom", -0.01],
                    ["anchor_1", "right", 0.1],
                    ["anchor_1", "bottom", 0.025],
                ]
            ],
        },
        "billing_address": {
            "anchor_1": {
                "regex": "Sold To:",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0],
                    ["anchor_1", "bottom", 0.02],
                    ["anchor_1", "right", 0.1],
                    ["anchor_1", "bottom", 0.075],
                ]
            ],
        },
        "supplier_name": {
            "anchor_1": {
                "regex": "Order Date:",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    [None, 0.0],
                    ["anchor_1", "bottom", 0.0],
                    [None, 0.2],
                    ["anchor_1", "bottom", 0.025],
                ]
            ],
        },
        "supplier_address": {
            "anchor_1": {
                "regex": "Order Date:",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    [None, 0.0],
                    ["anchor_1", "bottom", 0.025],
                    [None, 0.2],
                    ["anchor_1", "bottom", 0.1],
                ]
            ],
        },
        "receiver_name": {
            "anchor_1": {
                "regex": "Ship To:",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0],
                    ["anchor_1", "bottom", 0.0],
                    ["anchor_1", "right", 0.2],
                    ["anchor_1", "bottom", 0.025],
                ]
            ],
        },
        "receiver_address": {
            "anchor_1": {
                "regex": "Customer",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.6, 0.19, 0.67, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0686],
                    ["anchor_1", "bottom", 0.038],
                    ["anchor_1", "right", 0.062],
                    ["anchor_1", "bottom", 0.0727],
                ]
            ],
        },
        "total_amount": {
            "anchor_1": {"regex": "Tax:", "ignore_case": False, "selection_index": 0},
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "bottom", 0.0035],
                    ["anchor_1", "right", 0.1158],
                    ["anchor_1", "bottom", 0.015],
                ]
            ],
        },
        "net_amount": {
            "anchor_1": {"regex": "Order:", "ignore_case": False, "selection_index": 0},
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "top", 0.0],
                    ["anchor_1", "right", 0.1158],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "Net Order:",
                    "ignore_case": False,
                    "selection_index": 1,
                    "document_table_end": False,
                },
                {
                    "regex": "Continued",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                },
            ],
            "headers_exist_on_additional_pages": True,
            "line_anchor": {
                "column_name": "Amount",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.85,
                "right_coordinate": 0.95,
            },
            "remove_line_regex": "",
            "table_start_regex": "",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.0294],
                            ["line_anchor", "top", 0],
                            [None, 0.2],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": True,
                    "search_areas": [
                        [
                            [None, 0.0392],
                            ["line_anchor", "bottom", 0],
                            [None, 0.3531],
                            ["line_anchor", "bottom", 0.015],
                        ]
                    ],
                },
                "line_unit": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.3431],
                            ["line_anchor", "top", 0],
                            [None, 0.3797],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.4714],
                            ["line_anchor", "top", 0],
                            [None, 0.4859],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_quantity_bo": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.66],
                            ["line_anchor", "top", 0],
                            [None, 0.6673],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.7698],
                            ["line_anchor", "top", 0],
                            [None, 0.8101],
                            ["line_anchor", "top", 0.015],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.8389500000000001],
                            ["line_anchor", "top", 0],
                            [None, 0.9081],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
            },
        }
    ],
}

medline = {
    "header_items": {
        "billing_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Bill To:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.52, 0.23, 0.59, 0.25],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0036],
                    ["anchor_1", "bottom", 0.013],
                    ["anchor_1", "right", 0.3496],
                    ["anchor_1", "bottom", 0.0761],
                ]
            ],
        },
        "billing_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Bill To:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.52, 0.23, 0.59, 0.25],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0036],
                    ["anchor_1", "bottom", -0.0028],
                    ["anchor_1", "right", 0.3476],
                    ["anchor_1", "bottom", 0.0145],
                ]
            ],
        },
        "customer_po_date": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Order Confirmation",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.52, 0.06, 0.73, 0.08],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.1804],
                    ["anchor_1", "bottom", 0.0347],
                    ["anchor_1", "right", 0.1581],
                    ["anchor_1", "bottom", 0.0504],
                ]
            ],
        },
        "customer_purchase_order": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Standard Order",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.7, 0.16, 0.86, 0.18],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0048],
                    ["anchor_1", "top", -0.0321],
                    ["anchor_1", "right", 0.0345],
                    ["anchor_1", "top", -0.0163],
                ]
            ],
        },
        "order_number": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Order Confirmation",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.52, 0.06, 0.73, 0.08],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.1804],
                    ["anchor_1", "bottom", 0.0177],
                    ["anchor_1", "right", 0.1597],
                    ["anchor_1", "bottom", 0.035],
                ]
            ],
        },
        "receiver_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Ship To:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.05, 0.23, 0.14, 0.25],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0051],
                    ["anchor_1", "bottom", 0.0288],
                    ["anchor_1", "right", 0.3456],
                    ["anchor_1", "bottom", 0.0819],
                ]
            ],
        },
        "receiver_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Ship To:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.05, 0.23, 0.14, 0.25],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0032],
                    ["anchor_1", "bottom", -0.0012],
                    ["anchor_1", "right", 0.3515],
                    ["anchor_1", "bottom", 0.0261],
                ]
            ],
        },
        "supplier_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Service: 800-MEDLINE",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.15, 0.12, 0.37, 0.14],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0043],
                    ["anchor_1", "top", -0.0615],
                    ["anchor_1", "right", 0.1088],
                    ["anchor_1", "top", -0.013],
                ]
            ],
        },
        "supplier_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Order Confirmation",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.52, 0.06, 0.73, 0.08],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.3657],
                    ["anchor_1", "top", -0.0175],
                    ["anchor_1", "left", -0.0462],
                    ["anchor_1", "top", 0.001],
                ]
            ],
        },
        "total_amount": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Items Total",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.3292],
                    ["anchor_1", "top", -0.0051],
                    ["anchor_1", "right", 0.4742],
                    ["anchor_1", "bottom", 0.0044],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "Items Total",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                },
                {
                    "regex": "Thank you for your order. Pricing is subject to review.",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                },
            ],
            "headers_exist_on_additional_pages": False,
            "line_anchor": {
                "column_name": "Line",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.74,
                "right_coordinate": 0.853,
            },
            "remove_line_regex": "",
            "table_start_regex": "",
            "fields": {
                "line_description": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.104],
                            ["line_anchor", "top", 0.011878076701526241],
                            [None, 0.606],
                            ["line_anchor", "top", 0.029786074675892227],
                        ]
                    ],
                },
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.104],
                            ["line_anchor", "top", -0.004801303964884485],
                            [None, 0.2825],
                            ["line_anchor", "top", 0.010975395064241722],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.394],
                            ["line_anchor", "top", -0.0038986223275999654],
                            [None, 0.4425],
                            ["line_anchor", "top", 0.012492385355503843],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.58],
                            ["line_anchor", "top", 0],
                            [None, 0.673],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.74],
                            ["line_anchor", "top", 0],
                            [None, 0.853],
                            ["line_anchor", "top", 0.060687124187958186],
                        ]
                    ],
                },
            },
        }
    ],
}

organogenesis_0 = {
    "header_items": {
        "customer_po_date": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Confirmation",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.5, 0.03, 0.69, 0.07],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0074],
                    ["anchor_1", "bottom", 0.0128],
                    ["anchor_1", "right", 0.1705],
                    ["anchor_1", "bottom", 0.0259],
                ]
            ],
        },
        "customer_purchase_order": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Confirmation",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.5, 0.03, 0.69, 0.07],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0035],
                    ["anchor_1", "bottom", 0.0001],
                    ["anchor_1", "right", 0.1721],
                    ["anchor_1", "bottom", 0.0116],
                ]
            ],
        },
        "net_amount": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales total",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.56, 0.88, 0.64, 0.9],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0238],
                    ["anchor_1", "top", -0.0017],
                    ["anchor_1", "right", 0.1274],
                    ["anchor_1", "bottom", -0.0001],
                ]
            ],
        },
        "order_number": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales order",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.08, 0.28, 0.17, 0.3],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0323],
                    ["anchor_1", "bottom", 0.0021],
                    ["anchor_1", "right", 0.0332],
                    ["anchor_1", "bottom", 0.0251],
                ]
            ],
        },
        "receiver_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.18, 0.17, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0004],
                    ["anchor_1", "bottom", 0.0002],
                    ["anchor_1", "right", 0.2765],
                    ["anchor_1", "bottom", 0.0417],
                ]
            ],
        },
        "receiver_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.18, 0.17, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0004],
                    ["anchor_1", "top", 0.0014],
                    ["anchor_1", "right", 0.2706],
                    ["anchor_1", "bottom", -0.0028],
                ]
            ],
        },
        "supplier_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.18, 0.17, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0557],
                    ["anchor_1", "top", -0.0644],
                    ["anchor_1", "right", 0.1388],
                    ["anchor_1", "top", -0.0371],
                ]
            ],
        },
        "supplier_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.18, 0.17, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0553],
                    ["anchor_1", "top", -0.0774],
                    ["anchor_1", "right", 0.1322],
                    ["anchor_1", "top", -0.0644],
                ]
            ],
        },
        "total_amount": {
            "use_model_preds": False,
            "page_num": -1,
            "anchor_1": {
                "regex": "Sales tax",
                "ignore_case": False,
                "selection_index": -1,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0],
                    ["anchor_1", "bottom", 0.0],
                    ["anchor_1", "right", 0.2],
                    ["anchor_1", "bottom", 0.02],
                ]
            ],
        },
        "total_tax": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales tax",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0],
                    ["anchor_1", "top", 0.0],
                    ["anchor_1", "right", 0.1305],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "The sales amount takes into account list prices and your contracted rates for products and any",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                }
            ],
            "headers_exist_on_additional_pages": False,
            "line_anchor": {
                "column_name": "Item",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.7722,
                "right_coordinate": 0.9,
            },
            "remove_line_regex": "",
            "table_start_regex": "",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.011],
                            ["line_anchor", "top", -0.002396062246240449],
                            [None, 0.1047],
                            ["line_anchor", "top", 0.013668876260250762],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.0129],
                            ["line_anchor", "top", 0.012153724745099292],
                            [None, 0.1647],
                            ["line_anchor", "top", 0.027612602645529838],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.2239],
                            ["line_anchor", "top", -0.0005778804280586192],
                            [None, 0.2906],
                            ["line_anchor", "top", 0.013664543814365604],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.3482],
                            ["line_anchor", "top", -0.003305153155331364],
                            [None, 0.4408],
                            ["line_anchor", "top", 0.014880997472371982],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.7722],
                            ["line_anchor", "top", 0],
                            [None, 0.9],
                            ["line_anchor", "top", 0.02605627361472096],
                        ]
                    ],
                },
            },
        }
    ],
}

organogenesis_1 = {
    "header_items": {
        "customer_po_date": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Date...................:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.63, 0.08, 0.77, 0.11],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", -0.0001],
                    ["anchor_1", "top", -0.0024],
                    ["anchor_1", "right", 0.1297],
                    ["anchor_1", "bottom", 0.002],
                ]
            ],
        },
        "customer_purchase_order": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Number..............:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.63, 0.07, 0.77, 0.09],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", -0.0005],
                    ["anchor_1", "top", -0.0007],
                    ["anchor_1", "right", 0.1572],
                    ["anchor_1", "bottom", -0.0002],
                ]
            ],
        },
        "net_amount": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales total",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.61, 0.87, 0.7, 0.89],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0273],
                    ["anchor_1", "top", -0.0013],
                    ["anchor_1", "right", 0.1497],
                    ["anchor_1", "bottom", 0.0003],
                ]
            ],
        },
        "order_number": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales order",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.31, 0.18, 0.33],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0306],
                    ["anchor_1", "bottom", 0.0018],
                    ["anchor_1", "right", 0.0376],
                    ["anchor_1", "bottom", 0.0261],
                ]
            ],
        },
        "receiver_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.1, 0.21, 0.18, 0.23],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0035],
                    ["anchor_1", "bottom", -0.0009],
                    ["anchor_1", "right", 0.3572],
                    ["anchor_1", "bottom", 0.0533],
                ]
            ],
        },
        "receiver_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.1, 0.21, 0.18, 0.23],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0054],
                    ["anchor_1", "top", -0.0026],
                    ["anchor_1", "right", 0.3572],
                    ["anchor_1", "bottom", 0.0018],
                ]
            ],
        },
        "supplier_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.1, 0.21, 0.18, 0.23],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0601],
                    ["anchor_1", "top", -0.0699],
                    ["anchor_1", "right", 0.1552],
                    ["anchor_1", "top", -0.0426],
                ]
            ],
        },
        "supplier_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.1, 0.21, 0.18, 0.23],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0601],
                    ["anchor_1", "top", -0.0887],
                    ["anchor_1", "right", 0.1537],
                    ["anchor_1", "top", -0.0729],
                ]
            ],
        },
        "total_amount": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales tax",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.61, 0.9, 0.69, 0.92],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0419],
                    ["anchor_1", "bottom", 0.0028],
                    ["anchor_1", "right", 0.1588],
                    ["anchor_1", "bottom", 0.0144],
                ]
            ],
        },
        "total_tax": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Sales total",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.61, 0.87, 0.7, 0.89],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0364],
                    ["anchor_1", "bottom", 0.0146],
                    ["anchor_1", "right", 0.1532],
                    ["anchor_1", "bottom", 0.0276],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "The sales amount takes into account list prices and your contracted rates for products and any",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                }
            ],
            "headers_exist_on_additional_pages": False,
            "line_anchor": {
                "column_name": "Item",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.8835,
                "right_coordinate": 0.9835,
            },
            "remove_line_regex": "",
            "table_start_regex": "",
            "fields": {
                "line_description": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.0149],
                            ["line_anchor", "top", 0],
                            [None, 0.3651],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.0149],
                            ["line_anchor", "top", 0],
                            [None, 0.2298],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.2647],
                            ["line_anchor", "top", 0],
                            [None, 0.311],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.3835],
                            ["line_anchor", "top", 0],
                            [None, 0.4706],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.8835],
                            ["line_anchor", "top", 0],
                            [None, 0.9835],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
            },
        }
    ],
}

roche = {
    "header_items": {
        "billing_address": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.01, 0.19, 0.05, 0.2],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0812],
                    ["anchor_1", "top", -0.0958],
                    ["anchor_1", "right", 0.3917],
                    ["anchor_1", "top", -0.0333],
                ]
            ],
        },
        "billing_name": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "Acknowledgement",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.42, 0.03, 0.6, 0.06],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.294],
                    ["anchor_1", "bottom", 0.0252],
                    ["anchor_1", "right", -0.1554],
                    ["anchor_1", "bottom", 0.0555],
                ]
            ],
        },
        "customer_po_date": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Acknowledgement",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.42, 0.03, 0.6, 0.06],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0401],
                    ["anchor_1", "bottom", -0.0045],
                    ["anchor_1", "right", 0.1449],
                    ["anchor_1", "bottom", 0.0182],
                ]
            ],
        },
        "customer_purchase_order": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Page",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.85, 0.05, 0.89, 0.07],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.1692],
                    ["anchor_1", "bottom", 0.0133],
                    ["anchor_1", "right", 0.0366],
                    ["anchor_1", "bottom", 0.0291],
                ]
            ],
        },
        "order_number": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Page",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.85, 0.05, 0.89, 0.07],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.1672],
                    ["anchor_1", "bottom", 0.0479],
                    ["anchor_1", "right", 0.035],
                    ["anchor_1", "bottom", 0.0664],
                ]
            ],
        },
        "receiver_address": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.06],
                    ["anchor_1", "bottom", 0.002],
                    ["anchor_1", "right", 0.3669],
                    ["anchor_1", "bottom", 0.0863],
                ]
            ],
        },
        "receiver_name": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "SHIP TO",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.06],
                    ["anchor_1", "top", -0.0066],
                    ["anchor_1", "right", 0.3705],
                    ["anchor_1", "bottom", 0.0183],
                ]
            ],
        },
        "supplier_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SOLD TO",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.01, 0.08, 0.07, 0.1],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0844],
                    ["anchor_1", "top", -0.0466],
                    ["anchor_1", "right", 0.3409],
                    ["anchor_1", "top", -0.0168],
                ]
            ],
        },
        "supplier_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "SOLD TO",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.01, 0.08, 0.07, 0.1],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.0824],
                    ["anchor_1", "top", -0.0639],
                    ["anchor_1", "right", 0.3389],
                    ["anchor_1", "top", -0.0466],
                ]
            ],
        },
        "total_amount": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "Req Del Date",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0433],
                    ["anchor_1", "bottom", 0.0787],
                    ["anchor_1", "right", 0.0041],
                    ["anchor_1", "bottom", 0.0972],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "Total Amount",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                }
            ],
            "headers_exist_on_additional_pages": True,
            "line_anchor": {
                "column_name": "Unit price",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.0406,
                "right_coordinate": 0.0954,
            },
            "remove_line_regex": "",
            "table_start_regex": "",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.095],
                            ["line_anchor", "top", -0.0026075917841620333],
                            [None, 0.225],
                            ["line_anchor", "top", 0.011652116953702052],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.407],
                            ["line_anchor", "top", -0.0033447568784157333],
                            [None, 0.6805],
                            ["line_anchor", "top", 0.011652116953702052],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.6151],
                            ["line_anchor", "bottom", 0],
                            [None, 0.695],
                            ["line_anchor", "bottom", 0.02],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.721],
                            ["line_anchor", "top", -0.004254951053173084],
                            [None, 0.7967],
                            ["line_anchor", "top", 0.008618136371177754],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.8365],
                            ["line_anchor", "bottom", 0],
                            [None, 0.99],
                            ["line_anchor", "bottom", 0.02],
                        ]
                    ],
                },
            },
        }
    ],
}

surgitech = {
    "header_items": {
        "customer_po_date": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Order",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.03, 0.17, 0.07],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.004],
                    ["anchor_1", "bottom", 0.008],
                    ["anchor_1", "right", 0.0898],
                    ["anchor_1", "bottom", 0.0219],
                ]
            ],
        },
        "customer_purchase_order": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "External Document No.",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.3, 0.24, 0.32],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.006],
                    ["anchor_1", "bottom", -0.0015],
                    ["anchor_1", "right", 0.015],
                    ["anchor_1", "bottom", 0.0158],
                ]
            ],
        },
        "net_amount": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Amount Excl. Tax",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.82, 0.39, 0.93, 0.41],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0087],
                    ["anchor_1", "bottom", 0.1607],
                    ["anchor_1", "right", 0.0049],
                    ["anchor_1", "bottom", 0.1779],
                ]
            ],
        },
        "order_number": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Order",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.03, 0.17, 0.07],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.1627],
                    ["anchor_1", "top", 0.0033],
                    ["anchor_1", "right", 0.3866],
                    ["anchor_1", "bottom", -0.0011],
                ]
            ],
        },
        "receiver_address": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "External Document No.",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.3, 0.24, 0.32],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0052],
                    ["anchor_1", "top", -0.1449],
                    ["anchor_1", "right", 0.1672],
                    ["anchor_1", "top", -0.0491],
                ]
            ],
        },
        "receiver_name": {
            "use_model_preds": True,
            "page_num": 0,
            "anchor_1": {
                "regex": "Page",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.09, 0.08, 0.13, 0.11],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0068],
                    ["anchor_1", "bottom", 0.0338],
                    ["anchor_1", "right", 0.2732],
                    ["anchor_1", "bottom", 0.068],
                ]
            ],
        },
        "supplier_address": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Amount Excl. Tax",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.82, 0.39, 0.93, 0.41],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.1095],
                    ["anchor_1", "top", -0.237],
                    ["anchor_1", "right", 0.0112],
                    ["anchor_1", "top", -0.1437],
                ]
            ],
        },
        "supplier_name": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Shipment Method",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.71, 0.3, 0.83, 0.32],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0101],
                    ["anchor_1", "top", -0.1637],
                    ["anchor_1", "right", 0.1066],
                    ["anchor_1", "top", -0.1464],
                ]
            ],
        },
        "total_amount": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "USD Incl. Tax",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.56, 0.63, 0.67, 0.66],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.1543],
                    ["anchor_1", "top", 0.0003],
                    ["anchor_1", "right", 0.2712],
                    ["anchor_1", "bottom", 0.0006],
                ]
            ],
        },
        "total_tax": {
            "use_model_preds": False,
            "page_num": 0,
            "anchor_1": {
                "regex": "Amount Excl. Tax",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.82, 0.39, 0.93, 0.41],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0067],
                    ["anchor_1", "bottom", 0.1964],
                    ["anchor_1", "right", 0.0069],
                    ["anchor_1", "bottom", 0.2088],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "Subtotal",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": False,
                }
            ],
            "headers_exist_on_additional_pages": False,
            "line_anchor": {
                "column_name": "No.",
                "selection_index": 2,
                "vertical_alignment": "top",
                "left_coordinate": 0.7639,
                "right_coordinate": 0.8686,
            },
            "remove_line_regex": "",
            "table_start_regex": "",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.0788],
                            ["line_anchor", "top", 0],
                            [None, 0.1576],
                            ["line_anchor", "top", 0.08044444696969705],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.1592],
                            ["line_anchor", "top", -0.0015517651515151165],
                            [None, 0.389],
                            ["line_anchor", "top", 0.06996338636363642],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.3871],
                            ["line_anchor", "top", 0.0002664166666667134],
                            [None, 0.4435],
                            ["line_anchor", "top", 0.019054295454545456],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.5082],
                            ["line_anchor", "top", -0.0007840909090909287],
                            [None, 0.5969],
                            ["line_anchor", "top", 0.01800378787878787],
                        ]
                    ],
                },
                "line_tax": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.7016],
                            ["line_anchor", "top", -0.002299242424242509],
                            [None, 0.7561],
                            ["line_anchor", "top", 0.01723611363636368],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.7639],
                            ["line_anchor", "top", 0],
                            [None, 0.8686],
                            ["line_anchor", "top", 0.08044444696969705],
                        ]
                    ],
                },
            },
        }
    ],
}

applied_medical = {
    "header_items": {
        "customer_purchase_order": {
            "anchor_1": {
                "regex": "Customer PO",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "top", 0.0],
                    [None, 1],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
        "customer_po_date": {
            "anchor_1": {
                "regex": "Date",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.55, 0.19, 0.59, 0.21],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "top", 0.0],
                    ["anchor_1", "right", 0.1321],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
        "order_number": {
            "anchor_1": {
                "regex": "information",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.5, 0.14, 0.6, 0.16],
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "bottom", 0.0062],
                    ["anchor_1", "right", 0.1086],
                    ["anchor_1", "bottom", 0.0179],
                ]
            ],
        },
        "invoice_date": {
            "anchor_1": {
                "regex": "Confirmation",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.65, 0.08, 0.89, 0.12],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", 0.0555],
                    ["anchor_1", "bottom", 0.047],
                    ["anchor_1", "right", -0.0864],
                    ["anchor_1", "bottom", 0.0587],
                ]
            ],
        },
        "billing_name": {
            "anchor_1": {
                "regex": "Bill-to:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.03, 0.29, 0.09, 0.31],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0046],
                    ["anchor_1", "bottom", 0.0214],
                    ["anchor_1", "right", 0.139],
                    ["anchor_1", "bottom", 0.0331],
                ]
            ],
        },
        "billing_address": {
            "anchor_1": {
                "regex": "Bill-to:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.03, 0.29, 0.09, 0.31],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0046],
                    ["anchor_1", "bottom", 0.0365],
                    ["anchor_1", "right", 0.1917],
                    ["anchor_1", "bottom", 0.0937],
                ]
            ],
        },
        "supplier_address": {
            "anchor_1": {"regex": "Phone:", "ignore_case": False, "selection_index": 0},
            "search_areas": [
                [
                    ["anchor_1", "left", -0.01],
                    [None, 0],
                    ["anchor_1", "right", 0.4],
                    ["anchor_1", "top", 0.0],
                ]
            ],
        },
        "receiver_name": {
            "anchor_1": {
                "regex": "Ship-to:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.03, 0.14, 0.1, 0.16],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0046],
                    ["anchor_1", "bottom", 0.0214],
                    ["anchor_1", "right", 0.2],
                    ["anchor_1", "bottom", 0.0331],
                ]
            ],
        },
        "receiver_address": {
            "anchor_1": {
                "regex": "Ship-to:",
                "ignore_case": False,
                "selection_index": 0,
                "router_coordinates": [0.03, 0.14, 0.1, 0.16],
            },
            "search_areas": [
                [
                    ["anchor_1", "left", -0.0046],
                    ["anchor_1", "bottom", 0.03],
                    ["anchor_1", "right", 0.2],
                    ["anchor_1", "bottom", 0.1],
                ]
            ],
        },
        "net_amount": {
            "anchor_1": {
                "regex": "Sub Total",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0],
                    ["anchor_1", "top", 0.0],
                    [None, 1],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
        "total_amount": {
            "anchor_1": {
                "regex": "Total Amount",
                "ignore_case": False,
                "selection_index": 0,
            },
            "search_areas": [
                [
                    ["anchor_1", "right", 0.05],
                    ["anchor_1", "top", 0.0],
                    [None, 1],
                    ["anchor_1", "bottom", 0.0],
                ]
            ],
        },
    },
    "tables": [
        {
            "table_end": [
                {
                    "regex": "Sub Total",
                    "ignore_case": False,
                    "selection_index": 0,
                    "document_table_end": True,
                }
            ],
            "headers_exist_on_additional_pages": True,
            "line_anchor": {
                "column_name": "Amount",
                "selection_index": 0,
                "vertical_alignment": "top",
                "left_coordinate": 0.85,
                "right_coordinate": 1,
            },
            "remove_line_regex": "",
            "table_start_regex": "--------------------------------------------------------------------------------------------------------------------------------------------",
            "fields": {
                "line_item_id": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.094],
                            ["line_anchor", "top", 0],
                            [None, 0.1768],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_description": {
                    "read_to_next_line_item": True,
                    "search_areas": [
                        [
                            [None, 0.094],
                            ["line_anchor", "bottom", 0],
                            [None, 0.4042],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_quantity": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.4819],
                            ["line_anchor", "top", 0],
                            [None, 0.514],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.5186],
                            ["line_anchor", "top", 0],
                            [None, 0.5533],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_unit_price": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.6974],
                            ["line_anchor", "top", 0],
                            [None, 0.7479],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
                "line_amount": {
                    "read_to_next_line_item": False,
                    "search_areas": [
                        [
                            [None, 0.851],
                            ["line_anchor", "top", 0],
                            [None, 0.9016],
                            ["line_anchor", "bottom", 0],
                        ]
                    ],
                },
            },
        }
    ],
}

# Create a list to store the JSON data
template_dict_list = [
    arthrex,
    lake_court,
    medline,
    organogenesis_0,
    organogenesis_1,
    roche,
    surgitech,
    applied_medical,
]


#####################################################################################
# Post-process code
#####################################################################################


def post_process_predictions(
    model_preds: Union[Dict[str, Dict[str, Dict[str, Dict[str, str]]]], pd.DataFrame],
    top_n_preds: Union[Dict[str, Union[List[float], str]], List[str]],
    token_merge_type: str = "MIXED_MERGE",
    token_merge_xdist_regular: float = 1.0,
    label_merge_x_regular: str or None = None,
    token_merge_xydist_regular: float = 1.0,
    label_merge_xy_regular: str or None = None,
    token_merge_xdist_wide: float = 1.5,
    label_merge_x_wide: str or None = None,
    constraint_dict: Dict[str, int] = {},
    output_labels: str = "INCLUDE_O",
    parse_line_items: bool = False,
    line_item_completeness: float = 0.6,
    try_templates: bool = False,
    template_dicts: List = None,
    templates_input_dir: str = "./",
    templates_use_model_preds_mapping: Dict[str, List[str]] = {},
    use_camelot_tables: bool = False,
    images_dir_camelot: str = "",
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Post-processes the model predictions for Document Information Extraction.

    Parameters:
    -----------
    model_preds: pandas DataFrame
        A DataFrame containing the model predictions, with columns for the document ID, page ID, coordinates,
        text content, label, and OCR confidence.
    top_n_preds: list of str
        A list of the predicted label names for the model.
    token_merge_type: str, optional (default='MIXED_MERGE')
        The type of token merging to use, either 'MIXED_MERGE' or 'NO_MERGE'.
    token_merge_xdist_regular: float, optional (default=1.0)
        The distance threshold for merging tokens on the x-axis, between 0.0 and 1.0.
    label_merge_x_regular: str or None, optional (default=None)
        The label names to merge on the x-axis, either 'ALL' or a string of labels separated by '|'.
    token_merge_xydist_regular: float, optional (default=1.0)
        The distance threshold for merging tokens on the x- and y-axes, between 0.0 and 1.0.
    label_merge_xy_regular: str or None, optional (default=None)
        The label names to merge on the x- and y-axes, either 'ALL' or a string of labels separated by '|'.
    token_merge_xdist_wide: float, optional (default=1.5)
        The distance threshold for wide token merging on the x-axis, between 1.0 and 10.0.
    label_merge_x_wide: str or None, optional (default=None)
        The label names to merge for wide token merging on the x-axis, either 'ALL' or a string of labels separated by '|'.
    output_labels: str or list, optional (default='INCLUDE_O')
        The type of output labels to include. Options are 'INCLUDE_O' (include all labels), 'EXCLUDE_O' (exclude "O" labels),
        or a list of label names to include.
    parse_line_items: bool, optional (default=False)
        Whether to parse line items from the input data.
    line_item_completeness: float, optional (default=0.6)
        The completeness threshold for parsing line items, between 0.0 and 1.0.
    try_templates: bool, optional (default=False)
        Whether to try extracting data using templates.
    templates_dict_dir: str, optional (default='')
        The directory path to the template configuration files.
    templates_input_dir: str, optional (default='./')
        The directory path to the input files for templates.
    templates_use_model_preds_mapping: dict, option (default={})
        A dictionary of similar labels to be used when `use_model_preds=True`. This is useful for properly parsing name/address fields. For example, if the templates specify the `receiver_address` is in a particular location, but `use_model_preds=True` and the model predicts the text in that location as the `billing_address` we will convert the `billing_address` text to be the `receiver_address`.
    use_camelot_tables: bool, optional (default=False)
        Whether to use Camelot tables for extracting line items.
    images_dir_camelot: str, optional (default='')
        The directory path to the images for Camelot tables.

    Returns:
    --------
    dict
        A dictionary containing the processed data for each document, with document IDs as keys
    """

    if isinstance(model_preds, pd.DataFrame):
        # in post processor, input a dataframe converted from an annotation set and the class name
        df_batch, class_names = model_preds, list(top_n_preds)
    # backward compatibility
    else:
        df_batch, class_names = pp.via2df(model_preds, top_n_preds)

    docs = df_batch.groupby("doc_id")
    final_results = {}

    for j, (doc_ID, doc_df) in enumerate(docs):
        start_time = time.time()
        print(
            f"\n\n---------- Processing document {j}/{len(docs)} : {doc_ID} ----------"
        )

        # Merge nearby boxes of same label - page by page
        (
            doc_ids,
            page_ids,
            coords,
            texts,
            labels,
            ocr_confidence,
            class_probabilities,
        ) = ([], [], [], [], [], [], [])
        for k, (page_ID, page_df) in enumerate(doc_df.groupby("page_id")):
            page_df["page"] = k
            page_df = box_process(page_df)  # sort boxes in a consistent order
            texts_list = page_df["text"].tolist()
            labels_list = page_df["label"].tolist()
            ocr_score_list = page_df["ocr_confidence"].tolist()
            page_df["xmax"] = page_df["x"] + page_df["width"]
            page_df["ymax"] = page_df["y"] + page_df["height"]
            mean_h = page_df.height.mean() + page_df.height.std()
            texts_boxes_list = page_df[["y", "x", "ymax", "xmax"]].values.tolist()
            for class_name in class_names:
                page_df[class_name] = page_df[class_name].astype(float)
            class_probabilities_list = page_df[class_names].values.tolist()

            if token_merge_type == "MIXED_MERGE":
                (
                    texts_list,
                    texts_boxes_list,
                    labels_list,
                    ocr_score_list,
                    class_probabilities_list,
                ) = merge_tokens(
                    texts=texts_list,
                    texts_boxes=texts_boxes_list,
                    labels=labels_list,
                    ocr_conf=ocr_score_list,
                    class_probabilities=class_probabilities_list,
                    token_merge_xdist_regular=token_merge_xdist_regular,
                    token_merge_xydist_regular=token_merge_xydist_regular,
                    token_merge_xdist_wide=token_merge_xdist_wide,
                    label_merge_x_regular=label_merge_x_regular,
                    label_merge_x_wide=label_merge_x_wide,
                    label_merge_xy_regular=label_merge_xy_regular,
                    mean_h=mean_h,
                    class_names=class_names,
                )

            doc_ids.extend([doc_ID] * len(texts_list))
            page_ids.extend([page_ID] * len(texts_list))
            coords.extend(texts_boxes_list)
            texts.extend(texts_list)
            labels.extend(labels_list)
            ocr_confidence.extend(ocr_score_list)
            class_probabilities.extend(class_probabilities_list)

        df = pd.DataFrame(
            {
                "doc_id": doc_ids,
                "page_id": page_ids,
                "xmin": np.array(coords)[:, 1],
                "ymin": np.array(coords)[:, 0],
                "xmax": np.array(coords)[:, 3],
                "ymax": np.array(coords)[:, 2],
                "text": texts,
                "label": labels,
                "ocr_confidence": ocr_confidence,
            }
        )

        if len(class_probabilities) > 0:
            probabilities_df = pd.DataFrame(class_probabilities, columns=class_names)
            df["probability"] = probabilities_df.max(axis=1)
            df = df.join(probabilities_df)

        if parse_line_items:
            print("Executing line item parsing method..")
            try:
                pdf_fname = ""
                if use_camelot_tables:
                    # log.info('Trying to parse line items using camelot tables..')
                    print("Trying to parse line items using camelot tables..")
                    pdf_fname = glob(f"{templates_input_dir}/{doc_ID}.[Pp][Dd][Ff]")[0]

                df = pp.parse_lines(
                    df,
                    doc_df,
                    line_item_completeness=line_item_completeness,
                    use_camelot_tables=use_camelot_tables,
                    pdf_fname=pdf_fname,
                    images_dir=images_dir_camelot,
                    money_token_parse_method=None,
                )

            except Exception as e:
                traceback.print_exc()
                print(f"Line item could not be parsed: {doc_ID}")
                print(e)
                df["line"] = None

        if constraint_dict:
            df = pp.constrain_predictions(df, constraint_dict)

        if output_labels == "EXCLUDE_O":
            df = df.loc[df["label"].str.strip().str.len() > 0].reset_index(drop=True)
            print("output all labels")
        elif isinstance(output_labels, list):
            df = df.loc[df["label"].isin(output_labels)].reset_index(drop=True)
            print(f"output labels: {output_labels} ")

        print("   *** Results using ML method ***")
        if df is not None:
            pp.print_output(df, line_items=parse_line_items, verbose=verbose)
            df.reset_index(inplace=True)
        else:
            df = pd.DataFrame()

        # ===== Templates ======
        if try_templates and os.path.isdir(templates_input_dir):
            try:
                print("*** Checking templates ***")
                fname = glob(f"{templates_input_dir}/{doc_ID}.[Pp][Dd][Ff]")[0]

                doc_dict, use_model_preds = process_templates(fname, template_dicts)
                if doc_dict:
                    print("   *** Template matched. Results using templates ***")
                    doc_df = dict2csv(doc_ID, doc_dict)
                    if use_model_preds:
                        keep_model_preds = []
                        for use_model_preds_label in use_model_preds:
                            label_list = []
                            for (
                                key,
                                label_list,
                            ) in templates_use_model_preds_mapping.items():
                                if use_model_preds_label in label_list:
                                    break

                            label_model_preds = df[df["label"].isin(label_list)]
                            temp_preds = doc_df[
                                doc_df["label"] == use_model_preds_label
                            ]

                            for i, row in label_model_preds.iterrows():
                                ious = temp_preds.apply(
                                    lambda x: bb_intersection_over_union(
                                        [x.xmin, x.ymin, x.xmax, x.ymax],
                                        [row.xmin, row.ymin, row.xmax, row.ymax],
                                    ),
                                    axis=1,
                                )

                                if any(ious > 0.2):
                                    row.label = use_model_preds_label
                                    keep_model_preds.append(pd.DataFrame(row).T)

                        if keep_model_preds:
                            keep_model_preds = pd.concat(keep_model_preds).reset_index(
                                drop=True
                            )
                            doc_df = doc_df[
                                ~doc_df["label"].isin(keep_model_preds["label"])
                            ]
                            doc_df = pd.concat([doc_df, keep_model_preds])

                    if not doc_df.empty:
                        pp.print_output(doc_df, line_items=parse_line_items)
                else:
                    print("  *** No template found. using ML results ***")
                    doc_df = df

            except Exception as e:
                print(
                    f"Error while using templates: {doc_ID}. Using ML results instead."
                )
                print(e)
                doc_df = df
        # ===== Templates ======
        else:
            doc_df = df
        final_results.update({doc_ID: doc_df})

        print(f"post-processing time: {time.time() - start_time} seconds")

    return final_results

import os
import sys
import logging
from argparse import ArgumentParser, Namespace, ArgumentTypeError

from typing import Any, Optional, Literal

import json
import pandas as pd
import numpy as np

import pathlib
import regex
from glom import glom, flatten
import copy
from collections import Counter


def natural_sort(input_list: list[str]) -> list[str]:
    """
    Sorts list of strings by natural sort (numbers then letters); same as VIA

    Arguments
    -----
    input_list: list[str]
        list of strings

    Returns
    -----
    input_list: list[str]
        sorted list of strings

    """


    ## from https://www.w3resource.com/python-exercises/data-structures-and-algorithms/python-search-and-sorting-exercise-38.php
    def alphanum_key(key):
        return [int(s) if s.isdigit() else s.lower() for s in regex.split("([0-9]+)", key)]
    return sorted(input_list, key=alphanum_key)


def float_0_1(input_fraction: str) -> float:
    """
    Checks whether input str can be converted to float between 0 and 1

    Arguments:
    -----
    input_fraction: str
        input float as string

    Returns
    -----
    input_fraction: float
        input string as float

    Raises
    -----
    ValueError
        when not a float between 0 < x <= 1

    """

    try:
        input_fraction = float(input_fraction)
    except ValueError:
        raise ArgumentTypeError(f"Fraction {input_fraction} cannot be made into a float")
    
    if input_fraction <=1 and input_fraction > 0:
        return input_fraction
    else:
        raise ArgumentTypeError(f"Fraction must be 0 < f <= 1, {input_fraction} is not")


def pos_numb(number: str) -> int:
    """
    Check whether input string is a positive integer

    Arguments:
    -----
    number: str
        input number as string

    Returns
    -----
    number: int
        input number as integer

    Raises
    ----
    ValueError
        when not a positive integer

    """


    try:
        number = int(number)
    except ValueError:
        raise ArgumentTypeError(f"Number {number} cannot be made into an int")
    
    if isinstance(number,int) and number > 0:
        return number    
    else:
        raise ArgumentTypeError(f"Number {number} is not an acceptable int")


def setup_env(args: Namespace) -> None:
    """
    Initializes logging to 'logs' directory while also allowing stdin 

    Arguments:
    -----
    args: Namespace
        args.level: str
            either INFO or DEBUG
        args.log_name: str
            name of log path

    """


    results = None
    if not os.path.exists('logs'):
        os.mkdir('logs')
        results = True

    
    logging_dict = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }   
    
    logging.basicConfig(
        filename=f"logs/{args.log_name}.log",
        level=logging.DEBUG,
        format = "%(asctime)s  |  %(levelname)s  |  %(message)s",
        force = True
    )
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_dict[args.level])
    formatter = logging.Formatter("%(asctime)s  |  %(levelname)s  |  %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    

    if results:
        logging.debug(f"Making \"logs\"")
        
    logging.debug(f"Log file name {args.log_name}.log")


def load_json(json_path: str) -> dict[str, Any]:
    """
    Load JSON VIA annotation from string path

    Arguments
    -----
    json_path: str
        path to VIA annotation set
    
    Returns
    -----
    json_data: dict[str, Any]
        VIA annotation as python dict
    """
    logging.debug(f"Loading JSON from {json_path}")

    with open(json_path, 'r', encoding="utf-8") as f:
        json_data = json.loads(f.read())
    return json_data



def exclusion_list(
        main_via: dict[str, Any],
        file_list: list[str]
    ) -> list[str]:
    """
    Create list of pages, e.g. {file_name}+{page_numb}.{ext}{size}, that should be excluded from subsetting

    Arguments
    -----
    main_via: dict[str, Any]
        VIA annotation set
    file_list: list[str]
        list of text, csv, VIA json files, or page/file names to exclude

    Returns
    -----
    exclusion: list[str]
        list of pages like {file_name}+{page_numb}.{ext}{size} to be excluded from

    """
    
    logging.info(f"Loading pages to exclude from {file_list}")
            
    page_ext = [".png",".jpg",".jpeg", ".pdf", ""]
            
    exclusion = []
    unique_names = []
            
    ## open various files and do intersection on set of returned possible names
    for file in file_list:
        ext = pathlib.Path(file).suffix
        if ext == ".csv":
            unique_names = natural_sort(list(set(natural_sort(pd.read_csv(file).values.flatten().tolist())) | set(unique_names)))
        elif ext == ".txt":
            with open(file,'r') as f:
                lines = f.readlines()
            unique_names = natural_sort(list(set(natural_sort(lines)) | set(unique_names)))
        elif ext == ".json":
            unique_names = natural_sort(list( set(natural_sort(load_json(file)["_via_image_id_list"])) | set(unique_names)))
        elif ext in page_ext:
            unique_names = natural_sort(list( set([file]) | set(unique_names) ))
        else:
            logging.warning(f"File extension {ext} in {file} not supported for subsetting")
            raise SystemExit(1)
        
    ## loop through each unique name and find valid pages
    ## see this https://stackoverflow.com/questions/3640359/regular-expressions-search-in-list
    for unique_name in unique_names:
        unique_path = pathlib.Path(unique_name)
        
        if "+" in unique_path.stem:
            if unique_path.suffix == "":
                valid_names = list(filter(
                    regex.compile(f"{regex.escape(unique_path.stem)}\..*").match,
                    main_via["_via_image_id_list"]
                    ))
            else:
                valid_names = list(filter(
                    regex.compile(f"{regex.escape(unique_path.stem)}{regex.escape(unique_path.suffix)}.*").match,
                    main_via["_via_image_id_list"]
                    ))
            
        else:
            valid_names = list(filter(
                regex.compile(f"{regex.escape(unique_path.stem)}\+[0-9]*{regex.escape(unique_path.suffix)}.*").match,
                main_via["_via_image_id_list"]
                ))
            
        ## choose your favorite: https://www.geeksforgeeks.org/python-union-two-lists/#
        ## exclusion is a list of pages, e.g, "name+0.jpg12312"
        exclusion = natural_sort(list(
            set(natural_sort(valid_names)) | set(natural_sort(exclusion))
        ))
    
    msg = f"Excluding {exclusion}" if exclusion else "Nothing to exclude"
    logging.debug(msg=msg)
    
    return exclusion


def via_template() -> dict[str, Any]:
    """
    Blank VIA template

    Returns
    -----
    template: dict[str, Any]
        empty VIA template
    """

    template = {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 1,
                    "show_region_shape": True,
                    "show_image_policy": "all"
                },
                "image": {
                    "region_label": "label",
                    "region_color": "__via_default_region_color__",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION"
                }
            },
            "core": {
                "buffer_size": 18,
                "filepath": {},
                "default_filepath": ""
            },
            "project": {
                "name": ""
            }
        },
        "_via_img_metadata": {},
        "_via_attributes": {},
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": []
    }
    
    return template


def merge(
        main_via: dict[str, Any],
        input_list: list[str]
    ) -> dict[str, Any]:
    """
    Merge bounding boxes and options of main VIA and list of other VIA sets
    
    Argument
    -----
    main_via: dict[str, Any]
        Dictionary of main VIA to merge
    input_list: list[str]
        List of JSON annotation set names wished to be merged with main
        
    Returns
    -----
    main_via: dict[str, Any]
        Merged main VIA    
    """


    
    def options_union(
            main_attributes: dict[str, Any], 
            merge_attributes: dict[str, Any],
            text: Optional[bool] = False
        ) -> dict[str, Any]:
        """
        Union of VIA attribute options
        
        Arguments
        -----
        main_attributes: dict[str, Any]
            Dictionary of options from the main VIA annotation set
        merge_attributes: dict[str, Any]
            Dictionary of options from VIA annotation set wanting to be merged
        text: Optional[bool]
            Boolean value of whether merging text or not
            
        Returns
        -----
        new_dict: dict[str, Any]
            Attributes dictionary with union of options
        """
        
        label_dict = {
            "type": "radio" if not text else "text",
            "description": "",
            "default_options": {}
        }
        
        if not text:
            label_dict["options"] = {}
        
        new_dict = copy.deepcopy(label_dict)
        
        options_union = list(
            set(natural_sort(list(main_attributes["options"].keys()))) | set(natural_sort(list(merge_attributes["options"].keys())))
        )
        
        for key in options_union:
            new_dict["options"][key] = ""
        
        return new_dict
    
    
    def boundingbox_union(
            main_via: dict[str, Any],
            new_img_metadata: dict[str, Any],
            new_doc_id: str,
            iou_threshold: Optional[float] = 0.05 
        ) -> dict[str, Any]:
        """
        Union of bounding boxes of main annotation set and single page annotations
        
        Arguments
        -----
        main_via: dict[str, Any]
            Main VIA annotation set
        new_img_metadata: dict[str, Any]
            Single page annotations from annotation set to be merged
        new_doc_id: str
            Document ID of new annotation set (without the size)
        iou_threshold: Optional[float] = 0.05
            if iou value of bounding boxes greater than iou_threshold it won't be merged, i.e. repeat annotation
        
        Returns
        -----
        main_via: dict[str, Any]
            Merged annotation set
        """
        
        def iou(
                box_A: dict[str, str | int],
                box_B: dict[str, str | int]
            ) -> int | float:
            """
            Calculate Intersection over Union (IOU) of two bounding boxes
            
            Arguments
            -----
            box_A: dict[str, str | int]
                VIA bounding box annotation dictionary
            box_b: dict[str, str | int]
                VIA bounding box dictionary
            Returns
            -----
            IOU: int | float
                Value of IOU
            """

            ## see https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
            
            xA = max(box_A["x"], box_B["x"])
            yA = max(box_A["y"], box_B["y"])
            xB = min(box_A["width"] + box_A["x"], box_B["width"] + box_B["x"])
            yB = min(box_A["height"] + box_A["y"], box_B["height"] + box_B["y"])
            
            I = abs(max((xB - xA, 0)) * max((yB - yA), 0))
            if I == 0:
                return 0
            
            U = abs(box_A["width"]*box_A["height"]) + abs(box_B["width"]*box_B["height"]) - I
            
            return I/float(U)
        
        
        if new_doc_id not in main_via["_via_image_id_list"]:
            ## if document not present in main, add all boxes
            logging.debug(f"Merging entire page {new_doc_id}")
            main_via["_via_img_metadata"][new_doc_id] = new_img_metadata[new_doc_id]
            main_via["_via_image_id_list"].append(new_doc_id)
            
        else:
            logging.debug(f"Merging selected bboxes from {new_doc_id}")
            
            output_matrix = np.zeros((
                len(main_via["_via_img_metadata"][new_doc_id]["regions"]), len(new_img_metadata[new_doc_id]["regions"])
                ))

            if not output_matrix.size > 0:
                pass

            for main_region_idx, main_region in enumerate(main_via["_via_img_metadata"][new_doc_id]["regions"]):
                output_matrix[main_region_idx,:] = glom(
                    target = new_img_metadata[new_doc_id]["regions"], 
                    spec = ("*.shape_attributes", [lambda r: iou(main_region["shape_attributes"], r)])
                    )
            for new_region_idx in range(len(new_img_metadata[new_doc_id]["regions"])):
                if all(output_matrix[:,new_region_idx] < iou_threshold):
                    main_via["_via_img_metadata"][new_doc_id]["regions"].append(new_img_metadata[new_doc_id]["regions"][new_region_idx])

                
        return main_via
        
        
    for via_to_merge in input_list:
        logging.info(f"Merging {via_to_merge}")
        to_merge_data = load_json(via_to_merge)
        
        ## delete text if not present in all anno sets
        main_text_exist = "text" in main_via["_via_attributes"]["region"].keys()
        merge_text_exist = "text" in to_merge_data["_via_attributes"]["region"].keys()
        
        if not main_text_exist or not merge_text_exist:
            logging.info("Text attribute not found in both, deleting text attribute")
            if not main_text_exist:
                main_via = delete(
                    main_via = main_via,
                    attribute = {"region":"text"}
                )
            
            if not merge_text_exist:
                to_merge_data = delete(
                    main_via = to_merge_data,
                    attribute = {"region":"text"}
                )
        
        ## ensure all classes present in main anno set
        for attribute in ["label","page","page_class"]:
            if attribute in to_merge_data["_via_attributes"]["region"].keys():
                main_via["_via_attributes"]["region"][attribute] = options_union(
                    main_attributes = main_via["_via_attributes"]["region"][attribute],
                    merge_attributes = to_merge_data["_via_attributes"]["region"][attribute]
                    )
                
            if attribute in to_merge_data["_via_attributes"]["file"].keys():
                main_via["_via_attributes"]["file"][attribute] = options_union(
                    main_attributes = main_via["_via_attributes"]["file"][attribute],
                    merge_attributes = to_merge_data["_via_attributes"]["file"][attribute]
                    )
        
        ## add bounding boxes/new pages
        for new_doc_id in to_merge_data["_via_image_id_list"]:
            main_via = boundingbox_union(
                main_via = main_via,
                new_img_metadata = to_merge_data["_via_img_metadata"],
                new_doc_id = new_doc_id
            ) 
          
        ## update image id list with ones not present
        main_via["_via_image_id_list"] = natural_sort(list(
            set(natural_sort(main_via["_via_image_id_list"])) | set(natural_sort(to_merge_data["_via_image_id_list"]))
            ))
    
    return main_via


def get_distribution(main_via: dict[str, Any]) -> dict[str, Any]:
    """
    Counts the number of every annotation class in annotation set by individual pages
    
    Arguments
    -----
    main_via: dict[str, Any]
        Main VIA annotation set
    
    Returns
    -----
    distribution_dict: dict[str, Any]
        Dictionary of presence of class in a given page (>0 present, 0 not present)    
    """

    ## meta_key: if file or region
    ## key: class, page_class, label, etc.
    ## option: specific types for label, class, etc., e.g., invoice_date or referral
        
    ## get page/region classes for file and region
    meta_keys = {
        meta_key: [key for key in main_via["_via_attributes"][meta_key].keys() 
                    if key != "text" and "options" in main_via["_via_attributes"][meta_key][key].keys()] 
        for meta_key in main_via["_via_attributes"].keys()
    }
    key_template = {}
    distribution_dict = {"keys": {}, "page_names": {}}
        
    for meta_key in meta_keys.keys():
        key_template[meta_key] = {}
        distribution_dict["keys"][meta_key] = {}
        for key in meta_keys[meta_key]:
            key_template[meta_key][key] = {}
            distribution_dict["keys"][meta_key][key] = {}
            for option in main_via["_via_attributes"][meta_key][key]["options"].keys():
                key_template[meta_key][key][option] = 0
                distribution_dict["keys"][meta_key][key][option] = ""
            key_template[meta_key][key]["classes_present"] = 0
        
    for page_name in main_via["_via_image_id_list"]:
        distribution_dict["page_names"][page_name] = copy.deepcopy(key_template)
        for meta_key in meta_keys.keys():
            if meta_key == "region":
                for key in meta_keys[meta_key]:
                    for region in main_via["_via_img_metadata"][page_name]["regions"]:
                        ## check to see which class is present and add to page's stats
                        if key in region["region_attributes"].keys():
                            if region["region_attributes"][key] in distribution_dict["page_names"][page_name][meta_key][key].keys():
                                distribution_dict["page_names"][page_name][meta_key][key][region["region_attributes"][key]] += 1
                        
                    ## records total number of classes present in page for each key
                    distribution_dict["page_names"][page_name][meta_key][key]["classes_present"] = sum(glom(
                        target = distribution_dict["page_names"][page_name][meta_key][key], 
                        spec = ("*", [lambda numb: False if numb == 0 else True])
                    ))
                        
            if meta_key == "file":
                for key in meta_keys[meta_key]:
                    if key in main_via["_via_img_metadata"][page_name]["file_attributes"].keys():
                        distribution_dict["page_names"][page_name][meta_key][key][main_via["_via_img_metadata"][page_name]["file_attributes"][key]] += 1 
                        distribution_dict["page_names"][page_name][meta_key][key]["classes_present"] = main_via["_via_img_metadata"][page_name]["file_attributes"][key]     
        
    return distribution_dict


def subset(
        main_via: dict[str, Any],
        base_name: str,
        number: Optional[list[int]] = [],
        fractions: Optional[list[float]] = [],
        from_name: Optional[list[str]] = [],
        distribution: Optional[Literal["random","token","page"]] = "random",
        subset_by: Optional[Literal["page","document"]] = "page",
        exclusion: Optional[list[str]] = [],
        main_seed: Optional[int] = 1234,
        total_subsets: Optional[int] = 1000
    ) -> dict[str, dict[str, Any]]: 
    """
    Create page or document-level subsets of a VIA annotation set
    
    Arguments
    -----
    main_via: dict[str, Any]
        Main VIA annotation set to be subset from
    base_name: str
        Name from which subsets will derive their name from
    number: Optional[list[int]] = []
        List of amounts of pages or documents which will be subsetted
    fractions: Optional[list[float]] = []
        List of fractions of pages or documents which will be subsetted
    from_name: Optional[list[str]] = []
        List of viable page names from which a subset will be made from
    distribution: Optional[Literal["random","token","page"]] = "random"
        How distribution of classes in main_via will be preserved in subsetting
            "random" = randomly distributed
            "token" = attempts to preserve distribution of token classes in pages/docs
            "page" = attempts to preserver distribution of page classes in pages/docs
    subset_by: Optional[Literal["page","document"]] = "page"
        Whether to subset by "page" or by "document"
    exclusion: Optional[list[str]] = []
        Pages to be excluded from during subsetting
    main_seed: Optional[int] = 1234
        Seed for subsetting
    total_subsets: Optional[int] = 1000
        Total number of sub-seeds to choose for each subset subsetted
        
    Returns
    -----
        subsets: dict[str, dict[str, Any]]
            Subsets returned with name as key and the subsetted VIA annotation set as value
    """

    def populate_subset(
            main_via: dict[str, Any],
            subset_name: str,
            to_include: list[str]
        ) -> dict[str, Any]:
        """
        Populate the new subset from list of pages to include
        
        Arguments
        -----
        main_via: dict[str, Any]
            Main VIA annotation set to get annotations from
        subset_name: str
            Name of subset
        to_include: list[str]
            List of pages pull annotations from
        
        Returns
        -----
        subset_w_boxes: dict[str, Any]
            Filled subset annotation set with annotations
        """

        logging.debug(f"Populating subset {subset_name}")
        
        global via_template
        
        subset_w_boxes = via_template()
        
        subset_w_boxes["_via_settings"]["project"]["name"] = subset_name
        subset_w_boxes["_via_attributes"] = copy.deepcopy(main_via["_via_attributes"])
        
        pages_to_include = []
        for page in to_include:
            if regex.findall("\.[a-z]{3,4}",page):
                pages_to_include.append(page)
            else:
                pages_to_include.extend([regex.findall(f"{regex.escape(page)}\+[0-9]*\.[a-zA-Z0-9]*",item) for item in main_via["_via_image_id_list"]])
            
        pages_to_include = list(set(pages_to_include))
            
        for page in pages_to_include:
            subset_w_boxes["_via_image_id_list"].append(page)
            subset_w_boxes["_via_img_metadata"][page] = copy.deepcopy(main_via["_via_img_metadata"][page])
        
        return subset_w_boxes
    
    
    def choose_rand_pages(
            main_via_image_list: list[str],
            distribution: Literal["random","token","page"],
            subset_by: Literal["page","document"],
            distribution_dict: dict[str, Any],
            seed: int,
            base_name: str,
            exclusion: Optional[list[str]] = [],
            fraction: Optional[float] = [],
            number: Optional[int] = []
        ) -> list[str]:
        """
        Choose pages to fill subsets from
        
        Arguments
        -----
        main_via_images_list: list[str]
            List of all pages from main annotation set
        distribution: Literal["random","token","page"]
            How the distribution of classes in pages is preserved
        seed: int
            Seed used for choosing pages
        base_name: str
            Base name for subsets
        exclusion: Optional[list[str]] = []
            List of pages to exclude from subsetting
        fraction: Optional[float] = []
            Fractional amount of pages or documents to subset to
        number: Optional[int] = []
            Number of pages or documents to subset to
        
        Returns
        -----
        pages: list[str]
            List of chosen pages for the subset
        
        """

        def get_doc_names(dirty_id_list: list[str]) -> set[str]:
            """
            Returns set of document names from page names that have size, page number, extension
            
            Arguments
            -----
            dirty_id_list: list[str]
                list of pages with size, page number, ext to be removed
            
            Returns
            -----
                cleaned_id_set: set[str]
                    set of document names
            """

            page_r = regex.compile("\+[0-9]+")
            size_r = regex.compile("(?<=\.[a-zA-Z]{3,4})[0-9]+")
            ext_r = regex.compile("\.[a-z]{3,4}")

            no_page_list = [
                regex.sub(page_r, "", page) for page in dirty_id_list
            ]
            
            no_page_size_list = [
                regex.sub(size_r, "", page) for page in no_page_list
            ]
            
            cleaned_id_list = [
                regex.sub(ext_r,"",page) for page in no_page_size_list
            ]
            
            return set(cleaned_id_list)
        
        def find_page_names(
                doc_names: list[str],
                page_list: list[str]
            ) -> set[str]:
            """
            Returns set of page names for a given list of documents
            
            Arguments
            -----
            doc_names: list[str]
                List of document names with page number, size, or extension
            page_list: list[str]
                List of pages from documents
                
            Returns
            -----
            page_names: set[str]
                List of pages of specificed doc_names
            """

            page_names = []
            for doc_name in doc_names:
                page_names.extend(
                    [
                        item for item in page_list if regex.findall(f"{regex.escape(doc_name)}\+[0-9]*\.[a-zA-Z0-9]*",item)
                    ]
                )
            
            return set(page_names)
        
        main_minus_exclusion = list(set(main_via_image_list) - set(exclusion))
        
        if subset_by == "document":
            main_minus_exclusion = list(get_doc_names(main_via_image_list) - get_doc_names(exclusion))
            to_remove = list(set(copy.deepcopy(distribution_dict["page_names"])) - find_page_names(main_minus_exclusion, main_via["_via_image_id_list"]))
            _ = [distribution_dict["page_names"].pop(page) for page in to_remove]
            
        elif subset_by == "page":
            ##  remove excluded pages from distribution_dict
            _ = [distribution_dict["page_names"].pop(page) for page in copy.deepcopy(distribution_dict["page_names"]) if page not in main_minus_exclusion]
        
        
        rng = np.random.default_rng(seed)
        pages = []
        total = len(main_minus_exclusion)
        
        ## get pages
        
        if distribution == "random" or subset_by == "document":
            meta_key = "file" if "class" or "page_class" in main_via["_via_attributes"]["file"].keys() else "region"
            N = int(total*fraction) if fraction else number
            
            if N > total:
                logging.warning(f"Size, {N}, selected greater than number of pages available, {total}")
                return []
                       
            if subset_by == "page":
                pages = list(rng.choice(np.array(main_minus_exclusion), size = N, replace = False))
            elif subset_by == "document":
                docs = list(rng.choice(np.array(main_minus_exclusion), size = N, replace = False))
                pages = list(find_page_names(
                    doc_names = docs,
                    page_list = main_via["_via_image_id_list"]
                ))
            
        elif distribution == "token" or distribution == "page" and subset_by == "page":
            accept_key = ["page_class", "class"] if distribution == "page" and "page_class" in main_via["_via_attributes"]["file"].keys() or "class" in main_via["_via_attributes"]["file"].keys() else ["label"]
            meta_key = "file" if accept_key == ["page_class", "class"] else "region"
            
            logging.debug(f"Distributing subset by {meta_key} using {accept_key}")
            
            ##  size = int(N * n/total) for
            ##  N number of pages in subset
            ##  n number of pages with specific number of classes present
            ##  total total number of pages to choose from
            
            N = number if number else int(fraction*total)
            
            if N > total:
                logging.warning("Size selected greater than number of pages available")
                return []
            
            for key in accept_key:
                if key in distribution_dict["keys"][meta_key].keys():
                    ## get list of number of classes for each page
                    tot_page_classes = glom(
                        target = distribution_dict["page_names"],
                        spec = f"*.{meta_key}.{key}.classes_present"
                    )
                    
                    ## numb, like 0,1,2,3,4,... number of classes present in page or page class
                    numbs, ns = zip(*natural_sort((Counter(tot_page_classes)).items()))
                    probs = [n/total for n in ns]
                    
                    sizes = np.floor(N*np.asarray(probs)).astype(int)
                    if sum(sizes) != N:
                        difs = (sizes/N) - probs
                        idx = np.argsort(difs)
                        for i in range(N-sum(sizes)):
                            sizes[idx[0]] += 1
                            
                            difs = sizes/N - probs
                            idx = np.argsort(difs)
                    
                    for numb, size in zip(numbs, sizes):
                        pages_w_n_classes = []
                        
                        ##  add page to list of approved pages if n unique classes present
                        pages_w_n_classes = [main_minus_exclusion[idx] for idx in range(total) if tot_page_classes[idx] == numb]
                        
                        pages = list(
                                set(list(rng.choice(pages_w_n_classes, size = size, replace = False))) | set(pages)
                        )
        
        ## return distribution dictionary
        
        for page in copy.deepcopy(distribution_dict["page_names"]).keys():
            if page not in pages:
                distribution_dict["page_names"].pop(page)
                
        for class_type in copy.deepcopy(distribution_dict["keys"][meta_key]).keys():
            page_names = list(distribution_dict["page_names"].keys())
            distribution_keys = list(distribution_dict["page_names"][page_names[0]][meta_key][class_type].keys())
            
            returned_keys = copy.deepcopy(distribution_keys)
            returned_keys.remove("classes_present")
            returned_keys = ["page_name","classes_present"] + returned_keys
            
            returned_distribution = {dist_key: [] for dist_key in returned_keys}
            distribution_per_page = glom(
                target = distribution_dict["page_names"],
                spec = f"*.{meta_key}.{class_type}"
            )
            
            for i, page_distribution in enumerate(distribution_per_page):
                returned_distribution["page_name"].append(page_names[i])
                for dist_key in distribution_keys:
                    returned_distribution[dist_key].append(page_distribution[dist_key])
            
            if fraction:
                numb_type = "fraction"
            elif number:
                numb_type = "number"
            else:
                numb_type = "random"
            
            csv_name = f"{base_name}_{N}_{seed}_{class_type}_{numb_type}.csv"
            logging.debug(f"Saving csv of distribution to logs/{csv_name}")
            
            df = (pd.DataFrame(returned_distribution)).sort_values(by=["classes_present"], ascending = False)
            
            df.to_csv(f"logs/{csv_name}", index = False)
         
        return pages
    
    global get_distribution
    
    logging.debug("Getting distribution")
    distribution_dict = get_distribution(main_via = main_via)
    
    ##  total seeds to use
    size = int((len(number) if number else 0) + (len(fractions) if fractions else 0) +  (1 if from_name else 0))
    rng = np.random.default_rng(main_seed)
    seed_list = rng.choice(total_subsets, size = size, replace = False)
      
    subsets = {}
    seed_idx = 0
    
    if not number and not fractions and not from_name:
        logging.error("number, fraction, and from_name all None; at least one should be not None")
        raise SystemExit(1)
    
    if from_name:
        to_include = []
        
        subset_name = f"{base_name}_fromname"
        
        for page_name in from_name:
            page_path = pathlib.Path(page_name)
            
            if "+" in page_path.stem:
                if page_path.suffix == "":
                    valid_names = list(filter(
                        regex.compile(f"{regex.escape(page_path.stem)}\..*").match,
                        main_via["_via_image_id_list"]
                        ))
                else:
                    valid_names = list(filter(
                        regex.compile(f"{regex.escape(page_path.stem)}{regex.escape(page_path.suffix)}.*").match,
                        main_via["_via_image_id_list"]
                        ))
                
            else:
                valid_names = list(filter(
                    regex.compile(f"{regex.escape(page_path.stem)}\+[0-9]*{regex.escape(page_path.suffix)}.*").match,
                    main_via["_via_image_id_list"]
                    ))
            
            to_include = natural_sort(list(
                set(natural_sort(to_include)) | set(natural_sort(valid_names))
            ))
                
        to_include = natural_sort(list(
            set(to_include) - set(exclusion)
        ))
                
        subsets[subset_name] = populate_subset(
            main_via = main_via,
            subset_name = subset_name,
            to_include = to_include)
    
    if number:
        for numb_pages in number:
            to_include = choose_rand_pages(
                main_via_image_list = main_via["_via_image_id_list"],
                distribution = distribution,
                distribution_dict = copy.deepcopy(distribution_dict),
                seed = seed_list[seed_idx],
                subset_by = subset_by,
                exclusion = exclusion,
                number = numb_pages,
                base_name = base_name
            )
            
            if to_include:
                subset_name = f"{base_name}_{numb_pages}_{seed_list[seed_idx]}"
                subsets[subset_name] = populate_subset(
                    main_via = main_via,
                    subset_name = subset_name,
                    to_include = to_include
                )
            
            seed_idx += 1
        
    if fractions:
        for fraction in fractions:
            to_include = choose_rand_pages(
                main_via_image_list = main_via["_via_image_id_list"],
                distribution = distribution,
                distribution_dict = copy.deepcopy(distribution_dict),
                seed = seed_list[seed_idx],
                subset_by = subset_by,
                exclusion = exclusion,
                fraction = fraction,
                base_name = base_name
            )
            
            if to_include:
                subset_name = f"{base_name}_frac{int(fraction*100)}_{seed_list[seed_idx]}"
                subsets[subset_name] = populate_subset(
                    main_via = main_via,
                    subset_name = subset_name,
                    to_include = to_include
                )
            
            seed_idx += 1
    
    return subsets


def change(
        main_via: dict[str, Any],
        file_map: Optional[dict[str, str]] = {},
        token_cls_map: Optional[dict[str, str]] = {},
        page_cls_map: Optional[dict[str, str]] = {},
        ext_map: Optional[dict[str, str]] = {}
    ) -> dict[str, Any]:
    """
    Changes file name, token class, page class, or extension in annotation set
    
    Arguments
    -----
    main_via: dict[str, Any]
        VIA annotation set to be changed
    file_map: Optional[dict[str, str]] = {}
        Map of file/page/document name as key to new file/page/document name as value
    token_cls_map: Optional[dict[str, str]] = {}
        Map of token label as key to new token label as value
    page_cls_map: Optional[dict[str, str]] = {}
        Map of page class label as key to new page class label as value
    ext_map: Optional[dict[str, str]] = {}
        Map of one extension as key to another as value; do not include "."
    
    Returns
    -----
    changed_via: dict[str, Any]
        Changed VIA annotation set
        
    Raises
    -----    
    SystemExit
        When change is not possible
    
    """

    changed_via = copy.deepcopy(main_via)
    
    if not file_map and not token_cls_map and not page_cls_map and not ext_map:
        logging.warning("Nothing passed to change")
        return changed_via
    
    if file_map:
        ## get mappings from this file to that file
        for this_file, that_file in file_map.items():
            
            this_file_path = pathlib.Path(this_file)
            that_file_path = pathlib.Path(that_file)
                
            if "+" in this_file_path.stem:
                if this_file_path.suffix == "":
                    these_pages = list(filter(
                        regex.compile(f"{regex.escape(this_file_path.stem)}\..*").match,
                        main_via["_via_image_id_list"]
                        ))
                else:
                    these_pages = list(filter(
                        regex.compile(f"{regex.escape(this_file_path.stem)}{regex.escape(this_file_path.suffix)}.*").match,
                        main_via["_via_image_id_list"]
                        ))
                
            else:
                these_pages = list(filter(
                    regex.compile(f"{regex.escape(this_file_path.stem)}\+[0-9]*{regex.escape(this_file_path.suffix)}.*").match,
                    main_via["_via_image_id_list"]
                    ))
            those_pages = [regex.sub(".+(?=\+[0-9]*)",that_file_path.stem,page) for page in these_pages]
                
            for this_page, that_page in zip(these_pages, those_pages):
                if this_page in main_via["_via_image_id_list"]:
                    changed_via["_via_image_id_list"].remove(this_page)
                    changed_via["_via_image_id_list"].append(that_page)
                    
                    changed_via["_via_img_metadata"].pop(this_page)
                    changed_via["_via_img_metadata"][that_page] = main_via["_via_img_metadata"][this_page]
                    changed_via["_via_img_metadata"][that_page]["filename"] = regex.sub("(?<=\.[a-zA-Z]{3,4})[0-9]*","",that_page)
                else:
                    logging.warning(f'page \"{this_page}\" not found in pages: \n {main_via["_via_image_id_list"]}')
                    raise SystemExit(1)
    
    if token_cls_map:
        for this_token_cls, that_token_cls in token_cls_map.items():
            if this_token_cls in main_via["_via_attributes"]["region"]["label"]["options"].keys():
                changed_via["_via_attributes"]["region"]["label"]["options"].pop(this_token_cls)
                changed_via["_via_attributes"]["region"]["label"]["options"][that_token_cls] = ""
                
                for name, page in main_via["_via_img_metadata"].items():
                    ## get indices of labels to change
                    idxes = []
                    for i, region in enumerate(page["regions"]):
                        if "label" in region["region_attributes"].keys():
                            if region["region_attributes"]["label"] == this_token_cls:
                                changed_via["_via_img_metadata"][name]["regions"][i]["region_attributes"]["label"] = that_token_cls
                    
            else:
                logging.warning(f"label \"{this_token_cls}\" not found in label")
                raise SystemExit(1)
    
    if page_cls_map:
        for this_page_cls, that_page_cls in page_cls_map.items():
            
            if this_page_cls in main_via["_via_attributes"]["file"]["page_class"]["options"].keys():
                changed_via["_via_attributes"]["file"]["page_class"]["options"].pop(this_page_cls)
                changed_via["_via_attributes"]["file"]["page_class"]["options"][that_page_cls] = ""
                
                for name in main_via["_via_image_metadata"]:
                    ## get indices of labels to change
                    if changed_via["_via_img_metadata"][name]["file_attributes"]["page_class"] == this_page_cls:
                        changed_via["_via_img_metadata"][name]["file_attributes"]["page_class"] = that_page_cls

            else:
                logging.warning(f"label \"{this_page_cls}\" not found in label")
                raise SystemExit(1)
    
    if ext_map:
        for this_ext, that_ext in ext_map.items():
            if "." in this_ext:
                this_ext = this_ext.replace(".","")
            if "." in that_ext:
                that_ext = that_ext.replace(".","")
            
            for i, page in enumerate(copy.deepcopy(changed_via["_via_image_id_list"])):
                ## use copy.deepcopy to prevent errors related to rewriting a looping variable
                
                name = pathlib.Path(page).stem
                size = [size for size in regex.findall("(?<=\.[a-zA-Z]{3,4})[0-9]*", pathlib.Path(page).suffix) if size != ""][0]
                ext = pathlib.Path(page).suffix[1:-len(size)]
                
                if ext == this_ext:
                    changed_via["_via_image_id_list"].remove(page)
                    changed_via["_via_image_id_list"].append(f"{name}.{that_ext}{size}")
                    
                    changed_via["_via_img_metadata"][f"{name}.{that_ext}{size}"] = changed_via["_via_img_metadata"].pop(page)
                    changed_via["_via_img_metadata"][f"{name}.{that_ext}{size}"]["filename"] = f"{name}.{that_ext}"
    
    return changed_via


def delete(
        main_via: dict[str, Any],
        attribute: Optional[dict[str, str | list[str]]] = {},
        token_classes: Optional[list[str]] = [],
        page_classes: Optional[list[str]] = [],
        files: Optional[list[str]] = []
    ) -> dict[str, Any]:
    """
    Delete attribute, token class, page class, or files from VIA annotation set
    
    Arguments
    -----
    main_via: dict[str, Any]
        Main VIA annotation set to be altered
    attribute: Optional[dict[str, str | list[str]]] = {}
        Dictionary where key like "region", "file" and value is string or dictionary
    token_classes: Optional[list[str]] = []
        Remove specific token classes and annotations
    page_classes: Optional[list[str]] = []
        Remove specific page class from "page_class" attribute
    files: Optional[list[str]] = []
        List of pages/documents/files to be deleted
         
    Returns
    -----
    changed_via: dict[str, Any]
        Changed via annotation set
        
    Raises
    ------
    SystemExit
        When unableto delete attribute
    """

    changed_via = copy.deepcopy(main_via)
    
    if attribute:
        for key, attribute_to_change_l in attribute.items():
            for attribute_to_change in attribute_to_change_l if isinstance(attribute_to_change_l, list) else [attribute_to_change_l]:
                if key == "region":
                    for name, page in main_via["_via_img_metadata"].items():
                        for idx, region in reversed(list(enumerate(page["regions"]))):
                            region_keys = changed_via["_via_img_metadata"][name]["regions"][idx]["region_attributes"].keys()
                            if attribute_to_change in region_keys:
                                if len(region_keys) != 1:
                                    changed_via["_via_img_metadata"][name]["regions"][idx]["region_attributes"].pop(attribute_to_change)
                                elif len(region_keys) == 1:
                                    changed_via["_via_img_metadata"][name]["regions"].remove(changed_via["_via_img_metadata"][name]["regions"][idx])
                            
                elif key == "file":
                    for name, page in main_via["_via_img_metadata"].items():
                        if attribute_to_change in page["file_attributes"].keys():
                            changed_via["_via_img_metadata"][name]["file_attributes"].pop(attribute_to_change)
                
                elif attribute_to_change in changed_via["_via_attributes"][key].keys():
                    try:
                        changed_via["_via_attributes"][key].pop(attribute_to_change)
                    except:
                        logging.exception(f"Unable to delete {attribute_to_change}")
                        raise SystemExit(1)
    
    if token_classes:
        for token_class in token_classes:
            try:
                changed_via["_via_attributes"]["region"]["label"]["options"].pop(token_class)
            except:
                logging.exception(f"Unable to delete {token_class}")
                continue
            
            for name, page in main_via["_via_img_metadata"].items():
                for idx, region in enumerate(page["regions"]):
                    if "label" in region["region_attributes"].keys():
                        if region["region_attributes"]["label"] == token_class:
                            if len(region["region_attributes"].keys()) != 1:
                                changed_via["_via_img_metadata"][name]["regions"][idx]["region_attributes"].pop("label")
                            else:
                                changed_via["_via_img_metadata"][name]["regions"].remove(region)
    
    if page_classes:
        name_for_class = "page_class"
        
        for page_class in page_classes:
            try:
                changed_via["_via_attributes"]["file"][name_for_class]["options"].pop(page_class)
            except:
                logging.exception(f"Unable to delete {page_class}")
                continue
            
            for name, page in main_via["_via_img_metadata"].items():
                if page["file_attributes"][name_for_class] == page_class:
                    changed_via["_via_img_metadata"][name]["file_attributes"].pop(name_for_class)
    
    if files:
        valid_pages = []
        
        for name in files:
            name_path = pathlib.Path(name)
                
            if "+" in name_path.stem:
                if name_path.suffix == "":
                    valid_names = list(filter(
                        regex.compile(f"{regex.escape(name_path.stem)}\..*").match,
                        main_via["_via_image_id_list"]
                    ))
                else:
                    valid_names = list(filter(
                        regex.compile(f"{regex.escape(name_path.stem)}{regex.escape(name_path.suffix)}.*").match,
                        main_via["_via_image_id_list"]
                    ))
                
            else:
                valid_names = list(filter(
                    regex.compile(f"{regex.escape(name_path.stem)}\+[0-9]*{regex.escape(name_path.suffix)}.*").match,
                    main_via["_via_image_id_list"]
                ))
            
            valid_pages = natural_sort(list(
                set(natural_sort(valid_names)) | set(natural_sort(valid_pages))
            ))
        
        logging.debug(f"Deleting the following files: {valid_pages}")
        for file in valid_pages:
            changed_via["_via_image_id_list"].remove(file)
            changed_via["_via_img_metadata"].pop(file)
    
    return changed_via


def start_surgery(args: Namespace) -> None:
    """
    Prepares and send annotation set(s) to surgery
    
    Arguments
    -----
    args: Namespace
        See arguments()
            
    Returns
    -----
    None
        Saves altered annotation sets
    
    Raises
    -----  
    SystemExit
        When fail to load changing maps or deletion attribute
    
    """

    logging.info(f"Running \"{' '.join(sys.argv[1:])}\" through \"{os.getcwd()}\"")
    
    main_via = load_json(args.via_name)

    if args.surgery == "merge":
        ## call merge
        logging.info("Beginning \"merging\" steps")
        main_via = merge(
            main_via = main_via,
            input_list = args.input
            )
        with open(args.new_name,"w", encoding="utf-8") as f:
            json.dump(main_via, f)
        
        
    elif args.surgery == "subset":
        ## call subset
        logging.info("Beginning \"subsetting\" steps")
        
        if args.exclude:
            exclusion = exclusion_list(
                main_via = main_via,
                file_list = args.exclude
            )
        else:
            exclusion = []
            
        
        subsets = subset(
            main_via = main_via,
            base_name = args.base_name,
            number = args.number,
            fractions = args.fraction,
            distribution = args.distribution,
            exclusion = exclusion,
            from_name = args.from_name,
            main_seed = args.seed
            )
        
        for subset_name, subset_data in subsets.items():
            with open(f"{subset_name}.json", "w", encoding="utf-8") as f:
                json.dump(subset_data, f, sort_keys = True)
        
        
    elif args.surgery == "change":
        logging.info("Beginning \"change\" steps")
        
        ## load stuff
        if args.file_map:
            try:
                logging.debug(f"Loading file mappings {args.file_map}")
                file_map = json.loads(args.file_map)
            except:
                logging.exception("Could not load file map, exiting")
                raise SystemExit(1)
        elif not args.file_map:
            file_map = {}
            
        if args.token_cls:
            try:
                logging.debug(f"Loading token class mappings {args.token_cls}")
                token_cls_map = json.loads(args.token_cls)
            except:
                logging.exception("Could not load token class map, exiting")
                raise SystemExit(1)
        elif not args.token_cls:
            token_cls_map = {}

        if args.page_cls:
            try:
                logging.debug(f"Loading page class mappings {args.page_cls}")
                page_cls_map = json.loads(page_cls_map)
            except:
                logging.exception("Could not load ")
                raise SystemExit(1)
        elif not args.page_cls:
            page_cls_map = {}
            
        if args.ext:
            try:
                logging.debug(f"Loading extension mappings {args.ext}")
                ext_map = json.loads(args.ext)
            except:
                logging.exception("Could not load extension map")
                raise SystemExit(1)
        elif not args.ext:
            ext_map = {}

        
        ## call change
        main_via = change(
            main_via = main_via,
            file_map = file_map,
            token_cls_map = token_cls_map,
            page_cls_map = page_cls_map,
            ext_map = ext_map
        )
        
        change_name = args.new_name if args.new_name else args.via_name
        
        with open(change_name,"w", encoding="utf-8") as f:
            json.dump(main_via, f, sort_keys = True)
        
        
    elif args.surgery == "delete":
        if args.attribute:
            try:
                logging.debug(f"Loading attribute deletion {args.attribute}")
                attribute = json.loads(args.attribute)
            except:
                logging.exception("Could not load attribute deletion map")
                raise SystemExit(1)
        elif not args.attribute:
            attribute = {}
        
        logging.info("Beginning \"deletion\" steps")
        main_via = delete(
            main_via = main_via,
            attribute = attribute,
            token_classes = args.token_cls,
            page_classes = args.page_cls,
            files = args.file
        )
        
        new_name = args.via_name if not args.new_name else args.new_name
        
        with open(new_name,"w", encoding="utf-8") as f:
            json.dump(main_via, f, sort_keys = True)


def arguments() -> ArgumentParser:
    main_parser = ArgumentParser(
        description = ""
    )
    main_parser.add_argument(
        "-n","--via_name",
        type=str, required=False,
        help = "Name of primary VIA file"
    )
    
    
    surgery_parser = main_parser.add_subparsers(
        help = "Different commands for \"via_surgery.py\"",
        dest = "surgery"
    )
    
    
    merge_parser = surgery_parser.add_parser(
        "merge",
        help = "This tool can merge two or more VIA anno sets or CSV VIA labels together"
    )
    merge_parser.add_argument(
        "--new_name",
        type=str, default=None,
        help = "New name for main VIA file with default being overwrite main VIA"
    )
    merge_parser.add_argument(
        "-i","--input",
        nargs="+", type=str, default = None,
        help = "secondary VIA files which will"  
    )
    
    
    subset_parser = surgery_parser.add_parser(
        "subset",
        help = "Create new VIA annotations that are subsets of a \"main\" set"
    )
    subset_parser.add_argument(
        "--base_name",
        type=str, required=True,
        help = "Base name for subsets to be named after"
    )
    subset_parser.add_argument(
        "--from_name",
        nargs="+", type=str,
        help = "Creating subsets by passing name of documents/pages w/ extensions"
    )
    subset_parser.add_argument(
        "--number",
        nargs="+", type=pos_numb,
        help = "Generate subsets with number of files in subset"
    )
    subset_parser.add_argument(
        "--fraction",
        nargs="+", type=float_0_1,
        help = "Generate a subset with "
    )
    subset_parser.add_argument(
        "--exclude",
        nargs="+", type=str, default = None,
        help = "Exclude subset from using doc/page. Autimatically detects format from .json, .txt, .csv"
    )
    subset_parser.add_argument(
        "--distribution",
        type=str, choices=["random","token","page"], default="token",
        help = ""
    )
    
    
    change_parser = surgery_parser.add_parser(
        "change",
        help = "Change-in-place the existing VIA annotation set: change page names, extensions, labels, regions"
    )
    change_parser.add_argument(
        "--new_name",
        type=str, default=None,
        help = "New name for main VIA file with default option being overwrite main VIA"
    )
    change_parser.add_argument(
        "--file_map",
        type=str,
        help = "Mappings of one document name and/or its pages to another name"
    )
    change_parser.add_argument(
        "--token_cls",
        type=str,
        help = "Mappings of one token class to another name"
    )
    change_parser.add_argument(
        "--page_cls",
        type=str,
        help = "Mappings of one page class to another name"
    )
    change_parser.add_argument(
        "--ext",
        type=str, 
        help = "Mappings of one extension to another name, i.e., {\"from_this\":\"to_that\"}" 
    )
    
    
    delete_parser = surgery_parser.add_parser(
        "delete",
        help = "Contains functionality to delete portions of annotation set"
    )
    delete_parser.add_argument(
        "--new_name",
        type=str, default=None,
        help = "New name for main VIA file with default being overwrite main VIA"
    )
    delete_parser.add_argument(
        "--attribute",
        type=str,
        help = "Delete page attribute, e.g., text, class (all), label (all)"
    )
    delete_parser.add_argument(
        "--token_cls",
        nargs="+", type=str,
        help = "Delete selected token classes from all regions"
    )
    delete_parser.add_argument(
        "--page_cls",
        nargs="+", type=str,
        help = "Delete selected page classes from all pages"
    )
    delete_parser.add_argument(
        "--file",
        nargs="+", type=str,
        help = "Delete specific pages and or documents and their associated annotations from VIA file"
    )
    
    
    main_parser.add_argument(
        "-s", "--seed",
        type=int, default=1234, required = False,
        help = "Main seed to generate other seeds for subsets"
    )
    main_parser.add_argument(
        "--log_name",
        type=str, default="log", required=False,
        help = "Name of .log file"
    )
    main_parser.add_argument(
        "-l", "--level",
        type=str,choices=["INFO","DEBUG"], default="INFO", required=False,
        help = "Log level"
    )
    return main_parser


def main(argv=None) -> None:
    parser = arguments()
    args = parser.parse_args(argv)
    setup_env(args = args)

    start_surgery(args = args)
    
    logging.info("Surgery finished")


if __name__ == "__main__":
    main()
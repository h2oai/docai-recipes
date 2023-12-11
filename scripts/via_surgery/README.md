## Requirements
- Python 3.10+
    - pandas
    - numpy
    - regex
    - glom

## Merging
```bash
python -m via_surgery -n "main_via.json" merge --new_name "merged_via.json" -i "via_to_merge_1.json" "via_to_merge_2.json"
```

## Subsetting
```bash
python -m via_surgery -n "main_via.json" subset --base_name "dataset" --number 10 20 20 500 --fraction 0.25 0.77 0.01 --exclude "this_file.jpg" "docs_named_this" "this_specific_page+1.jpg" --distribution "token"
```

> Does not support page subsetting of PDFs

## Changing
```bash
python -m via_surgery -n "main_via.json" change --new_name "changed_via.json" --file_map '{\"this-name\":\"that_name\"}' --token_cls '{\"this_class\":\"that_class\"}' --page_cls '{\"this_class\":\"that_class\", \"another_class\":\"different_class\"}' --ext '{\"this_ext\":\"that_ext\"}'
```
May need to try using double single-quotes around the regular expression list. E.g. `-token_cls ''{\"this_class\":\"that_class\"}''`

## Deleting
```bash
python -m via_surgery -n "main_via.json" delete --new_name "deletion_via.json" --token_cls "this token" "that_token" "incorrect-token" --page_cls "that_page" "this_page" --file "foo.jpg" "bar+0.png"
```

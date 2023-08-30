The post-processing scripts in this repository are organized into folders based on the Document AI version they are
compatible with.

Short descriptions of the scripts in each folder are provided below.

- `post_processor_1.py`: Merges tokens and creates a flat JSON output.
- `post_processor_2.py`: Merges tokens, extracts line items from a supply-chain document.
- `post_processor_3.py`: Modified version of post_processor_2.py with template method for entity extraction.
- `post_processor_4.py`: Outputs all document information in flat JSON, convertible to CSV.
- `post_processor_5.py`: Extracts image snippets of entities, provides top-n answers per class.
- `post_processor_6.py`: Offers empty answers for unidentified model classes in supply-chain documents.
- `post_processor_7.py`: Offers empty answers for unidentified model classes in general documents.
- `post_processor_8.py`: Accepts template dicts for post-processing.
- `post_processor_9.py`: Returns redacted image.
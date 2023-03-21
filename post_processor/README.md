The post-processing scripts in this repository are organized into two folders, v1 and v2.

The v1 folder consists of scripts that are compatible with Document AI v0.5.0 and prior versions. These scripts will still work with Document AI v0.6.0 and newer versions. However, after Document AI v1.0.0, the v1 scripts will no longer be supported.

The v2 folder, on the other hand, contains scripts that are compatible with Document AI v0.6.0 and later versions. These scripts are recommended as they offer more features and flexibility in the post-process function compared to the v1 scripts.

Short descriptions of the scripts in each folder are provided below.
- `post_processor_1.py`: This script is a simple example of a post-processing script that can be used to merge tokens and output a flat JSON file.
- `post_processor_2.py`: This script is a simple example of a post-processing script that can be used to merge tokens and extract line items from a supply-chain document.
- `post_processor_3.py`: This script is an example of modified post-processor_2.py that also uses template method to extract entities from a supply-chain document.
- `post_processor_4.py`: This script is an example that outputs all available information from a document in a flat JSON format. The output can be easily converted to a CSV file.
- `post_processor_5.py`: This script is a more complex example that can be used to extract image snippets of the entities and also provides top-n answers for each class.
- `post_processor_6.py`: This script is an example that also provides empty answers for those classes that are not present in a supply-chain document.
- `post_processor_7.py`: This script is an example that also provides empty answers for those classes that are not present in a general document.
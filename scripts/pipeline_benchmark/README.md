<!--  Use with consent of H2O.ai  -->
<!--        April 12, 2023        -->

# **Scoring Pipeline Testing**
---
---
# **Table of Contents**

- [Table of Contents](#table-of-contents)
- [How to use `pipeline.py`](#how-to-use-pipelinepy)
    - [Introduction](#introduction)
    - [Setting up your environment](#setting-up-your-environment)
    - [Authentication through ...](#authentication-through)
        - [`config.yaml`](#configyaml-recommended)
        - [`h2o`](#h2o)
        - [`curl`](#curl)
    - [Parser commands](#parser-commands)
    - [Example workflows](#example-workflows)
    - [Known issues](#known-issues)


---
---

# **How to use `pipeline.py`**
## **Introduction**

`pipeline.py` is a tool to rapidly deploy custom scoring pipelines to a HAIC Document AI environment and test their speed. This tool should be used to help rigorously test the scoring pipeline process and the various options one can select when deploying a scoring pipeline - Kubernetes parameters, OCR method, post-processors, page/token classification model, number of replicas, number of requests, and dataset to test againt (PDFs, JPGs, PNGs, large images, etc.).

The scoring pipelines are built programatically from a CSV "recipe" file which must contain many of the parameters stated previously. The models used to build the scoring pipelines can be selected from any project, but at least one of either a page or token model must exist to use this tool. The datasets which are used to test the different possible arrangments of scoring pipeline parameters and models are stored locally where the `pipline.py` script is ran from. 

This tool tests all combinations of created pipelines and selected datasets, i.e., every dataset will be tested against every pipeline selected and or created.

In order to build these pipelines, however, special access is required. How to approve of this access is discussed in [Authentication through ...](#authentication-through), though using the `config.yaml` method is the recommended path.

## **Setting up your environment**

Download and install the `docai-scorer:0.2.4` Docker image from [here](https://h2oai.slack.com/archives/C0201BB8W9Z/p1678408753571289) using the following:

```
docker load < [name-of-image].tar.gz
```

To get your `python` client ready (through `conda` or similar) ensure the following:

- Python 3.8+
- install the necessary packages
    - argparse
    - os
    - requests
    - subprocess
    - json
    - PyYaml
    - pandas
    - numpy
    - time
    - tqdm
    - regex
    - PyPDF2
    - h2o-authn

Download and extract `scorer_datasets.zip` to the directory that contains the `pipeline.py` file so that the directory looks like the following:

```
| your_directory
    | scorer_datasets
        | dataset_1
            | file 1
            | file 2
            | ...
        | dataset_2
            | ...
        | ...
    | pipeline.py
    | config.yaml
```

Next, make a `config.yaml` file that contains the following (copy and pasting is recommended):
```
auth_base_url: null
benchmark: false
benchmark_results: null
docai_password: null
docai_user: null
dry_run: false
images: null
keycloak_client_id: null
keycloak_realm: null
list_images: false
log_level: DEBUG
name: null
num_replicas: null
num_requests: null
out_dir: null
pipeline: null

platform_client_id: null
platform_token: none
token_endpoint_url: null

scorer_base_url: null
scorer_logs: false
temp_image_dir: null
valid_image_file_extensions:
- .pdf
- .jpeg
- .jpg
- .png
- .bmp
- .tiff
- .gif
verbose: false
version: 0.2.4
```

## **Authentication through ...**
### **`config.yaml` (recommended)**

This method uses a YAML file that contains the necessary components to use ML-API using the `python` library `h2o_authn`. 

The `config.yaml` file must contain the following:
```
platform_client_id: <client-id>
platform_token: <token>
token_endpoint_url: https://<OIDC_URL>/protocol/openid-connect/token
```

The `platform_client_id` and `token_endpoint_url` (OIDC URL) can be found in the CLI and API access section of your HAIC environment. The `platform_token` can sometimes be found in CLI and API access, but, more likely, you must use either the [h2o](#h2o) or [curl](#curl) methods to get the necessary `platform_token` and copy it into the `config.yaml` file.

### **`h2o`**

Download the latest `h2o` package from [here](https://h2oai-cloud-release.s3.amazonaws.com/releases/ai/h2o/h2o-cloud/latest/index.html).

1. Place 'h2o' into `.local/bin` folder
2. Run
```
chmod +x .local/bin/h2o
h2o config setup
```
4. Input "Endpoint URL", "OIDC URL", and "Client ID"
5. Run
```
h2o config update-platform-token
h2o platform access-token
```
A long string should be displayed. If you are using this to get the `platform_token` for the [config.yaml](#configyaml-recommended) method, copy and paste the result into the `config.yaml` file.

### **`curl`**

Copy and paste the following `curl` command to recieve the `platform_token`. Update the command to include the appropriate `AUTHORIZATION_URL`, `PASSWORD`, `USERNAME`, and `CLIENT_ID`.
```
curl -X POST "<AUTHORIZATION_URL>" -H "Content-Type:application/x-www-form-urlencoded" -d "password=<PASSWORD>" -d "username=<USERNAME" -d "grant_type=password" -d "response_type=token" -d "client_id=<CLIENT_ID>"
```
Copy and paste the `access_token` field into the `config.yaml` file.

How to use this authentication method in the `pipeline.py` script will be covered in the following section, [Example workflows](#example-workflows).

## **Parser commands**
The following table provides a comprehensive list of parsing commands, an example, and their description.

Table 1: Parser commands
---
| Command | Example| Description |
| :-- | :--- |--- |
|`--pipeline_recipes`| pipeline_recipes.csv |CSV file containing necessary ingredients for creating pipelines|
|`--scorer_url`|"https://document-ai-scorer.cloud-qa.h2o.ai"|Scorer URL for HAIC Document AI environment|
|`--replicas`|8|Integer number of replicas to be created|
|`--requests`|8|Integer number of requests to keep posted at one time while scoring|
|`--datasets`|5_pdf dataset_1 dataset_2|Space delimited list of datasets found in `--image_supdir` folder|
|`--pipeline_list`|pipeline_list.csv|CSV that contains existing pipeline names to be tested or name of CSV file that will be created to store names of newly created pipelines|
|`--folder_output`|output|Name of folder where output files will be stored|
|`--results`|results.csv|CSV file that test results will be stored in|
|`{h2o,sso,curl}`|curl|Which authentication method to use. `h2o` requires the `h2o` module to be setup, `curl` requires more argument parsing commands (see below), and `sso` requries the `config.yaml` file|
|`--auth_url`|"https://keycloak.docai.h2o.ai/auth/realms/wave/protocol/openid-connect/token"|[`curl`] Base authorization URL|
|`--auth_realm`|"realm-name"|[`curl`] Realm for authorization|
|`--client_id`|"client-id"|[`curl`] Client ID for authorization|
|`--auth_pass`|password|[`curl`] User password for authorization|
|`--auth_user`|username|[`curl`] Username for authorization|
|`--image_supdir`|scorer_datasets|[`OPTIONAL`] Folder which `--datasets` reside in|
|`--log_level`|"DEBUG"|[`OPTIONAL`] Logging level. Either "DEBUG" (default) or "INFO"|
|`--docker_v`|"0.2.4"|[`OPTIONAL`] String containing version of `docai-scorer` image used|

> [ ! ] WARNING: `{h2o, sso, curl}` and their accompanying subcommands must come at end of command line arguments. 
>
> [ ! ] WARNING: `{h2o, sso, curl}` are mutually exclusive authentication methods; only choose one.
## **Example workflows**

Before pipelines can be created, the ingredients must be gathered in the recipes CSV files. The necessary ingredient categories can be found in Table 2 (below). But the required components that need to be filled out, i.e., not empty, are the following:

- `pipeline_name`
- One set from the following 
    - `project_token_name` and `model_token_name`
    - `project_page_name` and `model_page_name`
    - both

The `pipeline_name` field must be all lowercase and no "`[]_./()'" `" characters (just numbers, letters, and hyphens).

The `project_<>_name` and `model_<>_name` refers to the plain text names of the project and associated model for either `page` or `token` classification.

The available `ocr_method` options are the following:
```
"best_text_extract", "doctr_ocr", "paddle_ocr_arabic", "paddle_ocr_latin", "doctr_efficient_netb3", "doctr_efficient_netb0", "doctr_efficient_netv2m", "e3_best_text_extract", "pdf_text_extract", "tesseract_ocr"
```

The available `post_processor` options are the following:
```
"generic","supply-chain"
```

Table 2: Example `pipeline_recipes.csv` file
---
|pipeline_name |project_token_name|model_token_name|project_page_name         |model_page_name         |ocr_method            |post_processor|min_replicas|max_replicas|requests_cpu|requests_memory|limits_cpu|limits_memory|
|--------------|------------------|----------------|--------------------------|------------------------|----------------------|--------------|------------|------------|------------|---------------|----------|-------------|
|cba-best      |cba_test          |modelextract    |                          |                        |best_text_extract     |generic       |4           |8           |500         |8              |4         |8            |
|cba-e3best    |cba_test          |modelextract    |                          |                        |e3_best_text_extract  |generic       |4           |8           |500         |8              |4         |8            |
|cba-doctr     |cba_test          |modelextract    |                          |                        |doctr_ocr             |generic       |4           |8           |500         |8              |4         |8            |
|cba-effnetb3  |cba_test          |modelextract    |                          |                        |doctr_efficient_netb3 |generic       |4           |8           |500         |8              |4         |8            |
|cba-effnetb0  |cba_test          |modelextract    |                          |                        |doctr_efficient_netb0 |generic       |4           |8           |500         |8              |4         |8            |
|cba-effnetv2m |cba_test          |modelextract    |                          |                        |doctr_efficient_netv2m|generic       |4           |8           |500         |8              |4         |8            |
|cba-pplatin   |cba_test          |modelextract    |                          |                        |paddle_ocr_latin      |generic       |4           |8           |500         |8              |4         |8            |
|cba-pparabic  |cba_test          |modelextract    |                          |                        |paddle_ocr_arabic     |generic       |4           |8           |500         |8              |4         |8            |
|cba-tesseract |cba_test          |modelextract    |                          |                        |tesseract_ocr         |generic       |4           |8           |500         |8              |4         |8            |
|id-best       |ML Test           |id-card-october |                          |                        |best_text_extract     |generic       |4           |8           |500         |8              |4         |8            |
|id-page-best  |ML Test           |id-card-october |Testing Universal Pipeline|page class full training|best_text_extract     |generic       |4           |8           |500         |8              |4         |8            |
|id-doctr      |ML Test           |id-card-october |                          |                        |doctr_ocr             |generic       |4           |8           |500         |8              |4         |8            |
|uni-page-best |                  |                |Testing Universal Pipeline|page class full training|best_text_extract     |generic       |4           |8           |500         |8              |4         |8            |
|uni-page-doctr|                  |                |Testing Universal Pipeline|page class full training|doctr_ocr             |generic       |4           |8           |500         |8              |4         |8            |
---


Example Command (`config.yaml`):
```
python -m pipeline --pipeline_recipes pipeline_recipes.csv --scorer_url "https://document-ai-scorer.cloud-qa.h2o.ai" --replicas 8 --requests 8 --image_supdir scorer_datasets --datasets 5_pdf cba12_pdf cba12_png sroie_100_pdf docbank_100_jpg --folder_output output --results results.csv --pipeline_list pipeline_list.csv sso
```

Example Command (`h2o`):
```
python -m pipeline --pipeline_recipes pipeline_recipes.csv --scorer_url "https://document-ai-scorer.cloud-qa.h2o.ai" --replicas 8 --requests 8 --image_supdir scorer_datasets --datasets 5_pdf cba12_pdf cba12_png sroie_100_pdf docbank_100_jpg --folder_output output --results results.csv --pipeline_list pipeline_list.csv h2o
```

Example Command (`curl`):
```
python -m pipeline --pipeline_recipes pipeline_recipes.csv --scorer_url "https://document-ai-scorer.cloud-qa.h2o.ai" --replicas 8 --requests 8 --image_supdir scorer_datasets --datasets 5_pdf cba12_pdf cba12_png sroie_100_pdf docbank_100_jpg --folder_output output --results results.csv --pipeline_list pipeline_list.csv curl --auth_url --
```
## **Known issues**

> **Failed `grep` times**:
>
> While the `docai-scorer` is running from the Docker image, if you print a newline in the terminal, the script will fail.

> **Scorer timeout**:
>
> If a job takes more than 3600 seconds (1 hour), the job will fail but the program will progress.

> **Failed jobs**:
>
> If a job fails, e.g., a 4XX error or 5XX error, the program will not progress and must be terminated.

> **OS**:
> 
> This script has only been tested on Windows Subsystem for Linux (WSL) on Ubuntu 20.04. This code should work in most OSs (Windows, Ubuntu, MacOS, and virtual environments).
---
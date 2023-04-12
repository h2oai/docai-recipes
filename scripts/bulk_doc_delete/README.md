<!--  Use with consent of H2O.ai  -->
<!--        April 12, 2023        -->

# **Deleting Old Document Sets**

---
---
# **Table of Contents**

- [Table of Contents](#table-of-contents)
- [How to use `bulk_doc_delete.py`](#how-to-use-bulk_doc_deletepy)
    - [Introduction](#introduction)
    - [Setting up your environment](#setting-up-your-environment)
    - [Authentication through ...](#authentication-through)
        - [`config.yaml`](#configyaml-recommended)
        - [`h2o`](#h2o)
        - [`curl`](#curl)
    - [Parser commands](#parser-commands)
    - [Example workflows](#example-workflows)
    - [Known Issues](#known-issues)
---
---
#  **How to use `bulk_doc_delete.py`**
## **Introduction**
The ability to purge sensitive documents is necessary to safeguard customer privacy and prevent unnecessary risk. `bulk_doc_delete.py` was created to facilitate the purging of document sets and their associated dependencies, e.g., , (simply document sets from now on) from H2O Document AI (`docai`) environments.

This lightweight CLI `python` script can be used to delete document sets that are older than or younger than a certain age (in days); thus, to accommodate constraints pertaining to retention of customer data, document sets greater than a certain age can be removed programatically.

> [ ! ] WARNING: this tool has it's deleted function enabled. Care and proper use must be ensured so that the desired document sets are deleted - deletion is irreversible.

## **Setting up your environment**

This tool requires the following `python` packages:

```
- Python [3.8+]
    - os
    - argparse
    - requests
    - datetime
    - json
    - PyYaml
    - numpy
    - subprocess
    - h2o-authn
```

> [ % ] NOTE: there are no specific requirements for the versions of the `python` libraries; just ensure that they are compatible with each other.

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

|Command|Example|Description|
|:---|----|----|
|`--mlapi_url`|"https://ml-api.cloud-qa.h2o.ai/v1alpha"|The MLAPI URL found by replacing beginning portion of `docai` environment URL with `ml-api` and appending `v1alpha` at end.|
|`--delete_older`|30|Deletes document sets older than a certain age in days.|
|`--delete_younger`|10|Deletes document sets younger than a certain age in days.|
|`--filtering`|projects_to_delete.txt|[`OPTIONAL`] Text file containing projects that this script will document sets from with each project display name positioned on separate lines.|
|`--preview`|True|[`OPTIONAL`] Displays Document Hierarchy of document sets, their parent projects, and their associated dependencies in JSON format. If enabled, no document sets will be deleted.|
|`--save_preview`|preview_name.json|[`OPTIONAL`] Saves Document Hierarchy to specified JSON file.|
|`{h2o,sso,curl}`|sso|Authentication method used to access `ml-api`. Mutually exclusive options; only choose one and must be called at the end of the command chain.|
|`--config`|template_config.yaml|[`sso`] Name of YAML file containing necessary `sso` login information.|
|`--auth_url`|"https://authorization.url/auth/realms/realm/protocol/openid-connect/token"|[`curl`] Authorization URL used to obtain the `access token` for `ml-api`.|
|`--password`|password|[`curl`] Password for `docai` account.|
|`--username`|username|[`curl`] Username of `docai` account.|
|`--client_id`|haic-client-id|[`curl`] HAIC client ID.|

## **Example Workflows**
```
python -m bulk_doc_delete --mlapi_url "https://ml-api.appstore-install.h2o.dev/v1alpha" --filtering projects.txt --save_preview output.json --delete_older 60 curl --auth_url "https://keycloak.appstore-install.h2o.dev/auth/realms/appstore/protocol/openid-connect/token" --password password --username username --client_id client
```

```
python -m bulk_doc_delete --mlapi_url "https://ml-api.appstore-install.h2o.ai/v1alpha" --delete_older 30 --preview True sso --config config.yaml
```

```
python -m bulk_doc_delete --mlapi_url "https://ml-api.appstore-install.h2o.ai/v1alpha" --delete_younger 5 --preview True --save_preview preview.json h2o
```

> [ ! ] WARNING: when `--preview True` is not included, document sets *will* be permanantly deleted. When in doubt, set `--preview True`.

## **Known issues**

> **Deleting failed sets:**
> 
> This tool will not delete document sets or annotation sets that have failed.

> **Failed projects:**
> 
> If a project is only comprised of failed document and annotation sets, the program will fail. These document sets must be deleted manually.

---
<!--  Use with consent of H2O.ai  -->
<!--        April 12, 2023        -->

# **Deleting Pipelines**

---
---
# **Table of Contents**

- [Table of Contents](#table-of-contents)
- [How to use `pipeline_delete.py`](#how-to-use-pipeline_deletepy)
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
#  **How to use `pipeline_delete.py`**
## **Introduction**
As of H2O.ai's Document AI (`docai`) version `0.6`, there doesn't exist a frontend method for deleting pipelines or all pipelines in a given project or group of projects. Therefore, this tool should be used to delete pipelines from a `docai` environment.

> [ ! ] WARNING: The delete funcationality is enabled, so pipelines *will* be deleted if either the project they reside in or name is given to the tool.

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
    - subprocess
    - regex
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
|`--scorer_url`|"https://document-ai-scorer.name.h2o.ai"|URL to `docai` scorer.|
|`--projects`|"Project 1" "Project-2" "pRoJecT_3"|Space delimited list with each project surrounded by quotation marks ("").|
|`--pipelines`|"Pipeline_1" "pipeline-2" "pipeline-10"|Space delimited list with each pipeline name surrounded by quoatation marks ("").|
|`--preview`|True|[`OPTIONAL`] Displays which pipeline will be deleted if `--preview True` is disabled|
|`{h2o,sso,curl}`|sso|Authentication method used to access `ml-api`. Mutually exclusive options; only choose one and must be called at the end of the command chain.|
|`--auth_url`|"https://authorization.url/auth/realms/realm/protocol/openid-connect/token"|[`curl`] Authorization URL used to obtain the `access token` for `ml-api`.|
|`--password`|password|[`curl`] Password for `docai` account.|
|`--username`|username|[`curl`] Username of `docai` account.|
|`--client_id`|haic-client-id|[`curl`] HAIC client ID.|

## **Example Workflows**
```
python -m pipeline_delete --scorer_url "https://document-ai-scorer.name.h2o.ai" --projects "Project 1" "Test project" --pipelines "1st pipeline" "another pipeline" curl --auth_url "https://keycloak.appstore-install.h2o.dev/auth/realms/appstore/protocol/openid-connect/token" --password password --username username --client_id client
```

```
python -m pipeline_delete --scorer_url "https://document-ai-scorer.name.h2o.ai" --pipelines "1st pipeline" "another pipeline" sso
```

```
python -m pipeline_delete --scorer_url "https://document-ai-scorer.name.h2o.ai" --projects "Project 1" "Test project" --preview True h2o
```

> [ ! ] WARNING: when `--preview True` is not included, pipelines *will* be permanantly deleted. When in doubt, set `--preview True`.

## **Known issues**

> **Using project names:**
> 
> This tool uses a regular expression to transform the provided `scorer_url` into a MLAPI URL which is used to faciliate plain-name usage of projects. If you get an error reading `|  FAILURE  |  Could not get list of projects`, notify the maker of this script for help.

---
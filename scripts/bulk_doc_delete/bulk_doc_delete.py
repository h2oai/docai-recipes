###    Use with consent of H2O.ai, INC.    ###
###              July 20, 2023             ###

# Update the script to continue deleting other documents even if one document fails to delete. @chuan.ning1 23/06/23

import os
import requests
import datetime
import json
import yaml
import numpy as np
import subprocess
import h2o_authn

from subprocess import Popen, PIPE
from requests import get, post, delete
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime, timezone, timedelta
from h2o_authn import TokenProvider


def arguments():
    parser = ArgumentParser(description="Delete old documents and annotation sets from Document AI (won't delete failed document sets) \n\n\
EXAMPLE COMMANDS: \n\
    python -m bulk_doc_delete --mlapi_url \"https://ml-api.appstore-install.h2o.dev/v1alpha\" --filtering projects.txt --save_preview output.json --delete_older 60 curl --auth_url \"https://keycloak.appstore-install.h2o.dev/auth/realms/appstore/protocol/openid-connect/token\" --password password --username username --client_id client\n\
    \n\
    python -m bulk_doc_delete --mlapi_url \"https://ml-api.appstore-install.h2o.ai/v1alpha\" --delete_older 30 --preview True sso --config config.yaml\n\
    \n\
    python -m bulk_doc_delete --mlapi_url \"https://ml-api.appstore-install.h2o.ai/v1alpha\" --delete_younger 30 h2o",
                            formatter_class=RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help='Authorization option', dest='auth')

    parser.add_argument(
        "--mlapi_url", required=True, type=str,
        help="URL for MLAPI actions.\n\
Take normal Document AI URL and repalce first portion with \"ml-api\" and append \"/v1alpha\"\n\
    e.g., --mlapi_url \"https://ml-api.<>.h2o.<>/v1alpha\"\n\
WARNING: Must enclose with \"\" to ensure proper script execution"
    )

    parser.add_argument(
        "--delete_older", required=False, type=float, default=None,
        help="Delete document and annotation sets older than X days\n\
Mutually exclusive with --delete_younger"
    )
    parser.add_argument(
        "--delete_younger", required=False, type=float, default=None,
        help="Delete document and annotation sets younger than X days\n\
Mutually exclusive with --delete_older"
    )
    parser.add_argument(
        "--filtering", required=False, type=str, default="All projects",
        help="Name of text file in the same location as this .py script. \n\
Contains names of projects which this program will scan documents for deletion.\n\
Project names on individual lines\n\
     e.g., --filtering project_filter.txt"
    )
    parser.add_argument(
        "--preview", required=False, type=bool, default=False,
        help="Boolean option (True or False) with False as default\n\
Outputs formatted json with format \n\
    {\n\
      {\n\
        \"project_name_1\": {\n\
          \"document_set_1\":{\n\
            \"annotationSets\":[List, of, annotation, sets and their statuses],\n\
            \"age:\" \"age in days\",\n\
            \"createTime\": \"UTC time of creation\"\n\
          },\n\
          \"document_set_2\":{\n\
            ...\n\
            },\n\
          ...\n\
        },\n\
        \"project_name_2\": {\n\
          \"document_set_1\":{\n\
            \"annotationSets\":[List, of, annotation, sets and their statuses],\n\
            \"age:\" \"age in days\",\n\
            \"createTime\": \"UTC time of creation\"\n\
          },\n\
          \"document_set_2\":{\n\
            ...\n\
            },\n\
            ...\n\
        ...\n\
        }\n\
      }\n\
    }")
    parser.add_argument(
        "--save_preview", required=False, type=str, default=None,
        help="Name of json file to which the Hierarchy of Deleted Documents will be saved to.\n\
    e.g., --save_preview output.json"
    )

    parser_sso = subparsers.add_parser('sso', help='Authentication using the \"h2o_authn\" library')
    parser_h2o = subparsers.add_parser('h2o',
                                       help='Authentication using the h2o module (requries h2o to be setup prior to use)')
    parser_curl = subparsers.add_parser('curl', help='Authentication using curl to website')

    parser_sso.add_argument(
        "--config", required=True, type=str, default='config.yaml',
        help="YAML file containing \"platform_client_id\", \"platform_token\", and \"token_endpoint_url\" for authorization."
    )

    parser_curl.add_argument(
        "--auth_url", required=True, type=str,
        help="Authentication URL for MLAPI authenticaion.\n\
    e.g., --auth_url \"https://<>.<>.h2o.<>/auth/realms/<>/protocol/openid-connect/token\"\n\
WARNING: Must enclose with \"\" to ensure proper script execution"
    )
    parser_curl.add_argument(
        "--password", required=True, type=str,
        help="Password for authentication"
    )
    parser_curl.add_argument(
        "--username", required=True, type=str,
        help="Username for authentication"
    )
    parser_curl.add_argument(
        "--client_id", required=True, type=str,
        help="Client id for authentication"
    )
    return parser


def main(argv=None):
    parser = arguments()
    args = parser.parse_args(argv)

    if args.delete_younger == None and args.delete_older == None:
        raise SystemExit(
            "|  EXITING   |  Please include either \"--delete_younger\" or \"--delete_older\" in command, please.")
    if args.delete_younger != None and args.delete_older != None:
        raise SystemExit(
            "|  EXITING   |  Illegal inclusion of both \"--delete_younger\" and \"--delete_older\" parameters. Remove one, please.")

    if '/v1alpha' not in args.mlapi_url:
        raise SystemExit(f"|  EXITING   |  You must include \"/v1alpha\" at the end of \"{args.mlapi_url}\", please.")

    if 'ml-api' not in args.mlapi_url:
        raise SystemExit(f"|  EXITING   |  You must have \"ml-api\" at the beginning of \"{args.mlapi_url}\", please.")

    ACCESS_TOKEN = get_access_token(args=args)

    delete_document_sets(args=args, ACCESS_TOKEN=ACCESS_TOKEN)


def get_access_token(args):
    if args.auth == 'curl':
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "password": args.password,
            "username": args.username,
            "grant_type": "password",
            "response_type": "token",
            "client_id": args.client_id
        }

        try:
            access_post = post(url=args.auth_url, headers=headers, data=data)
            access_post.raise_for_status()
            print('|  SUCCESS   |  Received ACCESS_TOKEN through curl')
            return access_post.json()["access_token"]
        except requests.exceptions.HTTPError as err:
            raise SystemExit(f'|  EXITING   |  ACCESS_POST CODE {access_post.status_code}: {err}')

    elif args.auth == 'h2o':
        access_output, access_error = Popen(['h2o', 'platform', 'access-token'],
                                            stdout=PIPE, stderr=PIPE, universal_newlines=True).communicate()
        if access_error != '':
            raise SystemExit(f'|  EXITING   |  ACCESS_POST: {access_error}')
        else:
            print('|  SUCCESS   |  Received ACCESS_TOKEN through h2o app')
            return access_output.strip()

    elif args.auth == 'sso':
        config = {}

        if os.path.exists(args.config) == True:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            template = {
                'platform_client_id': None,
                'platform_token': None,
                'token_endpoint_url': None
            }

            with open('template_sso.yaml', 'w') as f:
                yaml.dump(template, f)

            raise SystemExit(f'|  EXITING   |  Could not find YAML config file {args.config}; making template.')

        token_provider = TokenProvider(
            refresh_token=config['platform_token'],
            client_id=config['platform_client_id'],
            token_endpoint_url=config['token_endpoint_url'],
            expiry_threshold=timedelta(seconds=120)
        )

        try:
            print('|  SUCCESS   |  Received ACCESS_TOKEN through \"h2o_authn\" library')
            return token_provider.token()
        except Exception as err:
            raise SystemExit(f'|  EXITING   |  SSO_TOKEN: {err}')


def delete_document_sets(args, ACCESS_TOKEN):
    if args.filtering != "All projects":
        with open(args.filtering, 'r') as f:
            projects_to_delete_from = f.read().splitlines()
    else:
        projects_to_delete_from = []

    default_headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    try:
        project_get = get(url=f'{args.mlapi_url}/projects', headers=default_headers)
        project_get.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(f'|  EXITING   |  PROJECT_GET CODE {project_get.status_code}: {err}')

    document_hierarchy = {}
    print('|  SUCCESS   |  Looking through approved projects')
    for project_dict in project_get.json()["projects"]:
        if project_dict["displayName"] in projects_to_delete_from or args.filtering == "All projects":
            project_uuid = project_dict["name"][9:]

            try:
                document_get = get(url=f"{args.mlapi_url}/projects/{project_uuid}/documentAI/documentSets:search",
                                   headers=default_headers)
                document_get.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise SystemExit(f'|  EXITING   |  DOCUMENT_GET CODE {document_get.status_code}: {err}')

            for document_dict in document_get.json()["documentSets"]:

                date_format = r'%Y-%m-%dT%H:%M:%S.%fZ'
                document_time = datetime.strptime(document_dict["createTime"], date_format).replace(tzinfo=timezone.utc)
                now_time = datetime.utcnow().replace(tzinfo=timezone.utc)

                document_age = (now_time - document_time).total_seconds() / 3600 / 24

                if args.delete_younger == None and document_age > args.delete_older:
                    try:
                        dependencies_get = get(url=f"{args.mlapi_url}/{document_dict['name']}:getDependencies",
                                               headers=default_headers)
                        dependencies_get.raise_for_status()
                    except requests.exceptions.HTTPError as err:
                        raise SystemExit(f'|  EXITING   |  DEPENDENCIES_GET CODE {dependencies_get.status_code}: {err}')

                    try:
                        annotation_get = get(
                            url=f'{args.mlapi_url}/projects/{project_uuid}/documentAI/annotationSets:search',
                            headers=default_headers)
                    except requests.exceptions.HTTPError as err:
                        raise SystemExit(f'|  EXITING   |  ANNOTATION_GET CODE {annotation_get.status_code}: {err}')

                    if project_dict["displayName"] not in document_hierarchy:
                        document_hierarchy[project_dict["displayName"]] = {}

                    document_hierarchy[project_dict["displayName"]][document_dict["displayName"]] = {
                        "annotationSets": [{annotation_set["displayName"]: ""} for annotation_set in
                                           dependencies_get.json()["dependencies"] if
                                           annotation_set["artifactName"][-1] == "1"],
                        "age": f"{np.round(document_age, 3)}",
                        "createTime": f"{document_time.date()} {document_time.time()} UTC"
                    }

                    for annotation_dict in annotation_get.json()['annotationSets']:
                        for i, anno_dict in enumerate(
                                document_hierarchy[project_dict["displayName"]][document_dict["displayName"]][
                                    "annotationSets"]):
                            if list(anno_dict.keys())[0] == annotation_dict["displayName"]:

                                document_hierarchy[project_dict["displayName"]][document_dict["displayName"]][
                                    "annotationSets"][i][annotation_dict["displayName"]] = annotation_dict["status"]

                                if annotation_dict["status"] != "ARTIFACT_STATUS_AVAILABLE":
                                    print(
                                        f"|  WARNING   |  \"{annotation_dict['displayName']}\" reports it\'s status as \"{annotation_dict['status']}\"")
                                break
                    if args.preview == False:
                        print(
                            f'|  DELETING  |  Deleting "{document_dict["displayName"]}" from "{project_dict["displayName"]}" with age {np.round(document_age, 2)} days')
                        document_delete = delete(url=f'{args.mlapi_url}/{document_dict["name"]}:bulkDelete',
                                                     headers=default_headers)
                        if(not document_delete.ok):
                            print(f'|  DELETE FAILED {document_delete.status_code} |  {args.mlapi_url}/{document_dict["name"]}:bulkDelete')
                        else:
                            print(f'|  DELETE SUCCESS {document_delete.status_code} |  {args.mlapi_url}/{document_dict["name"]}:bulkDelete')

                elif args.delete_younger != None and document_age < args.delete_younger:
                    try:
                        dependencies_get = get(url=f"{args.mlapi_url}/{document_dict['name']}:getDependencies",
                                               headers=default_headers)
                        dependencies_get.raise_for_status()
                    except requests.exceptions.HTTPError as err:
                        raise SystemExit(f'|  EXITING   |  DEPENDENCIES_GET CODE {dependencies_get.status_code}: {err}')

                    try:
                        annotation_get = get(
                            url=f'{args.mlapi_url}/projects/{project_uuid}/documentAI/annotationSets:search',
                            headers=default_headers)
                    except requests.exceptions.HTTPError as err:
                        raise SystemExit(f'|  EXITING   |  ANNOTATION_GET CODE {annotation_get.status_code}: {err}')

                    if project_dict["displayName"] not in document_hierarchy:
                        document_hierarchy[project_dict["displayName"]] = {}

                    document_hierarchy[project_dict["displayName"]][document_dict["displayName"]] = {
                        "annotationSets": [{annotation_set["displayName"]: ""} for annotation_set in
                                           dependencies_get.json()["dependencies"] if
                                           annotation_set["artifactName"][-1] == "1"],
                        "age": f"{np.round(document_age, 3)}",
                        "createTime": f"{document_time.date()} {document_time.time()} UTC"
                    }

                    for annotation_dict in annotation_get.json()['annotationSets']:
                        for i, anno_dict in enumerate(
                                document_hierarchy[project_dict["displayName"]][document_dict["displayName"]][
                                    "annotationSets"]):
                            if list(anno_dict.keys())[0] == annotation_dict["displayName"]:

                                document_hierarchy[project_dict["displayName"]][document_dict["displayName"]][
                                    "annotationSets"][i][annotation_dict["displayName"]] = annotation_dict["status"]

                                if annotation_dict["status"] != "ARTIFACT_STATUS_AVAILABLE":
                                    print(
                                        f"|  WARNING   |  \"{annotation_dict['displayName']}\" reports it\'s status as \"{annotation_dict['status']}\"")
                                break
                    if args.preview == False:
                        print(
                            f'|  DELETING  |  Deleting "{document_dict["displayName"]}" from "{project_dict["displayName"]}" with age {np.round(document_age, 2)} days')
                        document_delete = delete(url=f'{args.mlapi_url}/{document_dict["name"]}:bulkDelete',
                                                     headers=default_headers)
                        if (not document_delete.ok):
                            print(f'|  DELETE FAILED {document_delete.status_code} |  {args.mlapi_url}/{document_dict["name"]}:bulkDelete')
                        else:
                            print(f'|  DELETE SUCCESS {document_delete.status_code} |  {args.mlapi_url}/{document_dict["name"]}:bulkDelete')

    if args.save_preview != None:
        if '.json' not in args.save_preview:
            print(
                f"Document Hierarchy could no be saved. Please specify a \".json\" file for --save_preview {args.save_preview}.")
        else:
            with open(args.save_preview, 'w') as save_hierarchy_json:
                json.dump(document_hierarchy, save_hierarchy_json)

    if args.preview == True:
        return print(json.dumps(document_hierarchy, indent=4))


if __name__ == "__main__":
    main()

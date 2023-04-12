import os
import argparse

import json
import yaml
import subprocess
import requests
import h2o_authn
import regex

from argparse import ArgumentParser, Namespace
from subprocess import Popen, PIPE
from requests import post, get, delete
from h2o_authn import TokenProvider

def arguments() -> ArgumentParser:
    main_parser = ArgumentParser(description = 'Deletion tool to delete pipelines by name or in specific projects')
    authparsers = main_parser.add_subparsers(
        help = 'Authorization method; through curl or the h2o module',
        dest='auth'
        )
    main_parser.add_argument(
        '--projects', type=str, nargs='+', required=False, default=None,
        help = 'Project names wrapped in \"\" and separated by spaces')
    main_parser.add_argument(
        '--pipelines', type=str, nargs='+', required=False, default=None,
        help = 'Pipeline names wrapped in \"\" and separatd by spaces')
    main_parser.add_argument(
        '--scorer_url', type=str, required=True,
        help = 'Document AI scorer url'
    )
    main_parser.add_argument(
        '--preview', type=bool, required=False, default=False,
        help = 'Preview which pipelines will be deleted'
    )
    
    ## Parsers for authentication
    
    parser_h2o = authparsers.add_parser(
        'h2o',
        help='Authentication using the h2o module (requries h2o to be setup prior to use)'
        )
    parser_sso = authparsers.add_parser(
        'sso',
        help='Authentication using \"h2o_authn\" module'
        )
    parser_curl = authparsers.add_parser(
        'curl',
        help='Authentication using curl to website'
        )
    
    ## Necessary curl parameters
    
    parser_curl.add_argument(
        '--auth_url', type=str, default=None,
        help = "Base authorization URL used to recieve access token using curl method"
        )
    parser_curl.add_argument(
        '--client_id', type=str, default=None,
        help = "Keycloak Client ID for curl authorization"
        )
    parser_curl.add_argument(
        '--auth_pass', type=str, default=None,
        help = "Password for Document AI environment."
        )
    parser_curl.add_argument(
        '--auth_user', type=str, default=None,
        help = "Username for Document AI environment"
        )
    
    return main_parser


def get_access_token(args: Namespace) -> str:
    if args.auth == 'h2o':
        access_output, access_error = Popen(['h2o', 'platform', 'access-token'],
                                            stdout=PIPE, stderr=PIPE,universal_newlines=True).communicate()
        if access_error != '':
            return SystemExit(f'|  FAILURE  |  Cannot get access token: {access_error}')
        else:
            return access_output.strip()
        
    elif args.auth == 'curl':
        headers = {
            "Content-Type":"application/x-www-form-urlencoded"
        }
    
        data = {
            "password": args.auth_pass,
            "username": args.auth_user,
            "grant_type": "password",
            "response_type": "token",
            "client_id": args.client_id
        }

        try:
            access_post = post(url = args.auth_url, headers=headers, data=data)
            access_post.raise_for_status()
            return access_post.json()["access_token"]
        except requests.exceptions.HTTPError as err:
            raise SystemExit(f'|  FAILURE  |  Cannot get access token, code {access_post.status_code}:{err}')
        
    elif args.auth == 'sso':
        if os.path.exists('config.yaml') == True:
            with open('config.yaml','r') as f:
                config = yaml.safe_load(f)
        else:            
            raise SystemExit(f'|  EXITING   |  Could not find YAML config file config.yaml')
            
        token_provider = TokenProvider(
            refresh_token = config['platform_token'],
            client_id = config['platform_client_id'],
            token_endpoint_url = config['token_endpoint_url']
        )
        try:
            print('|  SUCCESS  |  Received ACCESS_TOKEN through \"h2o_authn\" library')
            return token_provider.token()
        except Exception as err:
            raise SystemExit(f'|  EXITING   |  Cannot get SSO_TOKEN: {err}')


def delete_pipelines(args: Namespace, ACCESS_TOKEN: str) -> None:
    
    headers = {
        "accept":"application/json",
        "Authorization":f"Bearer {ACCESS_TOKEN}"
    }
    
    pipelines_to_delete = [pipeline for pipeline in args.pipelines]
    
    if args.projects:
        numb_projects = 0
        
        mlapi_url = regex.sub(r'(?<=https:\/\/)[a-zA-z0-9-_]*(?=.)','ml-api', args.scorer_url)
        
        print(f'|   INFO    |  mlapi_url: {mlapi_url}/v1alpha/projects')
        
        try:
            project_get = get(url = f'{mlapi_url}/v1alpha/projects', headers=headers)
            project_get.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(f'|  FAILURE  |  Could not get list of projects, code {project_get.status_code}:{err}')
        for project_dict in project_get.json()['projects']:
            if project_dict['displayName'] in args.projects:
                pipeline_out, pipeline_err = Popen(['curl','-H',f'Authorization: Bearer {ACCESS_TOKEN}',f'{args.scorer_url}/pipeline?projectId={project_dict["name"][9:]}'],
                                universal_newlines=True, stdout=PIPE, stderr=PIPE).communicate()
                if pipeline_out == []:
                    continue
                for pipeline_dict in json.loads(pipeline_out):
                    pipelines_to_delete.append(pipeline_dict['name'])
                numb_projects += 1
            if numb_projects == len(args.projects):
                break
            
    if pipelines_to_delete == []:
        raise SystemExit(f'No pipelines found; empty list: \"{pipelines_to_delete}\"')
            
    for pipeline in pipelines_to_delete:
        if args.preview == False:
            try:
                pipeline_post = delete(f'{args.scorer_url}/pipeline/{pipeline}', headers=headers)
                print(f'|  SUCCESS  |  Deleted \"{pipeline}\"')
                pipeline_post.raise_for_status()
            except requests.exceptions.HTTPError as err:
                print(f'|  FAILURE  |  Failed to delete \"{pipeline}\" code {pipeline_post.status_code}:{err}')
                pass
        else:
            print(f'|   INFO    |  \"{pipeline}\" will be deleted')


def main(argv = None) -> None:
    parser = arguments()
    
    args = parser.parse_args(argv)
   
    if args.auth == None:
        raise SystemExit('Choose either \"h2o\",\"sso\", or \"curl\" authentication')
    
    if args.pipelines and args.projects == None:
        raise SystemExit('Either a pipeline or project containing a pipeline must be included')
   
    ACCESS_TOKEN = get_access_token(args = args)
    
    delete_pipelines(args = args, ACCESS_TOKEN = ACCESS_TOKEN)
    
if __name__ == '__main__':
    main()
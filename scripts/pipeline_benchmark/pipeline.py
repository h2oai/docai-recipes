###    Use with consent of H2O.ai, INC.    ###
###              April 4, 2023             ###

import argparse
import os

import requests
import subprocess
import json
import yaml
import pandas as pd
import numpy as np
import time
import regex
import PyPDF2
import h2o_authn

from PyPDF2 import PdfReader
from tqdm import tqdm, trange
from subprocess import Popen, PIPE, run
from requests import get, post, delete
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone, timedelta
from h2o_authn import TokenProvider


"""
python -m pipeline --pipeline_recipes pipeline_recipes.csv --scorer_url "https://document-ai-scorer.cloud-qa.h2o.ai" --replicas 8 --requests 8 --image_supdir scorer_datasets --datasets 5_pdf cba12_pdf cba12_png sroie_100_pdf docbank_100_jpg --folder_output output --results results.csv --pipeline_list pipeline_list.csv h2o 
"""

def arguments() -> ArgumentParser:
    main_parser = ArgumentParser(description='Tool to automatically create, test, and delete scoring pipelines for the Document AI service.')
    authparsers = main_parser.add_subparsers(
        help = 'Authorization method; through curl or the h2o module',
        dest='auth'
        )
    
    main_parser.add_argument(
        '--pipeline_recipes',type=str,required=True,
        help="CSV containing pertinent information to create desired piplines: pipeline_name, project_token_name, model_token_name, ocr_method, and kubernetes parameters."
        )
    main_parser.add_argument(
        '--scorer_url',type=str,required=True,
        help = "URL for Document AI scorer environment with \"\" enclosing it."
        )
    main_parser.add_argument(
        '--replicas', type=int, required=True,
        help = "Integer number of replicas"
        )
    main_parser.add_argument(
        '--requests', type=int, required=True,
        help = "Integer number of requests"
        )
    main_parser.add_argument(
        '--datasets',type=str,nargs='+',required=True,
        help = "Datasets names separated by spaces, e.g., '--datasets dataset_1 dataset_2 dataset_3'"
        )
    main_parser.add_argument(
        '--pipeline_list',type=str,required=True,
        help = "Name of CSV file that will contain name of created pipelines and or contains names of previously created pipelines"
        )
    main_parser.add_argument(
        '--folder_output',type=str,required=True,
        help = "Folder name where output files will be stored"
        )
    main_parser.add_argument(
        '--results', type=str, required=True, default='results.csv',
        help = "Name of results CSV file where results of tests will be stored."
        )
    
    
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
    
    parser_curl.add_argument(
        '--auth_url', type=str, default=None,
        help = "Base authorization URL used to recieve access token using curl method"
        )
    parser_curl.add_argument(
        '--auth_realm', type=str, default=None,
        help = "Keycloak Realm for Document AI environment"
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
    
    
    main_parser.add_argument(
        '--image_supdir',type=str, required=False,
        help = "Folder which all datasets are stored in."
        )
    main_parser.add_argument(
        '--log_level',type=str, required=False, default='DEBUG', choices=['DEBUG','INFO'],
        help = "Logging level used for Docker scorer container"
        )
    main_parser.add_argument(
        '--docker_v',type=str, required=False, default='0.2.4',
        help = "Docker scorer container version."
        )
    return main_parser

def main(argv=None) -> None:
    
    parser = arguments()
    args = parser.parse_args(argv)
    
    setup_environment(args)
    ACCESS_TOKEN = get_access_token(args)
    pipeline_df = pd.read_csv(args.pipeline_recipes).replace(r'^\s*$', np.nan, regex=True)
    
    wait = False
    try:
        pipeline_list = pd.read_csv(args.pipeline_list)
        if len(pipeline_list.index) == 0:
            wait=True
    except OSError:
        wait=True
    
    pipeline_df = get_uuids(
        args = args,
        ACCESS_TOKEN = ACCESS_TOKEN,
        df = pipeline_df
    )
    
    print('|  SUCCESS  |  Creating pipelines')
    for row in pipeline_df.iterrows():
        create_pipelines(
            args = args,
            ACCESS_TOKEN = ACCESS_TOKEN,
            row = row
            )
    
    if wait == True:
        for i in trange(7000, desc = 'Pipelines Loading'):
            time.sleep(0.01)
        print('\n')
    
    execute_pipelines(
        args = args,
        df = pipeline_df
        )
        
    delete_pipelines(
        args = args,
        df = pipeline_df
        )
   
   
def setup_environment(args: Namespace) -> None:
    if os.path.exists(args.pipeline_recipes) == False:
        recipes_template = pd.DataFrame({
            'pipeline_name':[],
            'project_token_name':[],
            'model_token_name':[],
            'project_page_name':[],
            'model_page_name':[],
            'ocr_method':[],
            'post_processor':[],
            'min_replicas':[],
            'max_replicas':[],
            'requests_cpu':[],
            'requests_memory':[],
            'limits_cpu':[],
            'limits_memory':[]
        })
        recipes_template.to_csv('template.csv',mode='w',index=False)
        SystemExit(f'|  FAILURE  |  No recipe CSV named \"{args.pipeline_recipes}\" found; creating template.')
    
    
    if os.path.exists(args.folder_output) == True:
        None if os.path.exists(f'{args.folder_output}/container_logs') else os.mkdir(f'{args.folder_output}/container_logs')
        None if os.path.exists(f'{args.folder_output}/output') else os.mkdir(f'{args.folder_output}/output')
    
    else:
        print('|  SUCCESS  |  Creating necessary directories')
        os.mkdir(args.folder_output)
        os.mkdir(f'{args.folder_output}/container_logs')
        os.mkdir(f'{args.folder_output}/output')
    
    try:
        with open('config.yaml', 'r') as f:
            config_file = yaml.safe_load(f)
        print('|  SUCCESS  |  Updating \"config.yaml\"')
            
        
        config_file['scorer_base_url'] = f"{args.scorer_url}"
        config_file['out_dir'] = f"{args.folder_output}/output"
        config_file['num_replicas'] = args.replicas
        config_file['num_requests'] = args.requests
        config_file['log_level'] = args.log_level
        if args.auth == 'h2o' or args.auth == 'sso':
            token_info, token_err = Popen(['h2o', 'platform', 'token-info'],
                                   stderr=PIPE,stdout=PIPE,universal_newlines = True).communicate()
            
            token_endpoint_url = regex.search(r'(https:\/\/)[\/a-zA-Z0-9._-]*(\/token)',token_info).group(0)
            platform_client_id = regex.search(r'(?<=Client ID[\s]*)[a-zA-Z0-9-]*(?=\n)',token_info).group(0)
            refresh_token = regex.search(r'(?<=Refresh Token[\s]*)[a-zA-Z0-9-._]*(?=\n)',token_info).group(0)
            
            config_file['token_endpoint_url'] = f'{token_endpoint_url}'
            config_file['platform_client_id'] = f'{platform_client_id}'
            config_file['platform_token'] = f'{refresh_token}'
            
            config_file['auth_base_url'] = None
            config_file['keycloak_client_id'] = None
            config_file['keycloak_realm'] = None
            config_file['docai_user'] = None
            config_file['docai_password'] = None
        elif args.auth == 'curl':
            config_file['auth_base_url'] = f"{args.auth_url}"
            config_file['keycloak_client_id'] = f"{args.client_id}"
            config_file['keycloak_realm'] = f"{args.auth_realm}"
            config_file['docai_user'] = f"{args.auth_user}"
            config_file['docai_password'] = f"{args.auth_pass}"
            
            config_file['token_endpoint_url'] = None
            config_file['platform_client_id'] = None
            config_file['platform_token'] = None
        with open('config.yaml', 'w') as f:
            yaml.dump(config_file, f)
    except OSError:
        print('|  WARNING  |  Could not find \"config.yaml\" file in this directory; creating one.')
        default_config = {
            "auth_base_url": None,
            "benchmark": False,
            "benchmark_results": None,
            "docai_password": None,
            "docai_user": None,
            "dry_run": None,
            "images": None,
            "keycloak_client_id": None,
            "keycloak_realm": None,
            "list_images": False,
            "log_level": 'DEBUG',
            "name": None,
            "num_replicas": None,
            "num_requests": None,
            "out_dir": None,
            "pipeline": None,
            "scorer_base_url": None,
            "scorer_logs": False,
            "temp_image_dir": None,
            "valid_image_file_extensions": [
                '.pdf',
                '.jpeg',
                '.jpg',
                '.png',
                '.bmp',
                '.tiff',
                '.gif'
            ],
            "verbose": False,
            "version": args.docker_v,
            "platform_token": None,
            "token_endpoint_url": None,
            "platform_client_id": None
        }
        with open('config.yaml','w') as f:
            yaml.dump(default_config,f)
        setup_environment(args = args)
     

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
            print('|  SUCCESS   |  Received ACCESS_TOKEN through \"h2o_authn\" library')
            return token_provider.token()
        except Exception as err:
            raise SystemExit(f'|  EXITING   |  SSO_TOKEN: {err}')


def get_uuids(args: Namespace, ACCESS_TOKEN: str, df: pd.DataFrame) -> pd.DataFrame:
    if df['pipeline_name'].isna == True:
        SystemExit('|  FAILURE  |  All pipelines must be named')
    
    
    default_headers = {
        "accept":"application/json",
        "Authorization":f"Bearer {ACCESS_TOKEN}"
    }
    
    mlapi_url = regex.sub(r'(?<=https:\/\/)[a-zA-z0-9-_]*(?=.)','ml-api',args.scorer_url)
    
    try:
        project_get = get(url = f'{mlapi_url}/v1alpha/projects', headers=default_headers)
        project_get.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(f'|  FAILURE  |  Could not get list of projects, code {project_get.status_code}:{err}')
    
    df.insert(df.columns.get_loc('project_token_name'),'project_token_uuid',np.nan)
    df.insert(df.columns.get_loc('model_token_name'),'model_token_uuid',np.nan)
    df.insert(df.columns.get_loc('project_page_name'),'project_page_uuid',np.nan)
    df.insert(df.columns.get_loc('model_page_name'),'model_page_uuid',np.nan)
    
    for project_dict in project_get.json()['projects']:
        
        df.loc[df.project_token_name == project_dict['displayName'],'project_token_uuid'] = project_dict['name'][9:]
        df.loc[df.project_page_name == project_dict['displayName'],'project_page_uuid'] = project_dict['name'][9:]
        
        if df['project_token_name'].isna().sum() == df['project_token_uuid'].isna().sum() and \
        df['project_page_name'].isna().sum() == df['project_page_uuid'].isna().sum():
            break
    
    for project_uuid in pd.concat((df['project_token_uuid'], df['project_page_uuid'])).dropna().unique():
        try:
            model_get = get(
                url = f'{mlapi_url}/v1alpha/projects/{project_uuid}/documentAI/models:search',
                headers=default_headers
            )
            model_get.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(f'|  FAILURE  |  Could not get models, code {model_get.status_code}:{err}')
        
        for model_dict in model_get.json()['models']:
            
            df.loc[df.model_token_name == model_dict['displayName'],'model_token_uuid'] = model_dict['name'][9:]
            df.loc[df.model_page_name == model_dict['displayName'],'model_page_uuid'] = model_dict['name'][9:]
            
            if df['model_token_name'].isna().sum() == df['model_token_uuid'].isna().sum() and \
            df['model_page_name'].isna().sum() == df['model_page_uuid'].isna().sum():
                break
                    
    return df


def create_pipelines(args: Namespace, ACCESS_TOKEN: str, row: pd.Series) -> None:
    pipeline_list = pd.read_csv(args.pipeline_list).to_numpy().flatten()
    
    default_headers = {
        "accept":"application/json",
        "Authorization":f"Bearer {ACCESS_TOKEN}"
    }
    
    ## getting pipeline data ready in 1-liners ##
    token_model = '' if pd.isna(row[1].model_token_name) == True else f"projects/{row[1].model_token_uuid}"
    page_model = '' if pd.isna(row[1].model_page_name) == True else f"projects/{row[1].model_page_uuid}"
    
    if token_model == '' and page_model == '':
        SystemExit(f'|  FAILURE  |  \"{row[1].pipeline_name}\" requires at least a token or page classification model')
    
    name = row[1].pipeline_name if pd.isna(row[1].pipeline_name) == False else SystemExit('|  FAILURE  |  Pipeline missing name')
    ocr_method = row[1].ocr_method if pd.isna(row[1].ocr_method) == False else 'best_text_extract'
    post_processor = row[1].post_processor if pd.isna(row[1].post_processor) == False else 'generic'
    min_replicas = row[1].min_replicas if pd.isna(row[1].min_replicas) == False else 1
    max_replicas = row[1].max_replicas if pd.isna(row[1].max_replicas) == False else 8
    requests_cpu = row[1].requests_cpu if pd.isna(row[1].requests_cpu) == False else '500'
    requests_memory = row[1].requests_memory if pd.isna(row[1].requests_memory) == False else '8'
    limits_cpu = row[1].limits_cpu if pd.isna(row[1].limits_cpu) == False else '4'
    limits_memory = row[1].limits_memory if pd.isna(row[1].limits_memory) == False else '8'
    
    project_uuid = row[1].project_page_uuid if pd.isna(row[1].project_token_uuid) == True else row[1].project_token_uuid 
    
    
    pipeline_data = {
		"name": name,
		"ocrMethod": ocr_method,
		"labellingModelName": token_model,
		"classificationModelName": page_model,
		"builtinPostProcessor": post_processor,
		"customPostProcessor": "",
		"autoscaler": {
			"minReplicas": int(min_replicas),
			"maxReplicas": int(max_replicas)
		},
		"resources": {
			"requests": {
				"cpu": f"{requests_cpu}m",
				"memory": f"{requests_memory}G"
			},
			"limits": {
				"cpu": f"{limits_cpu}",
				"memory": f"{limits_memory}G"
			}
		}
	}
    
    try:
        pipeline_post = post(f'{args.scorer_url}/pipeline?projectId={project_uuid}', headers=default_headers, json=pipeline_data)
        pipeline_post.raise_for_status()
    except requests.exceptions.HTTPError as err:
            print(f'|  FAILURE  |  Could not post {name}, code {pipeline_post.status_code}:{pipeline_post.json()}')
    
    
    df = pd.DataFrame({'pipeline_name': [name]})
    if name not in pipeline_list:
        if os.path.exists(args.pipeline_list) == True:
            df.to_csv(args.pipeline_list,mode='a', index=False, header=False)
        else:
            df.to_csv(args.pipeline_list,mode='w', index=False, header=True)


def execute_pipelines(args: Namespace, df: pd.DataFrame) -> None:
    pipeline_list = pd.read_csv(args.pipeline_list).to_numpy().flatten()
    
    ansi_remove = regex.compile(r'\x1B\[[0-9;]*[a-zA-Z]')
    
    for pipeline_name in pipeline_list:
        for dataset in args.datasets:
            print(f'|  SUCCESS  |  Running \"{dataset}\" through \"{pipeline_name}\"')            
            
            dataset_path = f'{args.image_supdir}/{dataset}'
            
            prep_config(args = args, pipeline_name = pipeline_name , dataset = dataset)
            
            u = run(['id','-u'], capture_output = True, text=True).stdout.strip('\n')
            g = run(['id','-g'], capture_output = True, text=True).stdout.strip('\n')
            
            dockerrun_out, dockerrun_err = Popen(['docker','run','-it','-u',f'{u}:{g}',
                '-v', f'{os.path.abspath(dataset_path)}:/home/appuser/app/cloud_image_dir',
                '-v', f'{os.path.abspath(args.folder_output)}/output:/home/appuser/app/cloud_results_dir',
                '-v', f'{os.path.abspath("config.yaml")}:/home/appuser/app/config.yaml',
                f'docai-scorer:{args.docker_v}', '-i', 'cloud_image_dir', '-o', 'cloud_results_dir', '--scorer-logs'],
                stderr = PIPE, universal_newlines = True).communicate()
            if dockerrun_err != '':
                print(f'|  FAILURE  |  Docker failure running \"{dataset}\" through \"{pipeline_name}\"')
                break
            
            container_name, dockercont_err = Popen(['docker','container','ls','-l','--format','"{{.Names}}"'],
                stdout = PIPE, stderr = PIPE, universal_newlines = True).communicate()
            container_name = container_name.strip('\n"')
            
            container_log_ansi, contlog_err = Popen(['docker','container','logs',container_name],
                stdout = PIPE, stderr = PIPE, universal_newlines = True).communicate()
            
            clean_log = ansi_remove.sub('', container_log_ansi)
            
            with open(f'{args.folder_output}/container_logs/{dataset}_{pipeline_name}_{args.replicas}_{args.requests}.txt','w') as f:
                f.write(clean_log)
                
            contdelete_out, contdelete_err = Popen(['docker', 'rm', container_name],
                stdout = PIPE, stderr = PIPE, universal_newlines = True).communicate()
            
            files = len([entry for entry in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, entry))])

            pdf_pages = 0
            pdfs = [pdf for pdf in os.listdir(dataset_path) if '.pdf' in pdf]
            for pdf in pdfs:
                pdf_pages += len(PdfReader(f'{dataset_path}/{pdf}').pages)
                
            total_pages = files + pdf_pages - len(pdfs)
            
            date_format = r'%Y-%m-%d %H:%M:%S.%f'
            times = [regex.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}', line)[0] for line in clean_log.splitlines()]
            total_time = (datetime.strptime(times[-1],date_format) - datetime.strptime(times[0],date_format)).total_seconds()
            s_per_image = total_time/total_pages
            
            print(f'Seconds per image: {s_per_image}')
            
            result_df = pd.DataFrame({
                'date': [datetime.now(timezone.utc).strftime(r'%Y-%m-%d %H:%M:%S %Z')],
                'replicas': [args.replicas],
                'requests': [args.requests],
                'time': [total_time],
                'sec/image': [s_per_image],
                'files': [files],
                'pages': [total_pages],
                'dataset': [dataset],
                'pipeline': [pipeline_name]
            })
            
            if os.path.exists(args.results):
                result_df.to_csv(args.results, mode='a', index=False, header=False)
            else:
                result_df.to_csv(args.results, mode='w', index=False, header=True)


def delete_pipelines(args: Namespace, df: pd.DataFrame) -> None:    
    ACCESS_TOKEN = get_access_token(args = args)
    
    headers = {
        "Authorization":f"Bearer {ACCESS_TOKEN}"
    }
    
    pipelines_to_delete = df['pipeline_name'].values.tolist()
    
    for pipeline in pipelines_to_delete:
        try:
            pipeline_post = delete(f'{args.scorer_url}/pipeline/{pipeline}', headers=headers)
            print(f'|  SUCCESS  |  Deleted \"{pipeline}\"')
            pipeline_post.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(f'|  FAILURE  |  Failed to delete \"{pipeline}\" code {pipeline_post.status_code}:{err}')
            pass
            
    pipelines_to_keep = pd.DataFrame({'pipeline_name':[pipeline for pipeline in pd.read_csv(args.pipeline_list).values.tolist()[0] if pipeline not in pipelines_to_delete]})
    
    pipelines_to_keep.to_csv(args.pipeline_list, mode='w', index=False, header=True)
 
    
def prep_config(args: list, pipeline_name: str, dataset: str):
    with open('config.yaml','r') as f:
        config_data = yaml.safe_load(f)
    
    config_data['pipeline'] = pipeline_name
    config_data['name'] = f'{dataset}_{pipeline_name}_{args.replicas}-{args.requests}'
    
    with open('config.yaml','w') as f:
        yaml.dump(config_data,f) 

    
if __name__ == "__main__":
    main()
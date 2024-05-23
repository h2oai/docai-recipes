#!/bin/bash

# Script to uninstall helm release of pipelines that are unable to start. It loads
# pipeline deployments and checks number of ready replicas. If there are no ready
# replicas, it uninstalls corresponding helm release (i.e. unpublishes pipeline).
#
# This script is made especially for environments with Document AI 0.8 and lower
# where failing pipelines are not visible in the UI.

function show_help() {
  echo "Usage: $0 -n <namespace> -d"
  echo ""
  echo "Options:"
  echo "  -n, --namespace     Namespace where scorers are published (default docai)"
  echo "  -d, --dry-run       Only print helm releases of failing pipelines. It does not perform any change in the cluster."
  echo "  -h, --help          Display this help message and exit"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

namespace="docai"
dry_run=0

while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--namespace)
      if [[ -n "$2" && "$2" != -* ]]; then
        namespace="$2"
        shift # past argument
        shift # past value
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -d|--dry-run)
      dry_run=1
      shift # past argument
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)    # unknown option
      shift # past argument
      ;;
  esac
done

# Check if helm is installed
if ! command_exists helm; then
    echo "Helm is not installed. Please install Helm."
    exit 1
fi

# Check if kubectl is installed
if ! command_exists kubectl; then
    echo "kubectl is not installed. Please install kubectl."
    exit 1
fi

# Get all pipeline deployments
deployments=$(kubectl get deployments -n "${namespace}" -l app=scorer -o jsonpath="{range .items[*]}{.metadata.namespace} {.metadata.name} {'replicas='}{.status.readyReplicas} {.metadata.annotations.meta\.helm\.sh\/release-name} {.metadata.labels.pipeline}{'\n'}{end}")


# Check if there are any deployments
if [ -z "$deployments" ]; then
    echo "No pipeline deployments (app=scorer) found in namespace $namespace."
    exit 0
fi

if [ "$dry_run" == "1" ]; then
  echo "Dry run mode. Only printing helm releases of failing pipelines:"
fi

# Loop through deployments and uninstall Helm releases where no replica is ready
while read -r namespace deployment_name replicas release_name pipeline; do

  replicas=$(echo "$replicas" | cut -d '=' -f2)

  if [[ "$replicas" == 0  || -z "$replicas" ]]; then

    if [ -z "$release_name" ]; then
        echo "Helm release name not found for pipeline: $pipeline (deployment: $deployment_name, namespace: $namespace)"
        continue
    fi

    if [ "$dry_run" == "1" ]; then
      echo "Helm release: $release_name for pipeline: $pipeline (deployment: $deployment_name, namespace: $namespace)"
      continue
    fi

    echo "Uninstalling Helm release: $release_name for pipeline: $pipeline (deployment: $deployment_name, namespace: $namespace)"
    if helm uninstall "$release_name" -n "$namespace"; then
        echo "Successfully uninstalled Helm release: $release_name"
    else
        echo "Failed to uninstall Helm release: $release_name"
    fi

  fi
done <<< "$deployments"
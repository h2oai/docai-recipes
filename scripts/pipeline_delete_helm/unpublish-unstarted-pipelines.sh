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

if [ "$dry_run" == "1" ]; then
  echo "Dry run mode. Only printing helm releases of failing pipelines:"
fi

found=0
index=1

# Check if there are any deployments
if [ -z "$deployments" ]; then
  echo "No pipeline deployments (app=scorer) found in namespace $namespace."
else
  # Loop through deployments and uninstall Helm releases where no replica is ready
  while read -r ns deployment_name replicas release_name pipeline; do

    replicas=$(echo "$replicas" | cut -d '=' -f2)

    if [[ "$replicas" == 0  || -z "$replicas" ]]; then

      if [ -z "$release_name" ]; then
          echo "Helm release name not found for pipeline: $pipeline (deployment: $deployment_name, namespace: $ns)"
          continue
      fi

      if [ "$dry_run" == "1" ]; then
        found=1
        echo "$index. Unstarted deployment for pipeline: $pipeline (Helm release: $release_name, deployment: $deployment_name, namespace: $ns)"
        ((index++))
        continue
      fi

      echo "Uninstalling Helm release: $release_name for pipeline: $pipeline (deployment: $deployment_name, namespace: $ns)"
      if helm uninstall "$release_name" -n "$ns"; then
          echo "Successfully uninstalled Helm release: $release_name"
      else
          echo "Failed to uninstall Helm release: $release_name"
      fi

    fi
  done <<< "$deployments"
fi

if [[ $found == 0 ]]; then
  echo "No unstarted pipeline deployments found in namespace $namespace."
fi

# Obtaining helm releases in failed state. These were not successfully installed and can be incomplete.
# Using hardcoded string "document-ai-scorer-"  is OK, because it is a name of helm chart which
# is baked in the backend and can't be changed by admin or user.
failing_releases=$(helm ls -n "${namespace}" --failed | grep "document-ai-scorer-" | awk '{print $1}')
if [ -z "$failing_releases" ]; then
  echo "No failing releases for helm chart \"document-ai-scorer\" found in namespace $namespace."
else
  while read -r release_name; do
    if [ "$dry_run" == "1" ]; then
      echo "$index. Failing Helm release: $release_name (namespace: $namespace)"
      ((index++))
      continue
    fi

    echo "Uninstalling Helm release: $release_name"
    if helm uninstall "$release_name" -n "$namespace"; then
        echo "Successfully uninstalled Helm release: $release_name"
    else
        echo "Failed to uninstall Helm release: $release_name"
    fi

  done <<< "$failing_releases"
fi

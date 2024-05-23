### Description

Script to uninstall helm release of unstarted pipelines. It loads pipeline deployments and checks number of ready replicas. If there are not ready replicas, it uninstalls corresponding helm release (i.e. unpublishes pipeline).

This script is made especially for environments with Document AI 0.8 and lower where failing pipelines are not visible in the UI.

### Pre-requisites

It internally uses `kubectl` and `helm`, so it can't work if these tools are not installed on host machine. Script checks for presence of these tools.

### Usage

can be printed with `-h|--help` argument:
```
Usage: ./unpublish-unstarted-pipelines.sh -n <namespace> -d

Options:
  -n, --namespace     Namespace where scorers are published (default docai)
  -d, --dry-run       Only print helm releases of failing pipelines. It does not perform any change in the cluster.
  -h, --help          Display this help message and exit
```

#### Dry run

Only prints helm releases of not started pipelines. It does not perform any change in the cluster.

Example:
```
./unpublish-unstarted-pipelines.sh --namespace docai-scorers --dry-run
```

Example output:
```
Dry run mode. Only printing helm releases of failing pipelines:
Helm release: docai-scorer-011b42 for pipeline: pipeline-b (deployment: docai-scorer-011b42, namespace: docai-scorers)
```

#### Normal run

Uninstalls helm releases of failing pipelines

Example:
```
./unpublish-unstarted-pipelines.sh --namespace docai-scorers
```

Example output:
```
Uninstalling Helm release: docai-scorer-5f76b3 for pipeline: failing (deployment: docai-scorer-5f76b3, namespace: docai-scorers)
release "docai-scorer-5f76b3" uninstalled
Successfully uninstalled Helm release: docai-scorer-5f76b3
Uninstalling Helm release: docai-scorer-a2bfb8 for pipeline: another-pipeline (deployment: docai-scorer-a2bfb8, namespace: docai-scorers)
release "docai-scorer-a2bfb8" uninstalled
Successfully uninstalled Helm release: docai-scorer-a2bfb8
```

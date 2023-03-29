# DOCUMENT AI RECIPE REPOSITORY

This repository is paired with H2O Document AI for sharing custom post-processing scripts, custom pipeline YAML configurations, and other common recipes.


## Custom pipeline YAML configurations
Custom pipeline YAML configurations are used to define the processing steps for DocAI scoring pipelines. These configuration files make it easy to customize the behavior of the pipeline by adjusting the settings in the YAML file. See example yaml files at `pipeline_config` folder.


## Custom Post-Processing Scripts
Custom post-processing scripts, coded in Python, are utilized to modify the output of a Document AI pipeline. These scripts can modify the model's output and generate the final pipeline output. The `post_processor` folder contains various sample scripts that can guide users in creating their own custom post-processing scripts.

## Notebooks
The `notes` folder contains various sample notebooks that can guide users in tuning their own custom post-processing scripts. Specifically, the notebooks show how to tune various post-processing parameters, such as how tokens are grouped together, how line items are extracted, how template methods are used, and other common post-processing tasks. 
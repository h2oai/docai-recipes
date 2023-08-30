# DOCUMENT AI RECIPE REPOSITORY

This repository is paired with H2O Document AI for sharing custom post-processing scripts, custom pipeline YAML configurations, and other common recipes.


## Custom pipeline YAML configurations
Custom pipeline YAML configurations are used to define the processing steps for DocAI scoring pipelines. These configuration files make it easy to customize the behavior of the pipeline by adjusting the settings in the YAML file. See example yaml files at `pipeline_config` folder.


## Custom Post-Processing Scripts
Customize Document AI pipeline output using Python post-processing scripts. These scripts refine model results to generate the final output.

The `post_processor` folder contains various sample scripts to aid users in crafting custom post-processing scripts.

### Version Compatibility
Different DocAI versions offer varying post-processing capabilities. To accommodate this:

1. Browse the relevant folder based on your DocAI version.
2. Select the appropriate post-processing script for your needs.
3. Use the script directly or modify it according to your needs.


## Notebooks
The `notes` folder contains various sample notebooks that can guide users in tuning their own custom post-processing scripts. Specifically, the notebooks show how to tune various post-processing parameters, such as how tokens are grouped together, how line items are extracted, how template methods are used, and other common post-processing tasks. 

## Scripts
Various scripts found in `scripts` to ease burden of testing frontend functionalities, including pipeline benchmarking, pipeline deletion, and deletion of document sets.

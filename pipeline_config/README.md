## To start a custom pipeline with the YAML configuration in H2O Document AI, follow these steps:
1. Navigate to the `Publish Pipeline` page and follow the instructions in this [guide](https://docs.h2o.ai/h2o-document-ai/guide/using-pipelines), until you reach the option for using a custom pipeline.
2. Toggle on `Use custom pipeline` to enable the use of a custom pipeline. 
3. Depending on your specific use cases, you can modify the tasks in the YAML file as follows:
-  OCR-only pipeline:  
\- Option 1: Copy the entire tasks section from the [OCR-only pipeline recipe](https://github.com/h2oai/docai-recipes/blob/main/pipeline_config/pipeline-ocr-only.yaml) and pastte it into the custom pipeline window to replace the existing tasks.   
\- Option 2: Start with the default tasks in the custom pipeline window, remove the `predict` task, and change the `class name` in the `PostProcess` task to `argus.processors.post_processors.ocr_only_post_processor.PostProcessor`
- Conditional processing pipeline:  
\- Copy the entire tasks section from the [conditional processing pipeline](https://github.com/h2oai/docai-recipes/blob/main/pipeline_config/pipeline-conditional-processing.yaml) YAML file and paste it into the custom pipeline window to replace the original tasks. Make sure to adjust the page class names to match those in your data and set the correct model names in the `artifacts` portion. Model name should follow the format  `projects/{project_id}/documentAI/models/{model_id}/versions/1`.  You can find the unique model name for a specific model by selecting it in the UI and toggling on `Use custom pipeline`.


## Additional resources

* [pipeline configs used in unit tests](https://github.com/h2oai/argus-ocr/tree/master/tests/data/config).  These are guaranteed to be valid by running univseral scorer tests against them.

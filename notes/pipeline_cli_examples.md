
A few example commands to invoke argus pipelines from CLI


### A scoring pipeline does OCR and token classifition prediction
```
python -m argus.pipeline \
   -t argus.processors.ocr_processors.Intake root_docs_path=5pdfs follow_symlinks=true \
   -t argus.processors.ocr_processors.PdfTextExtract \
   -t argus.processors.ocr_processors.NormalizeImages resample_to_dpi=300 normalize_image_format=.png \
   -t argus.processors.ocr_processors.GenericOcr ocr_style=TesseractOcr \
   --reorder_inputs None,0 \
   -t argus.processors.train_eval_processors.TrainEvalProcessor base_model_name_or_path=/PathToTokenClassificationModel \
   -t argus.processors.post_processors.generic_post_processor.PostProcessor labeling_threshold=0.2 \
   -o 5pdfs_scoring_in_one_go
```

Note: right now we don't have to reorder the input, we can use a parameter `predict_only` and set it to true.

### A scoring pipeline does OCR and page classifition prediction
```
python -m argus.pipeline \
   -t argus.processors.ocr_processors.Intake root_docs_path=5pdfs follow_symlinks=true \
   -t argus.processors.ocr_processors.PdfTextExtract \
   -t argus.processors.ocr_processors.NormalizeImages resample_to_dpi=300 normalize_image_format=.png \
   -t argus.processors.ocr_processors.GenericOcr ocr_style=TesseractOcr \
   -t argus.processors.train_eval_processors.TrainEvalProcessor base_model_name_or_path=/PathToPageClassificationModel predict_only=true \
   -t argus.processors.post_processors.generic_post_processor.PostProcessor labeling_threshold=0.2 \
   -o 5pdfs_scoring_in_one_go
```

### A scoring pipeline that do conditional processing
#### The scorer does OCR and page class prediction first and then send a documeent to independent token classifier according to the page class
```
python -m argus.pipeline \
  -t argus.processors.ocr_processors.Intake root_docs_path=test_imgs follow_symlinks=true \
  -t argus.processors.ocr_processors.PdfTextExtract \
  -t argus.processors.ocr_processors.NormalizeImages resample_to_dpi=300 normalize_image_format=.png \
  -t argus.processors.ocr_processors.GenericOcr ocr_style=GPUDocTROcr \
  -t argus.processors.train_eval_processors.TrainEvalProcessor base_model_name_or_path=/Path/To/Page_classifier predict_only=true \
  -t argus.processors.train_eval_processors.TrainEvalProcessor base_model_name_or_path=/Path/To/w9_token_classifier predict_only=true when:EOS "
p=pages[0]
if p.attributes.get('page_class','w2')=='w9':
    action=PROCESS
else:
    action=SKIP_PAGE
" EOS \
  -t argus.processors.train_eval_processors.TrainEvalProcessor base_model_name_or_path=/Path/To/w2_token_classifier predict_only=true when:EOS "
p=pages[0]
if p.attributes.get('page_class','w9')=='w2':
    action=PROCESS
else:
    action=SKIP_PAGE
" EOS \
  -t argus.processors.post_processors.generic_post_processor.PostProcessor labeling_threshold=0.2 \
  --write_all_steps \
  -o page_router
```

### A signature detector pipeline
Can be found [here](https://github.com/h2oai/argus-ocr/blob/master/tests/data/mini_dataset/DetectSignature_pas/readme.md). It's used to generate an annoset used in unit tests.

### A pipeline config showing how to process only the first page of each document, dropping all other pages.
Can be found [here](https://github.com/h2oai/argus-ocr/blob/master/tests/data/config/pipeline-page-subset.yaml). It's used to generate an annoset used in unit tests.
Run the snippet like this
```bash
python -m argus.pipeline -c tests/data/config/pipeline-page-subset.yaml -o try
```
See the final annoset like this
```bash
python -m argus.aq -s 'T' -i try/task_3_MiniProgram_0 -r yaml | less
```
The logs show that there are as many output pages as there are documents (even though the input set has some multi-page documents).

from typing import Dict, Any
from argus.processors.post_processors.utils import post_process as pp
from h2o_docai_scorer.post_processors.post_processor_supply_chain import PostProcessor as PostProcessorSupplyChain


class PostProcessor(PostProcessorSupplyChain):
    """Represents a last step in pipeline process that receives all pipeline intermediate
    results and translates them into a final json structure that will be returned to user.
    """

    def get_pages(self) -> Dict[int, Any]:
        return super().get_pages()

    def get_entities(self):
        if not self.has_labelling_model:
            return []

        docs = pp.post_process_predictions(
            model_preds=self.label_via_predictions,
            top_n_preds=self.label_top_n,
            token_merge_type="MIXED_MERGE",
            token_merge_xdist_regular=1.0,
            label_merge_x_regular="ALL",
            token_merge_xydist_regular=1.0,
            label_merge_xy_regular="address",
            token_merge_xdist_wide=1.5,
            label_merge_x_wide="phone|fax",
            output_labels="INCLUDE_O",
            verbose=True,
        )

        df_list = []
        for doc in docs:
            predictions = docs[doc]
            predictions = predictions.round(decimals=4)
            for idx, row in predictions.iterrows():
                df_list.append(row.to_dict())
        return df_list

    '''
    Converting the dictionary to a dataframe
    import pandas as pd
    import json
    
    f = open('result.json')
    dict_data = json.load(f)
    df = pd.DataFrame(dict_data['entities'])
    df.to_csv('result.csv')
    '''

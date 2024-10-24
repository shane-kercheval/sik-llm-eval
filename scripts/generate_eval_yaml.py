"""
Generate YAML files for the ChatRAG-Bench dataset.

In order to run this script, you need to have the ChatRAG-Bench dataset downloaded and processed in the /scripts/data/nvidia_processed directory.
Move the script to the root directory and run it.
"""

import pandas as pd

from llm_eval.eval import Eval
from llm_eval.checks import PrecisionCheck, RecallCheck, F1ScoreCheck
import uuid

if __name__ == '__main__':
    df = pd.read_parquet('scripts/data/nvidia_processed/nvidia_datasets_combined.parquet')

    # Map object to boolean
    df['answerable'] = df['answerable'].map({'True': True, 'False': False})

    ROWS_PER_DATASET = 100
    subset_df = df.groupby('dataset').apply(lambda x: x.sample(n=ROWS_PER_DATASET))

    for index, row in subset_df.iterrows():
        print(f'Processing RAG agent for: {index}|{row["dataset"]}')
        
        # Construct input dict
        system_prompt = row['system_prompt'].replace('System: ', '', 1)
        system_prompt_dict = {
            'content': system_prompt,
            'role': 'system',
        }
        message_list = row['input'].tolist()
        message_list.insert(0, system_prompt_dict)

        documents_list = row['documents'].tolist()
        input = {
            'messages': message_list,
            'documents': documents_list,
        }

        checks = [
            PrecisionCheck(),
            RecallCheck(),
            F1ScoreCheck(),
        ]
        
        eval_ = Eval(
            metadata={
                'uuid': str(uuid.uuid4()),
                'benchmark': 'Nvidia ChatRag-Bench',
                'dataset': row['dataset'],
                'uses_retrieval': row['uses_retrieval'],
                'answerable_from_context': row['answerable'],
                'tags': [
                    'RAG',
                    'retrieval',
                ],
                'source': row['reference'],
                'license': 'Creative Commons Attribution 3.0',
            },
            input=input,
            ideal_response=row['ground_truth_answers'].tolist(),
            checks=checks,
        )
        name = f"{index[1]}_{row['dataset']}"
        file_name = f'scripts/data/evals/{name}.yaml'
        eval_.to_yaml(file_name)
        # ensure it can be saved and re-loaded
        assert Eval.from_yaml(file_name)
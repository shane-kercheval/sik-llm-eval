"""
Generate YAML files for the ChatRAG-Bench dataset.

In order to run this script, you need to have the ChatRAG-Bench dataset downloaded and processed in
the /scripts/data/nvidia_processed directory. Move the script to the root directory and run it.

Additionally these libraries are required:

pip install datasets
pip install chromadb
"""

import os
import time
from copy import deepcopy
from openai import OpenAI
import pandas as pd
import uuid
import chromadb
from datasets import load_dataset
from llm_eval.eval import Eval
from llm_eval.checks import MatchCheck, MaxF1Score

from dotenv import load_dotenv
load_dotenv()


NVIDIA_DIR = 'examples/nvidia_chatrag_bench'
NVIDIA_DATA_DIR = os.path.join(NVIDIA_DIR, 'data')
NVIDIA_DATA_RAW_DIR = os.path.join(NVIDIA_DATA_DIR, 'huggingface_datasets')
NVIDIA_DATASET_PATH = os.path.join(NVIDIA_DATA_DIR, 'nvidia_datasets_combined.parquet')
EVALS_DIR = os.path.join(NVIDIA_DIR, 'evals')

def download_datasets() -> None:
    """
    Downloads/saves datasets from ChatRAG-Bench (test splits only).

    https://huggingface.co/datasets/nvidia/ChatRAG-Bench
    """
    # check if _test.parquet files already exist in the dataset directory
    # if they do, skip downloading
    if os.path.exists(NVIDIA_DATA_RAW_DIR):
        existing_files = os.listdir(NVIDIA_DATA_RAW_DIR)
        if any('_test.parquet' in file for file in existing_files):
            print('Data already downloaded. Skipping download.')
            return
    dataset_names = [
        'coqa', 'inscit', 'topiocqa', 'hybridial', 'doc2dial', 'quac', 'qrecc', 'doqa_cooking',
        'doqa_movies', 'doqa_travel', 'sqa',
    ]
    for dataset_name in dataset_names:
        print(f'Downloading `{dataset_name}`')
        dataset = load_dataset('nvidia/ChatRAG-Bench', dataset_name)
        if 'test' in dataset:
            file_path = os.path.join(NVIDIA_DATA_RAW_DIR, f'{dataset_name}_test.parquet')
            dataset['test'].to_parquet(file_path)
            print(f'{dataset_name}_test.parquet saved')
        else:
            print(f'skipping `{dataset_name}` - no test split')


def process_combine_datasets() -> None:  # noqa: PLR0915
    """
    Processes and combines datasets from ChatRAG-Bench. Saves the combined dataset to a parquet
    file. Each dataset has a slightly different structure, so we need to process them separately
    and combine them into a single dataset.

    'uses_retrieval' means that the RAG agent retrieves the most relevant documents from the
    `documents` value. Datasets like `doc2dial` have many documents per example, and the agent must
    select the most relevant one. Datasets like `doqa` have only one (or a few) documents per
    example, and the agent must use it as the context.
    """
    if os.path.exists(NVIDIA_DATASET_PATH):
        print('Combined dataset already exists. Skipping processing.')
        return

    doc2dial = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'doc2dial_test.parquet'))
    print(f"doc2dial: {doc2dial.shape}")

    doqa_cooking = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'doqa_cooking_test.parquet'))
    print(f"doqa_cooking: {doqa_cooking.shape}")

    doqa_movies = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'doqa_movies_test.parquet'))
    print(f"doqa_movies: {doqa_movies.shape}")

    doqa_travel = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'doqa_travel_test.parquet'))
    print(f"doqa_travel: {doqa_travel.shape}")

    hybridial = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'hybridial_test.parquet'))
    print(f"hybridial: {hybridial.shape}")

    qrecc = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'qrecc_test.parquet'))
    print(f"qrecc: {qrecc.shape}")

    quac = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'quac_test.parquet'))
    print(f"quac: {quac.shape}")

    sqa = pd.read_parquet(os.path.join(NVIDIA_DATA_RAW_DIR, 'sqa_test.parquet'))
    print(f"sqa: {sqa.shape}")

    def process_dataset(
            dataset: pd.DataFrame,
            dataset_name: str,
            subset:str | None = None,
            reference: str | None = None,
            dataset_license: str | None = None,
            system_prompt: str | None = None,
            additional_instructions: str | None = None,
        ) -> pd.DataFrame:
        final_dataset = []
        if dataset_license is None:
            dataset_license = 'Unknown'
        for row in dataset.to_dict(orient='records'):
            documents = row['ctxs'].tolist()
            # get unique text; create a lookup table for text to index
            unique_text = {x['text'] for x in documents}
            text_index_lookup = {text: index for index, text in enumerate(unique_text)}
            # ensure no duplicate indices
            assert len(text_index_lookup) == len(unique_text)
            # for each document, if we haven't already seen it, add it to the list and set doc_id
            exiting_ids = set()
            unique_documents = []
            for doc in documents:
                doc_id = text_index_lookup[doc['text']]
                if doc_id not in exiting_ids:
                    exiting_ids.add(doc_id)
                    doc['doc_id'] = doc_id
                    assert 'text' in doc
                    assert 'title' in doc
                    assert 'doc_id' in doc
                    unique_documents.append(doc)
            ground_truth_doc_id = None
            if 'ground_truth_ctx' in row:
                ground_truth_doc_id = text_index_lookup.get(row['ground_truth_ctx']['ctx'], None)
                if ground_truth_doc_id is None:
                    print(f"{dataset_name} - skipping example where ground truth document is not found in documents")  # noqa: E501
                    continue
                assert ground_truth_doc_id in {x['doc_id'] for x in unique_documents}

            input_messages = row['messages'].tolist()
            if additional_instructions:
                input_messages = [
                    {'content': additional_instructions, 'role': 'assistant'},
                    *input_messages,
                ]
            final_dataset.append({
                'dataset': dataset_name,
                'subset': subset,
                'answerable': not any('cannot find the answer' in x for x in row['answers']),
                'uses_retrieval': len(unique_documents) > 1,
                'reference': reference,
                'system_prompt': system_prompt,
                'input': input_messages,
                'documents': unique_documents,
                'ground_truth_doc_id': ground_truth_doc_id,
                'ground_truth_answers': row['answers'].tolist(),
                'license': dataset_license,
            })
        return final_dataset

    # from nvidia_src/dataset.py
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."  # noqa: E501
    final_dataset = []
    final_dataset.extend(process_dataset(
        doc2dial, 'doc2dial',
        reference='https://huggingface.co/datasets/IBM/doc2dial',
        system_prompt=system,
        dataset_license="Creative Commons Attribution 3.0 Unported License",
    ))
    final_dataset.extend(process_dataset(
        doqa_cooking, 'doqa', subset='cooking',
        reference='https://huggingface.co/datasets/community-datasets/doqa',
        system_prompt=system,
    ))
    final_dataset.extend(process_dataset(
        doqa_movies, 'doqa', subset='movies',
        reference='https://huggingface.co/datasets/community-datasets/doqa',
        system_prompt=system,
    ))
    final_dataset.extend(process_dataset(
        doqa_travel, 'doqa', subset='travel',
        reference='https://huggingface.co/datasets/community-datasets/doqa',
        system_prompt=system,
    ))
    final_dataset.extend(process_dataset(
        hybridial, 'hybridial',
        reference='https://github.com/entitize/HybridDialogue?tab=readme-ov-file',
        system_prompt=system,
    ))
    final_dataset.extend(process_dataset(
        qrecc, 'qrecc',
        system_prompt=system,
    ))
    final_dataset.extend(process_dataset(
        quac, 'quac',
        reference='https://www.tensorflow.org/datasets/catalog/quac',
        system_prompt=system,
        dataset_license="Apache 2.0 License",
    ))
    final_dataset.extend(process_dataset(
        sqa, 'sqa',
        reference='https://www.tensorflow.org/datasets/catalog/sqa',
        system_prompt=system,
        additional_instructions="Answer the following question with one or a list of items.",
    ))

    final_dataset = pd.DataFrame(final_dataset)
    final_dataset['ground_truth_doc_id'] = final_dataset['ground_truth_doc_id'].astype('Int64')

    assert set(final_dataset['answerable'].value_counts().index.tolist()) == {True, False}
    assert set(final_dataset['uses_retrieval'].value_counts().index.tolist()) == {True, False}
    assert final_dataset['input'].isna().sum() == 0
    assert final_dataset['input'].apply(lambda x: isinstance(x, list) and isinstance(x[0], dict)).all()  # noqa: E501
    assert final_dataset['documents'].isna().sum() == 0

    for i, row in final_dataset.iterrows():
        assert isinstance(row['documents'], list)
        for item in row['documents']:
            assert isinstance(item, dict)
            assert 'text' in item
            assert 'title' in item
            assert 'doc_id' in item
        if row['uses_retrieval']:
            assert len(row['documents']) > 1
        else:
            assert len(row['documents']) == 1
        if not pd.isna(row['ground_truth_doc_id']):
            assert isinstance(row['ground_truth_doc_id'], int)
            assert row['ground_truth_doc_id'] >= 0
            assert row['ground_truth_doc_id'] < len(row['documents'])
    assert final_dataset['ground_truth_answers'].isna().sum() == 0
    assert final_dataset['ground_truth_answers'].apply(lambda x: isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)).all()  # noqa: E501
    # ensure that the final dataset has unique documents
    assert final_dataset['documents'].apply(lambda x: len(x) == len({doc['doc_id'] for doc in x})).all()  # noqa: E501
    final_dataset.to_parquet(NVIDIA_DATASET_PATH)


def build_evals(sample_size_per_dataset: int) -> None:
    """
    Builds YAML files in the format that the llm-eval framework expects for the ChatRAG-Bench
    dataset.

    Args:
        sample_size_per_dataset: Number of rows to sample from the combined dataset, PER DATASET.
    """
    # check if evals already exist; if they do; skip
    if os.path.exists(EVALS_DIR):
        existing_files = os.listdir(EVALS_DIR)
        if any('.yaml' in file for file in existing_files):
            print('Evals already exist. Skipping eval creation.')
            return
    df = pd.read_parquet(NVIDIA_DATASET_PATH)  # noqa: PD901
    df['subset'] = df['subset'].fillna('')
    subset_df = df.groupby(['dataset', 'subset']).apply(lambda x: x.sample(n=sample_size_per_dataset))  # noqa: E501
    for index, row in subset_df.iterrows():
        print(f'Creating eval for: {index}|{row["dataset"]}')
        message_list = row['input'].tolist()
        documents_list = row['documents'].tolist()
        _input = {
            'messages': message_list,
            'documents': documents_list,
        }
        # The MatchCheck needs different information than the F1Score, so the candidate
        # or agent needs to return both the relevant document index and the generated response.
        # And the checks need to know where to find both.
        match_value_extractor = "response['relevant_document_id']"
        # the scores need both the `actual_response` and the `ideal_response`; Those are the names
        # of the parameters of the `call` method of these classes, so those names need to be the
        # keys in the value_extractor dictionary. The dictionary will be passed to the object
        # using keyword arguments.
        score_value_extractor = {
            'actual_response': "response['generated_response']",
            'ideal_response': 'ideal_response',
        }
        checks = []
        if pd.notna(row['ground_truth_doc_id']) and len(row['documents']) > 1:
            # we only want to do the match check for the evals that have a ground truth doc id
            checks += [
                MatchCheck(
                    value_extractor=match_value_extractor,
                    value=str(row['ground_truth_doc_id']),
                ),
            ]
        checks += [MaxF1Score(value_extractor=score_value_extractor, return_precision_recall=True)]
        unique_id = str(uuid.uuid4())
        eval_ = Eval(
            metadata={
                'uuid': unique_id,
                'benchmark': 'Nvidia ChatRag-Bench',
                'dataset': row['dataset'],
                'uses_retrieval': row['uses_retrieval'],
                'answerable_from_context': row['answerable'],
                'tags': ['RAG'],
                'source': row['reference'],
                'license': row['license'],
            },
            input=_input,
            ideal_response=row['ground_truth_answers'].tolist(),
            checks=checks,
        )
        file_name = os.path.join(EVALS_DIR, f'{unique_id}.yaml')
        eval_.to_yaml(file_name)
        # ensure it can be saved and re-loaded
        assert Eval.from_yaml(file_name)


class SimpleRAGAgent:
    """
    SimpleRAGAgent class used as an example for running evals against the ChatRAG-Bench
    dataset.
    """

    def __init__(
            self,
            model_name: str,
            use_all_messages_in_retrieval: bool,
            system_message: str = "",
            ):
        """
        Initialize the RAG agent.

        Args:
            model_name: The name of the ChatGPT model to use for the RAG agent e.g. 'gpt-4o-mini'.
            use_all_messages_in_retrieval: Whether to use all messages in conversation
                history for retrieval or only the last user message.
            system_message: Optional system message to guide the LLM in response generation.
        """
        self.model_name = model_name
        self.use_all_messages_in_retrieval = use_all_messages_in_retrieval
        self.system_message = system_message
        self.collection = None
        self.documents = []
        self.timings = {
            'document_storage_time': None,
            'retrieval_time': None,
            'inference_time': None,
        }

    def add_documents(self, documents: list[dict[str, int]]) -> None:
        """
        Add documents to the RAG agent's collection for retrieval.

        Args:
            documents (list[dict]): List of documents with 'text' and 'doc_id' keys.
        """
        self.documents = documents
        # if there is more than one document that means we will need to retrieve the most relevant
        # document from the collection
        if len(documents) > 1:
            start_time = time.time()
            chroma_client = chromadb.Client()
            self.collection = chroma_client.create_collection(name=str(uuid.uuid4()))
            docs = [doc['text'] for doc in documents]
            ids = [str(doc['doc_id']) for doc in documents]
            self.collection.add(documents=docs, ids=ids)
            self.timings['document_storage_time'] = time.time() - start_time

    def __call__(self, messages: list[dict]) -> tuple[int, str, dict]:
        """
        Processes the input messages, retrieves the most relevant document, and generates a response.

        Args:
            messages: Conversation history with roles and content.

        Returns:
            tuple[int, str, dict]: Tuple containing the index of the relevant document, the generated
            response, and a dictionary with retrieval and LLM processing times.
        """
        call_start_time = time.time()
        messages = deepcopy(messages)
        assert messages[-1]['role'] == 'user'
        # if there is more than one document that means we will need to retrieve the most relevant
        # document from the collection; otherwise, we can use the single document as the context
        if len(self.documents) > 1:
            retrieval_start_time = time.time()
            if self.use_all_messages_in_retrieval:
                query = " ".join([message["content"] for message in messages])
            else:
                query = messages[-1]["content"]
            # Query the collection for the most relevant document
            results = self.collection.query(query_texts=[query], n_results=1)
            # Extract the most relevant document's index
            relevant_document_id = int(results["ids"][0][0])

            self.timings['retrieval_time'] = time.time() - retrieval_start_time
            # documents are in format `[{'doc_id': 92, 'text': 'More Help F...`}]; we need to find
            # the document with the relevant doc_id
            relevant_document_text = next(
                doc['text'] for doc in self.documents if doc['doc_id'] == relevant_document_id
            )
        else:
            assert len(self.documents) == 1
            relevant_document_id = None
            relevant_document_text = self.documents[0]['text']

        # Prepare messages with the relevant document and optional system message
        if self.system_message:
            messages.insert(0, {'role': 'system', 'content': self.system_message})

        messages[-1]['content'] += f"\n\nRelevant Information:\n\n```\n{relevant_document_text}\n```"  # noqa: E501
        # Generate the response using the OpenAI API
        inference_start_time = time.time()
        response = OpenAI().chat.completions.create(model=self.model_name, messages=messages)
        generated_response = response.choices[0].message.content.strip()
        self.timings['inference_time'] = time.time() - inference_start_time
        self.timings['total_time'] = time.time() - call_start_time
        return relevant_document_id, generated_response, self.timings

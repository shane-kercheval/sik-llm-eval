import os
import pandas as pd

# 'uses_retrieval': means that the RAG agent retrieves the most relevant documents from the `documents` value.
#     datasets like doc2dial have many documents per example, and the agent must select the most relevant one.
#     datasets like doqa have only one (or a few) documents per example, and the agent must use it as the context.
#


data_dir = 'data/nvidia_raw'

doc2dial = pd.read_parquet(os.path.join(data_dir, 'doc2dial_test.parquet'))
print(f"doc2dial: {doc2dial.shape}")

doqa_cooking = pd.read_parquet(os.path.join(data_dir, 'doqa_cooking_test.parquet'))
print(f"doqa_cooking: {doqa_cooking.shape}")

doqa_movies = pd.read_parquet(os.path.join(data_dir, 'doqa_movies_test.parquet'))
print(f"doqa_movies: {doqa_movies.shape}")

doqa_travel = pd.read_parquet(os.path.join(data_dir, 'doqa_travel_test.parquet'))
print(f"doqa_travel: {doqa_travel.shape}")

hybridial = pd.read_parquet(os.path.join(data_dir, 'hybridial_test.parquet'))
print(f"hybridial: {hybridial.shape}")

qrecc = pd.read_parquet(os.path.join(data_dir, 'qrecc_test.parquet'))
print(f"qrecc: {qrecc.shape}")

quac = pd.read_parquet(os.path.join(data_dir, 'quac_test.parquet'))
print(f"quac: {quac.shape}")

sqa = pd.read_parquet(os.path.join(data_dir, 'sqa_test.parquet'))
print(f"sqa: {sqa.shape}")


output_dir = 'data/nvidia_processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_dataset(dataset, dataset_name, subset=None, reference=None, system_prompt=None, additional_instructions=None):
    final_dataset = []
    for row in dataset.to_dict(orient='records'):
        documents = row['ctxs'].tolist()
        # get unique text; create a lookup table for text to index
        unique_text = set([x['text'] for x in documents])
        text_index_lookup = {text: index for index, text in enumerate(unique_text)}
        # ensure no duplicate indices
        assert len(text_index_lookup) == len(unique_text)

        # for each document, if we haven't already seen it, add it to the list and set the doc_id
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
                print(f"{dataset_name} - skipping example where ground truth document is not found in documents")
                continue
            else:
                assert ground_truth_doc_id in set([x['doc_id'] for x in unique_documents])
        
        input_messages = row['messages'].tolist()
        if additional_instructions:
            input_messages = [{'content': additional_instructions, 'role': 'assistant'}] + input_messages
        
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
        })
    
    return final_dataset

# from nvidia_src/dataset.py
system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
final_dataset = []
final_dataset.extend(process_dataset(
    doc2dial, 'doc2dial',
    reference='https://huggingface.co/datasets/IBM/doc2dial',
    system_prompt=system,
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
assert final_dataset['input'].isnull().sum() == 0
assert final_dataset['input'].apply(lambda x: isinstance(x, list) and isinstance(x[0], dict)).all()
assert final_dataset['documents'].isnull().sum() == 0

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
assert final_dataset['ground_truth_answers'].isnull().sum() == 0
assert final_dataset['ground_truth_answers'].apply(lambda x: isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)).all()
# ensure that the final dataset has unique documents
assert final_dataset['documents'].apply(lambda x: len(x) == len(set([doc['doc_id'] for doc in x]))).all()

final_dataset.to_parquet(os.path.join(output_dir, 'nvidia_datasets_combined.parquet'))

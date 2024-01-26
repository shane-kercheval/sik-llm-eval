import os
import time
import yaml
from llm_evals.eval import Eval, Candidate
from llm_evals.llms.openai import OpenAIChat
from llm_evals.eval import eval_result_summarizer
from pprint import pprint
import asyncio
from dotenv import load_dotenv
load_dotenv()


def fetch_data(i, eval_, candidate):
    print(f"{i}Starting data fetch...", flush=True)
    # start = time.time()
    eval_._call_candidate(candidate)
    # response = eval_(candidate)
    # time.sleep(2)  # Simulates a delay, e.g., network delay
    # response = 'hello'
    # chat = OpenAIChat(temperature=1)
    # response = chat("Hello")
    # response = candidate("Hello. How are you?")
    # end = time.time()
    print(f"{i}Data fetched")
    return {
        'i': i,
        'eval_obj': eval_,
        # "time": duration,
        # 'response_len': len(responses[0]),
        # 'response': responses[0][0:50],
    }

async def main():
    with open('../evals/templates/candidate_openai_3.5_1106.yaml') as f:
        cand_config = yaml.safe_load(f)

    # with open('../evals/templates/candidate_hugging_face_endpoint_mistral_a10g.yaml') as f:
    #     cand_config = yaml.safe_load(f)
    #     cand_config['model_parameters']['endpoint_url'] = os.getenv('HUGGING_FACE_ENDPOINT_UNIT_TESTS')

    with open('../tests/fake_data/fake_eval_sum_two_numbers.yaml') as f:
        eval_config = yaml.safe_load(f)

    candidate = Candidate.from_dict(cand_config)
    eval_ = Eval(**eval_config)

    # results = fetch_data(1, eval_, candidate)
    # results['eval_obj'].responses


    from pprint import pprint
    import asyncio

    # async def main():
    # loop = asyncio.get_running_loop()
    loop = asyncio.get_event_loop()
    # List of tasks to run fetch_data with different arguments
    tasks = [
        loop.run_in_executor(
            None,
            fetch_data,
            i,
            Eval(**eval_config),
            Candidate.from_dict(cand_config),
        )
        for i in range(10)
    ]

    # Wait for all tasks to complete and gather results
    results = await asyncio.gather(*tasks)
        # print("Fetched data:", results)
    # asyncio.run(main())
    # Directly await the coroutine in a Jupyter Notebook cell
    # await main()
    for result in results:
        print((round(result['eval_obj'].duration, 1), result['eval_obj'].responses[0][0:50]))

    eval_results = [r['eval_obj']._execute_eval() for r in results]

    for er in eval_results:
        print(eval_result_summarizer(er))



start = time.time()
asyncio.run(main())
end = time.time()
print(f"Total time: {end - start}")

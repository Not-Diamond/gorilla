from datetime import datetime
from pathlib import Path
import argparse,json,os
from tqdm import tqdm
from model_handler.handler_map import handler_map
from model_handler.model_style import ModelStyle

from dotenv import load_dotenv

from openfunctions_evaluation import build_handler, load_file


test_categories = {
    "executable_simple": "gorilla_openfunctions_v1_test_executable_simple.json",
    "executable_parallel_function": "gorilla_openfunctions_v1_test_executable_parallel_function.json",
    "executable_multiple_function": "gorilla_openfunctions_v1_test_executable_multiple_function.json",
    "executable_parallel_multiple_function": "gorilla_openfunctions_v1_test_executable_parallel_multiple_function.json",
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "relevance": "gorilla_openfunctions_v1_test_relevance.json",
    "parallel_function": "gorilla_openfunctions_v1_test_parallel_function.json",
    "multiple_function": "gorilla_openfunctions_v1_test_multiple_function.json",
    "parallel_multiple_function": "gorilla_openfunctions_v1_test_parallel_multiple_function.json",
    "java": "gorilla_openfunctions_v1_test_java.json",
    "javascript": "gorilla_openfunctions_v1_test_javascript.json",
    "rest": "gorilla_openfunctions_v1_test_rest.json",
    "sql": "gorilla_openfunctions_v1_test_sql.json",
}


ALL_MODELS = [
    "gpt-3.5-turbo-FC",
    "gpt-4-FC",
    "gpt-4-1106-preview-FC",
    "gpt-4-turbo-preview-FC",
    "claude-3-opus-20240229-FC",
    "claude-3-sonnet-20240229-FC",
    "claude-3-haiku-20240307-FC",
    "gemini-pro",
    "mistral-small-latest-FC-Auto",
    "mistral-large-latest-FC-Auto"
    "Mixtral-8x7B-Instruct-v0.1-FC",
    "Mistral-7B-Instruct-v0.1-FC",
    "command-r-FC"
]


if __name__ == "__main__":
    load_dotenv("../../.env")
    temperature = 0.
    top_p = 1
    max_tokens = 1200
    test_category = "all"
    exec_time_str = dir_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path("./result") / exec_time_str
    for model in ALL_MODELS:
        handler = build_handler(model, temperature, top_p, max_tokens)
        if handler.model_style == ModelStyle.OSSMODEL:
            raise NotImplementedError
        else:
            test_cate, files_to_open = load_file(test_category)
            for test_category, file_to_open in zip(test_cate, files_to_open):
                print("Generating: " + file_to_open)
                test_cases = []
                with open("./data/" + file_to_open) as f:
                    for line in f:
                        test_cases.append(json.loads(line))
                num_existing_result = 0  # if the result file already exists, skip the test cases that have been tested.
                if os.path.exists(save_path / model / file_to_open.replace(".json", "_result.json")):
                    with open(save_path / model / file_to_open.replace(".json", "_result.json")) as f:
                        for line in f:
                            num_existing_result += 1
                for index, test_case in enumerate(tqdm(test_cases)):
                    if index < num_existing_result:
                        continue
                    user_question,functions = test_case["question"], test_case["function"]
                    if type(functions) is dict or type(functions) is str:
                        functions = [functions]
                    result, metadata = handler.inference(user_question, functions,test_category)
                    result_to_write = {
                        "idx": index,
                        "result": result,
                        "input_token_count": metadata["input_tokens"],
                        "output_token_count": metadata["output_tokens"],
                        "latency": metadata["latency"],
                        "message": metadata["message"]
                    }
                    handler.write(result_to_write, save_path, file_to_open)

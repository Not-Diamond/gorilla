import sys

sys.path.append("../")

from checker import ast_checker, executable_checker, executable_checker_rest
from eval_runner_helper import *
from tqdm import tqdm
import argparse


from dotenv import load_dotenv

# NOTE: This file should be run in the `eval_checker` directory


def single_executable_file_runner(
    handler, model_result, prompt, model_name, test_category
):
    assert len(model_result) == len(prompt)

    result = []
    correct_count = 0
    for i in tqdm(range(len(model_result))):
        raw_result = model_result[i]["result"]
        message = model_result[i]["message"]
        temp = {
            "id": i + 1,
            "valid": False,
            "model_name": model_name.replace("-FC", ""),
            "test_category": test_category,
            "message": message,
            "prompt": prompt[i],
            "model_result_raw": raw_result,
            "training_prompt": model_result[i]["message"]
        }
        try:
            decoded_result = handler.decode_execute(raw_result)
            temp["score"] = 1
        except Exception as e:
            temp["score"] = 0
            temp["error"] = [f"Failed to decode executable. {str(e)}"]
            temp["error_type"] = "executable_decoder:decoder_failed"
            result.append(temp)
            continue

        if "rest" in test_category:
            # REST is always single-functioned. Therefore we take the first one and pass it to the REST checker.
            if not is_rest_format_output(decoded_result):
                temp["score"] = 0
                temp["error"] = ["Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."]
                temp["error_type"] = "executable_decoder:rest_wrong_output_format"
                temp["model_result_decoded"] = str(decoded_result)
                result.append(temp)
                continue

            checker_result = executable_checker_rest(decoded_result[0], i)

        else:
            if not is_executable_format_output(decoded_result):
                temp["score"] = 0
                temp["error"] = ["Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."]
                temp["error_type"] = "executable_decoder:wrong_output_format"
                temp["model_result_decoded"] = str(decoded_result)
                result.append(temp)
                continue

            prompt_item = prompt[i]
            checker_result = executable_checker(decoded_result, prompt_item)

        if checker_result["valid"]:
            temp["score"] = 1
            correct_count += 1
        else:
            temp["score"] = 0

        result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = test_category + "_score.json"
    output_file_dir = os.path.join(OUTPUT_PATH, model_name)
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def single_relevance_file_runner(handler, model_result, model_name, test_category):

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        success = False
        decoded_result = None

        try:
            decoded_result = handler.decode_ast(model_result_item, language="Python")
            success = False
            if is_empty_output(decoded_result):
                success = True

        except Exception as e:
            success = True

        temp = {}
        temp["id"] = i + 1
        temp["model_name"] = model_name.replace("-FC", "")
        temp["test_category"] = test_category
        temp["valid"] = success
        temp["model_result"] = model_result_item
        temp["decoded_result"] = decoded_result
        temp["training_prompt"] = model_result[i]["message"]
        if success:
            correct_count += 1
            temp["score"] = 1
        else:
            temp["score"] = 0
            temp["error"] = [
                f"Valid syntax. Successfully decode AST when it should not."
            ]
            temp["error_type"] = "relevance_error:decoder_success"

        result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = test_category + "_score.json"
    output_file_dir = os.path.join(OUTPUT_PATH, model_name)
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def single_ast_file_runner(
    handler, model_result, prompt, possible_answer, language, test_category, model_name
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), "The length of the model result does not match the length of the prompt or possible answer. Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        training_prompt = model_result[i]["message"]
        prompt_item = prompt[i]["function"]
        possible_answer_item = possible_answer[i]
        temp = {
            "id": i + 1,
            "model_name": model_name,
            "test_category": test_category,
            "prompt": prompt[i],
            "model_result_raw": model_result_item,
            "possible_answer": possible_answer_item,
            "training_prompt": training_prompt
        }

        try:
            model_result_item = handler.decode_ast(model_result_item, language)
            temp["score"] = 1
        except Exception as e:
            temp["score"] = 0
            temp["error"] = [f"Invalid syntax. Failed to decode AST. {str(e)}"]
            temp["error_type"] = "ast_decoder:decoder_failed"
            result.append(temp)
            continue

        decoder_output_valid = is_function_calling_format_output(model_result_item)
        if not decoder_output_valid:
            temp["score"] = 0
            temp["error"] = ["Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."]
            temp["error_type"] = "ast_decoder:decoder_wrong_output_format"
            result.append(temp)
            continue

        checker_result = ast_checker(
            prompt_item,
            model_result_item,
            possible_answer_item,
            language,
            test_category,
            model_name,
        )

        if checker_result["valid"]:
            correct_count += 1
            temp["score"] = 1
        else:
            temp["score"] = 0
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
        result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = test_category + "_score.json"
    output_file_dir = os.path.join(OUTPUT_PATH, model_name)
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


#### Main runner function ####
def runner(model_names, test_categories, api_sanity_check):
    
    # A flag to indicate if the API has been tested.
    # We should always test the API with ground truth first before running the executable tests. 
    # Sometimes the API may not be working as expected and we want to catch that before running the evaluation to ensure the results are accurate.
    API_TESTED = False
    
    # Get a list of all entries in the folder
    entries = os.scandir(INPUT_PATH)

    # Filter out the subdirectories
    subdirs = [entry.path for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in subdirs:

        model_name = subdir.split(INPUT_PATH)[1]
        if model_names is not None and model_name not in model_names:
            continue

        model_name_escaped = model_name.replace("_", "/")

        files = [
            f
            for f in os.listdir(subdir)
            if os.path.isfile(os.path.join(subdir, f)) and not f.startswith(".")
        ]  
        # Check if there is only one file and that file is 'result.json'
        # If so, this is an OSS model result file and we need to special process it first
        if len(files) == 1 and files[0] == "result.json":
            result_json_file_path = os.path.join(subdir, "result.json")
            oss_file_formatter(result_json_file_path, subdir)
            print(f"Detected OSS model: {model_name}. result.json has been split into individual test category files.")


        # Pattern to match JSON files in this subdirectory
        json_files_pattern = os.path.join(subdir, "*.json")
        
        print(f"🦍 Model: {model_name}")
        
        # Find and process all JSON files in the subdirectory
        for model_result_json in glob.glob(json_files_pattern):

            if os.path.basename(model_result_json) == "result.json":
                continue

            test_category = extract_after_test(model_result_json)
            if test_categories is not None and test_category not in test_categories:
                continue

            handler = get_handler(model_name_escaped)

            # We don't evaluate chatable and SQL models in our current leaderboard
            if is_chatable(test_category) or is_sql(test_category):
                continue

            language = "Python"
            if is_java(test_category):
                language = "Java"
            if is_js(test_category):
                language = "JavaScript"
            
            print(f"🔍 Running test: {test_category}")

            model_result = load_file(model_result_json)
            record_cost_latency(LEADERBOARD_TABLE, model_name, model_result)

            if is_relevance(test_category):
                accuracy, total_count = single_relevance_file_runner(
                    handler, model_result, model_name, test_category
                )
                record_result(
                    LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
                continue

            # Find the corresponding test file
            prompt_file = find_file_with_suffix(PROMPT_PATH, test_category)
            prompt = load_file(prompt_file)

            if is_executable(test_category):
                # We only test the API with ground truth once
                if not API_TESTED and api_sanity_check:
                    print("---- Sanity checking API status ----")
                    api_status_sanity_check()
                    print("---- Sanity check Passed 💯 ----")
                    API_TESTED = True
                    
                accuracy, total_count = single_executable_file_runner(
                    handler, model_result, prompt, model_name, test_category
                )
                record_result(
                    LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
                

                continue

            # Find the corresponding possible answer file
            possible_answer_file = find_file_with_suffix(
                POSSIBLE_ANSWER_PATH, test_category
            )
            possible_answer = load_file(possible_answer_file)
            accuracy, total_count = single_ast_file_runner(
                handler,
                model_result,
                prompt,
                possible_answer,
                language,
                test_category,
                model_name,
            )
            record_result(
                LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
            )
            print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")


    # This function reads all the score files from local folder and updates the leaderboard table.
    # This is helpful when you only want to run the evaluation for a subset of models and test categories.
    # update_leaderboard_table_with_score_file(LEADERBOARD_TABLE, OUTPUT_PATH)
    # Write the leaderboard table to a file
    # generate_leaderboard_csv(LEADERBOARD_TABLE, OUTPUT_PATH)


ARG_PARSE_MAPPING = {
    "ast": [
        "simple",
        "multiple_function",
        "parallel_function",
        "parallel_multiple_function",
        "java",
        "javascript",
        "relevance",
    ],
    "executable": [
        "executable_simple",
        "executable_multiple_function",
        "executable_parallel_function",
        "executable_parallel_multiple_function",
        "rest",
    ],
    "all": [
        "simple",
        "multiple_function",
        "parallel_function",
        "parallel_multiple_function",
        "java",
        "javascript",
        "relevance",
        "executable_simple",
        "executable_multiple_function",
        "executable_parallel_function",
        "executable_parallel_multiple_function",
        "rest",
    ],
    "non-python": [
        "java",
        "javascript",
    ],
    "python": [
        "simple",
        "multiple_function",
        "parallel_function",
        "parallel_multiple_function",
        "relevance",
        "executable_simple",
        "executable_multiple_function",
        "executable_parallel_function",
        "executable_parallel_multiple_function",
        "rest",
    ],
}


INPUT_PATH = "../result/" + "20240417_183023/"
PROMPT_PATH = "../data/"
POSSIBLE_ANSWER_PATH = "../data/possible_answer/"
OUTPUT_PATH = "../score/" + "20240417_183023/"

# A dictionary to store the results
# Key is model name, value is a dictionary with keys as test category and values as a dictionary with accuracy and total count
LEADERBOARD_TABLE = {}


if __name__ == "__main__":
    load_dotenv("../../.env")
    parser = argparse.ArgumentParser(description="Process two lists of strings.")

    # Add arguments for two lists of strings
    parser.add_argument(
        "--model", nargs="+", type=str, help="A list of model names to evaluate"
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        type=str,
        help="A list of test categories to run the evaluation on",
    )
    parser.add_argument(
        "--skip-api-sanity-check",
        action='store_false',  
        default=True,    # Default value is True, meaning the sanity check is performed unless the flag is specified
        help="Skip the REST API status sanity check before running the evaluation. By default, the sanity check is performed.",
    )

    args = parser.parse_args()

    model_names = args.model
    api_sanity_check = args.skip_api_sanity_check
    test_categories = None
    if args.test_category is not None:
        test_categories = []
        for test_category in args.test_category:
            if test_category in ARG_PARSE_MAPPING:
                test_categories.extend(ARG_PARSE_MAPPING[test_category])
            else:
                test_categories.append(test_category)

    runner(model_names, test_categories, api_sanity_check)

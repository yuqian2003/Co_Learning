import argparse
import logging
import time
import numpy as np
import pandas as pd
import requests
import json
import re
from tqdm import tqdm
from utils import SparkApi
import json
import time


def ERNIE(message: str) -> str:
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-8k-0329?access_token=" + Baidu_get_access_token()
    message = message
    payload = json.dumps({
        "messages": [
            {
            }
        ],
        "disable_search": False,
        "enable_citation": False
    })
    payload_dict = json.loads(payload)

    payload_dict["messages"] = message
    payload = json.dumps(payload_dict)
    headers = {
        'Content-Type': 'application/json'
    }
    res = requests.request("POST", url, headers=headers, data=payload).json()
    # print(self.history)
    return res['result']


def Baidu_get_access_token()-> str:
    """
    Generate authentication token (Access Token) using API Key and Secret Key.
    :return: access_token, or None (if error occurs)
    """
    API_KEY = "enter your api key"
    SECRET_KEY = "enter your secret key"
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def Llama(message: str) -> str:
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_70b?access_token=" + Llama_get_access_token()
    message = message
    payload = json.dumps({
        "messages": [
            {
            }
        ],
        "disable_search": False,
        "enable_citation": False
    })
    payload_dict = json.loads(payload)

    payload_dict["messages"] = message
    payload = json.dumps(payload_dict)
    headers = {
        'Content-Type': 'application/json'
    }
    res = requests.request("POST", url, headers=headers, data=payload).json()
    # print(self.history)
    print(res)
    return res['result']


def Llama_get_access_token() -> str:
    """
    Generate authentication token (Access Token) using API Key and Secret Key.
    :return: access_token, or None (if error occurs)
    """
    API_KEY = "enter your api key"
    SECRET_KEY = "enter your secret key"
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def Spark(task_msg: str) -> str:
    SparkApi.main(
        "your spark id",
        "your spark api key",
        "your spark api secret",
        "wss://spark-api.xf-yun.com/v3.5/chat",
        "generalv3.5",
        task_msg
    )
    return str(SparkApi.answer)


class CodeGenerationEnv:
    """
    Class representing the environment for code generation and testing.
    It supports interactions with multiple models like ERNIE, Llama, and Spark.
    """

    def __init__(self, model_name):
        super(CodeGenerationEnv, self).__init__()
        self.history = []
        self.history_all = []
        self.reward = 100
        if model_name == "ERNIE":
            self.LLM = ERNIE
        elif model_name == "Llama":
            self.LLM = Llama
        elif model_name == "Spark":
            self.LLM = Spark

    def extract_code_blocks(self, text: str) -> str:
        """
        Extracts code blocks from the provided text based on the model in use.
        """
        if self.LLM == Llama:
            code_blocks = re.findall(r'```((?:(?!assert).)*)```', text, re.DOTALL)
        elif self.LLM == Spark:
            code_blocks = re.findall(r'```python\s+(.*?)\s+```', text, re.DOTALL)
        else:
            code_blocks = re.findall(r'```python\n(.*?)\n```\n(?:(?!assert).)*', text, re.DOTALL)
        return "\n".join(code_blocks)

    def code_correct(self, error_code: str, test: list) -> str:
        """
        Corrects the provided error code based on the test cases.

        Parameters:
        error_code (str): The erroneous Python code that needs to be corrected.
        test (list): A list of test cases that the corrected code should pass.

        Returns:
        str: The corrected code.
        """
        msg = test[0]
        task_msg = "You are a Python programming expert. There is an error in the following code, please correct the code: " + error_code + ". Your corrected code should pass these tests: " + msg + ". Please only directly give the correct code with out any explanation after your modification and forget the assert in your answer. Please using English to do this job! Please do not renamed the function!"
        self.history.append({"role": "user", "content": task_msg})
        self.history_all.append({"role": "user", "content": task_msg})
        rm = self.LLM(self.history)
        self.history.append({"role": "assistant", "content": rm})
        self.history_all.append({"role": "assistant", "content": rm})
        if rm[0:3] == "def":
            code_correct = rm
        else:
            code_correct = self.extract_code_blocks(rm)
        return code_correct

    def explain_code(self) -> None:
        """
        Asks the model to provide a short explanation of the code.
        """
        prompt = "Please explain the code you provided. Please using English to do this job! You need keep the explanation short "

        self.history.append({"role": "user", "content": prompt})
        self.history_all.append({"role": "user", "content": prompt})
        rm = self.LLM(self.history)
        self.history.append({"role": "assistant", "content": rm})
        self.history_all.append({"role": "assistant", "content": rm})

    def change_code(self, error_assert: str) -> str:
        """
        Asks the model to rewrite the code based on errors encountered during testing.

        Parameters:
        error_assert (str): The error message or failed assertion encountered during testing.

        Returns:
        str: The revised code.
        """
        prompt = "The code you generated can not pass these test samples: " + str(
            error_assert) + ", please rewrite the code according to the code explanation. Please only directly give the  code with out any explanation and forget the assert in your answer. Please using English to do this job!"

        self.history.append({"role": "user", "content": prompt})
        self.history_all.append({"role": "user", "content": prompt})
        rm = self.LLM(self.history)

        self.history.append({"role": "assistant", "content": rm})
        self.history_all.append({"role": "assistant", "content": rm})
        if rm[0:3] == "def":
            code_change = rm
        else:
            code_change = self.extract_code_blocks(rm)
        return code_change

    def run_test(self, generated_code: str, test_list: list, challenge_test_list: list) -> tuple:
        """
        Runs the generated code against provided test cases and challenge cases.

        Parameters:
        generated_code (str): The code generated or corrected by the language model.
        test_list (list): A list of standard test cases that the code should pass.
        challenge_test_list (list): A list of additional or challenging test cases.

        Returns:
        tuple: A tuple containing a boolean indicating if all tests passed, a value representing the reward/penalty, and the error/assertion message (if any).
        """
        function_name_match = re.search(r'def (\w+)\(', generated_code)  
        function_name = function_name_match.group(1)

        test_passed = True
        error_assert = None
        value = 0
        local_namespace = {}
        try:
            exec(generated_code, globals(), local_namespace)
        except Exception as e:
            print(f"Error in code: {generated_code}, {e}")
            test_passed = False
            error_assert = str(e)
            value -= 1
            return test_passed, value, error_assert

        for test in test_list:
            test1 = test.replace("assert ", "")
            test1 = test1.replace('\"', "'")
            test_name1 = re.search(r'\((.*)\)', test1)
            test_name = test_name1.group(1)
            test1 = f"{function_name}({test_name})"

            try:
                if not eval(test1, globals(), local_namespace):
                    test_passed = False
                    error_assert = test
                    value -= 0.5
                    break
            except Exception as e:
                print(f"Error in test: {test}, {e}")
                value -= 0.5
                error_assert = e
                test_passed = False
                break
        if test_passed and challenge_test_list:
            for ch in challenge_test_list:
                ch1 = ch.replace("assert ", "")
                ch1 = ch1.replace('\"', "'")

                ch_name1 = re.search(r'\((.*)\)', ch1)  # Extract the challenge test parameters.
                ch_name = ch_name1.group(1)
               
                ch1 = f"{function_name}({ch_name})"  # Format the function call for testing.

                try:
                    if not eval(ch1, globals(), local_namespace):  # Evaluate the challenge test case.
                        test_passed = False
                        error_assert = ch
                        value -= 0.2
                        break
                except Exception as e:
                    print(f"Error in ch: {ch}, {e}")
                    value -= 0.2
                    test_passed = False
                    error_assert = e
                    break
        if test_passed: 
            value += 3 if challenge_test_list else 2  # Reward based on passing all tests.

        return test_passed, value, error_assert

    def step(self, error_code: str, test_list: list, change_test_list: list) -> tuple:
        """
        The main function that orchestrates the code correction process.

        Parameters:
        error_code (str): The erroneous Python code that needs to be corrected.
        test_list (list): A list of standard test cases that the code should pass.
        change_test_list (list): A list of additional or challenging test cases.

        Returns:
        tuple: A tuple containing the history of interactions, the final corrected code, the accumulated reward, the number of corrections made, and the total time taken.
        """
        start_time = time.time()
        num = 0  

        # Step 1 first debug

        final_generated_code = self.code_correct(error_code, test_list)
        for run in range(1, 6):  # Allow up to 5 attempts to correct the code.


            # Step 2: Run the test cases
            test_passed, value, error_assert = self.run_test(final_generated_code, test_list, change_test_list)

            # Step 3: If the code passes the test, break the loop
            if test_passed == True:
                self.reward += value * (1 / run)
                num = run
                break

            # Step 3 If the code fails, explain and debug again
            else:
                self.explain_code()
                # Step 4 Retry debugging
                final_generated_code = self.change_code(error_assert)
                self.reward += value * run
                num = run
                # return 
            # print(self.history)
            if run > 2:  # Trim the history to manage memory after multiple attempts.
                del self.history[2]
                del self.history[2]
                del self.history[2]
                del self.history[2]
        total_time = time.time() - start_time
        return self.history_all, final_generated_code, self.reward, num, total_time

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Co-Learning: Code Learning for Multi-Agent reinforcement Collaborative Framework with Conversational Natural Language Interfaces'
    )
parser.add_argument('--model_name', type=str, default='Spark',
                    help='The name of the model to use for code generation and debugging. Options: [ERNIE, Llama, Spark]')
parser.add_argument('--original_data', type=str, default="Code_702.csv",
                    help='The path to the CSV file containing the original data with error codes.')
args = parser.parse_args()

if __name__ == "__main__":
    this_time = time.strftime('%Y_%m_%d %H_%M_%S')
    log_filename = "Log/" + args.model_name + "/" + args.model_name + '_debug.log' + this_time
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Original Data file: {}".format(args.original_data))
    new_data = args.model_name + this_time + ".csv"
    logging.info("New Data file: {}".format(new_data))

    # df = pd.read_csv("Code_With_Error_NEW.csv")
    df = pd.read_csv("dataset/" + args.original_data)
    df["history_json"] = np.nan
    df["final_code"] = np.nan
    df["reward"] = np.nan
    df["time"] = np.nan
    df["The number of errors corrected"] = np.nan

    with open("history/" + args.model_name + "/" + this_time + "dialogue_history.txt",
              "a") as f:
        # try:
        for i in tqdm(range(0, len(df))):
            history, generated_code, reward, num, total_time = CodeGenerationEnv(args.model_name).step(
                df.loc[i, "Error_code"], eval(df.loc[i, "test_list"]), eval(df.loc[i, "challenge_test_list"]))
            df.loc[i, "history_json"] = str(history)
            df.loc[i, "final_code"] = str(generated_code)
            df.loc[i, "reward"] = str(reward)
            df.loc[i, "time"] = str(total_time)
            df.loc[i, "The number of errors corrected"] = str(num)


            for entry in history:
                f.write("{}: {}\n".format(entry["role"], entry["content"]))

            log_msg = "Iteration: {}, Final Code: {}, Reward: {}, Errors Corrected: {}, Time: {}".format(i,
                                                                                                         generated_code,
                                                                                                         reward,
                                                                                                         num,
                                                                                                         total_time)
            logging.info(log_msg)

            df.to_csv("output/" + args.model_name + "/" + new_data, index=False)
            print("Iteration {} saved".format(i))

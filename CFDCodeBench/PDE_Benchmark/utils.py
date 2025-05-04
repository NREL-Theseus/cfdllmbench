import json
import re
import subprocess
import boto3
from openai import OpenAI
import logging
from datetime import datetime
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import matplotlib.pyplot as plt
import ast
import shutil
# from google import genai
# from google.genai import types


def generate_prompt(data):
    parts = [
        "You are given the following partial differential equation (PDE) problem:\n",
        "**Equation:**\n" + data.get("equation", "") + "\n",
        "**Boundary Conditions:**\n" + data.get("boundary conditions", "") + "\n",
        "**Initial Conditions:**\n" + data.get("initial conditions", "") + "\n",
        "**Domain:**\n" + data.get("domain", "") + "\n",
        "**Numerical Method:**\n" + data.get("numerical method", "") + "\n"
    ]

    # Check for 'save_values' and add to task description
    save_values = data.get("save_values", [])
    save_values_str = ", ".join(save_values) if save_values else "the relevant variables specified for the problem"
    # Always end with task specification for the code
    parts.append(
        "### Task:\n"
        "- Write Python code to numerically solve the given CFD problem. Choose an appropriate numerical method based "
        "on the problem characteristics.\n"
        "- If the problem is **unsteady**, only compute and save the **solution at the final time step**.\n"
        "- For each specified variable, save the final solution as a separate `.npy` file using NumPy:\n"
        "  - For **1D problems**, save each variable as a 1D NumPy array.\n"
        "  - For **2D problems**, save each variable as a 2D NumPy array.\n"
        "- The `.npy` files should contain only the final solution field (not intermediate steps) for each of the "
        "specified variables.\n"
        "- **Save the following variables** at the final time step:\n"
        + save_values_str + "\n"
                            "(Each variable should be saved separately in its own `.npy` file, using the same name as "
                            "provided in `save_values`).\n"
                            "- Ensure the generated code properly handles the solution for each specified variable "
                            "and saves it correctly in `.npy` format.\n"
                            "- **Return only the complete, runnable Python code** that implements the above tasks, "
                            "ensuring no extra explanations or information is included."
    )

    return "\n".join(parts)


def generate_mms_prompt(data):
    parts = [
        "You are given the following partial differential equation (PDE) to test using the **Method of Manufactured "
        "Solutions (MMS)**:\n",
        "**Equation:**\n" + data.get("equation", "") + "\n",
        "**Domain:**\n" + data.get("domain", "") + "\n",
        "**Numerical Method:**\n" + data.get("numerical method", "") + "\n"
    ]

    # Check for 'save_values' and add to task description
    save_values = data.get("save_values", [])
    save_values_str = ", ".join(save_values) if save_values else "the relevant solution variables"

    parts.append(
        "### Task:\n"
        "- Choose a smooth manufactured solution \( u(x, t) \) that fits the domain.\n"
        "- Derive a source term \( f(x, t) \) such that the modified PDE:\n"
        "  \n    Original_PDE + f(x, t) = 0\n"
        "  \n  holds true for the chosen manufactured solution.\n"
        "- Implement a numerical solver using the specified method to solve the PDE with the source term.\n"
        "- At the final time step, compare the numerical result to the exact (manufactured) solution.\n"
        "- For each variable in the list below, save a `.npy` file named after the variable (e.g., `u.npy`, `p.npy`).\n"
        f"  - Variables to save: {save_values_str}\n"
        "- Each `.npy` file should contain a Python dictionary with two entries:\n"
        "  \n    `{'numerical': ..., 'MMS': ...}`\n"
        "  \n  where `'numerical'` is the computed solution and `'MMS'` is the exact solution.\n"
        "- Example: for variable `u`, save using `np.save('u.npy', {'numerical': u_numeric, 'MMS': u_exact})`\n"
        "- **Return only the complete, runnable Python code** that performs all steps: MMS setup, solver, comparison, "
        "and saving."
    )

    return "\n".join(parts)


class PromptGenerator:
    def __init__(self, root_dir, json_file):
        self.problems = None
        self.root_dir = root_dir
        self.input_json = os.path.join(root_dir, 'prompt/PDE_TASK_QUESTION_ONLY.json')
        self.output_json = os.path.join(root_dir, f'prompt/{json_file}')
        self.json_file = json_file

    def load_problem_data(self):
        with open(self.input_json, "r") as f:
            self.problems = json.load(f)

    def create_prompts(self):
        output_data = {}
        for name, data in self.problems.items():
            if self.json_file == "prompts.json":
                output_data[name] = generate_prompt(data)
            elif self.json_file == "mms_prompts.json":
                output_data[name] = generate_mms_prompt(data)
            else:
                raise ValueError(f"Unsupported prompt type: {self.json_file}")
        return {"prompts": output_data}  # <-- wrap in top-level "prompts" key

    def save_prompts(self, prompts):
        if os.path.exists(self.output_json):
            print(f"Skipped: {self.output_json} already exists.")
        else:
            with open(self.output_json, "w") as f:
                json.dump(prompts, f, indent=2)
            print(f"Created: {self.output_json}")


# === Function to Execute Python Script and Capture Errors and Warnings ===
def execute_python_script(filepath, timeout=60):
    try:
        result = subprocess.run(["python3", filepath], capture_output=True, text=True, timeout=timeout)
        stderr_output = result.stderr.strip()

        if result.returncode == 0:
            if "warning" in stderr_output.lower():
                logging.warning(f"Execution completed with warnings:\n{stderr_output}")
                return f"⚠️ Execution completed with warnings:\n{stderr_output}", result
            else:
                logging.info("Execution successful, no errors detected.")
                return "Execution successful, no errors detected.", result

        logging.error(f"Execution failed with errors:\n{stderr_output}")
        return stderr_output, result

    except Exception as e:
        logging.error(f"❌ Unexpected error while running script {filepath}: {e}")
        return f"❌ Unexpected error: {e}", None


def build_system_prompt():
    # Shared instruction - system instruction
    instruction = ("You are a highly skilled assistant capable of generating Python code to solve CFD problems "
                   "using appropriate numerical methods."
                   "Given the problem description, you should reason through the problem and determine the best "
                   "approach for discretizing and solving it,"
                   "while respecting the specified boundary conditions, initial conditions, and domain.\n"
                   "For unsteady problems, save only the solution at the final time step. For 1D problems, "
                   "save a 1D array; for 2D problems, save a 2D array.\n"
                   "Ensure the code follows the user's specifications and saves the requested variables exactly "
                   "as named in `save_values`.\n"
                   "Your task is to generate the correct, fully runnable Python code for solving the problem "
                   "without additional explanations.")
    return instruction


def build_conversation(original_prompt, llm_model):
    # Shared instruction - system instruction
    instruction = build_system_prompt()

    # Prompt augmentation
    user_prompt = (original_prompt +
                   "If it is an unsteady problem, only save the solution at the final time step "
                   "If the problem is 1D, the saved array should be 1D. "
                   "If the problem is 2D, the saved array should be 2D.")
    # Determine system or user role for the instruction
    if llm_model in ["o3-mini", "sonnet-35", "haiku"]:
        conversation_history = [
            {"role": "user", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    elif llm_model == "gpt-4o":
        conversation_history = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    elif llm_model == "gemini":
        conversation_history = user_prompt
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return conversation_history


def extract_model_response(llm_model, response):
    # Extract model response
    if llm_model in ["gpt-4o", "o3-mini"]:
        model_response = response.choices[0].message.content.strip()
    elif llm_model in ["sonnet-35", "haiku"]:
        # Parse the response
        response_body = json.loads(response["body"].read().decode("utf-8"))

        # Extract model response
        if "content" in response_body:
            model_response = response_body["content"][0]["text"]
        else:
            model_response = "Error: No response received from model."
    elif llm_model == "gemini":
        model_response = response.text
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return model_response


def update_token_usage(llm_model, response, tokens_counts):
    if llm_model == "gpt-4o":
        usage = response.usage
        tokens_counts['total_input_tokens'] += usage.prompt_tokens
        tokens_counts['total_output_tokens'] += usage.completion_tokens
        cost = (usage.prompt_tokens / 1000) * 0.005 + (usage.completion_tokens / 1000) * 0.015
        tokens_counts['total_cost'] += cost

    elif llm_model == "o3-mini":
        # OpenAI's o3-mini (same cost as gpt-3.5-turbo)
        usage = response.usage
        tokens_counts['total_input_tokens'] += usage.prompt_tokens
        tokens_counts['total_output_tokens'] += usage.completion_tokens
        cost = (usage.prompt_tokens + usage.completion_tokens) / 1000 * 0.0005
        tokens_counts['total_cost'] += cost

    elif llm_model == "sonnet-35":
        # Anthropic Sonnet-3.5 (via Bedrock response metadata)
        metadata = response.get('usage', {})
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        tokens_counts['total_input_tokens'] += input_tokens
        tokens_counts['total_output_tokens'] += output_tokens
        cost = (input_tokens / 1000) * 0.003 + (output_tokens / 1000) * 0.015
        tokens_counts['total_cost'] += cost

    elif llm_model == "haiku":
        # Anthropic Haiku
        metadata = response.get('usage', {})
        input_tokens = metadata.get('input_tokens', 0)
        output_tokens = metadata.get('output_tokens', 0)
        tokens_counts['total_input_tokens'] += input_tokens
        tokens_counts['total_output_tokens'] += output_tokens
        cost = (input_tokens / 1000) * 0.00025 + (output_tokens / 1000) * 0.00125
        tokens_counts['total_cost'] += cost

    elif llm_model == "gemini":
        # Gemini response has model_usage field
        metadata = response.usage_metadata
        input_tokens = metadata.prompt_token_count
        output_tokens = metadata.candidates_token_count
        tokens_counts['total_input_tokens'] += input_tokens
        tokens_counts['total_output_tokens'] += output_tokens
        cost = (input_tokens / 1000) * 0.00025 + (output_tokens / 1000) * 0.0005
        tokens_counts['total_cost'] += cost

    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    # Log summary
    logging.info(f"[{llm_model}] Input Tokens: {tokens_counts['total_input_tokens']}, "
                 f"Output Tokens: {tokens_counts['total_output_tokens']}, "
                 f"Estimated Cost: ${tokens_counts['total_cost']:.4f}")


def extract_code(model_response):
    # Match ```python ...``` or ```...``` code blocks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", model_response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Try parsing the whole thing as Python code
    try:
        ast.parse(model_response)
        return model_response.strip()
    except SyntaxError:
        pass

    # Extract "Python-like" lines
    lines = model_response.splitlines()
    python_lines = []
    for line in lines:
        if re.match(r'^\s*(import|from|def|class|for|if|while|print|#|@|[a-zA-Z_]+\s*=)', line):
            python_lines.append(line)

    if python_lines:
        return "\n".join(python_lines).strip()

    return "# No valid Python code extracted"


def save_model_outputs(task_name, output_folder, model_response):
    # Extract Python code:
    # start as ```python / ``` / or pure python code
    extracted_code = extract_code(model_response)

    # Save the full model response
    response_file = os.path.join(output_folder, f"{task_name}.txt")
    with open(response_file, "w") as txt_file:
        txt_file.write(model_response)

    # Save the extracted Python code
    script_path = os.path.join(output_folder, f"{task_name}.py")
    with open(script_path, "w") as py_file:
        py_file.write(extracted_code)

    print(f"✅ Code saved: {script_path}")

    return script_path


def execute_check_errors(llm_model, script_path, task_name, conversation_history):
    # Execute and check for errors
    execution_feedback, _ = execute_python_script(script_path)

    if "no errors detected" in execution_feedback:
        print(f"🎯 {task_name} executed successfully without syntax errors.")
        logging.info(f"🎯 {task_name} executed successfully without syntax errors.")
        return True  # Exit function if no errors

    else:
        print(f"❌ Error detected in {task_name}, refining prompt...")
        logging.info(f"❌ Error detected in {task_name}, refining prompt...")
        logging.info(
            f"\n\n[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it.")
        updated_prompt = f"[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it."

        # Add the refine prompt feedback to the conversation as input
        if llm_model == "gemini":
            conversation_history += updated_prompt
        else:
            conversation_history.append({"role": "user", "content": updated_prompt})


def call_llm_api(llm_model, client, conversation_history, temperature, bedrock_runtime, inference_profile_arn):
    if llm_model == "o3-mini":
        # Call OpenAI o3-mini API
        response = client.chat.completions.create(
            model=llm_model,  # Specify the model
            messages=conversation_history
        )
    elif llm_model == "gpt-4o":
        # Call OpenAI GPT-4o API
        response = client.chat.completions.create(
            model=llm_model,  # Specify the model
            messages=conversation_history,
            temperature=temperature
        )
    elif llm_model in ["sonnet-35", "haiku"]:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "temperature": temperature,
            "messages": conversation_history
        }

        # Invoke AWS Bedrock API
        response = bedrock_runtime.invoke_model(
            modelId=inference_profile_arn,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
    elif llm_model == "gemini":
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation_history,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=build_system_prompt()
            )
        )
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")
    return response


def generate_code(llm_model, task_name, prompt, client, temperature, bedrock_runtime, inference_profile_arn,
                  output_folder, tokens_counts, max_retries=5):
    retries = 0
    original_prompt = prompt  # Keep the original prompt unchanged
    # Initialize an empty list to store the conversation history
    conversation_history = build_conversation(original_prompt, llm_model)

    while retries < max_retries:
        print(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        logging.info(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        try:
            response = call_llm_api(llm_model, client, conversation_history, temperature, bedrock_runtime,
                                    inference_profile_arn)
            # log the input message
            logging.info(
                "---------------------------------INPUT TO LLM FIRST-----------------------------------------")
            logging.info(conversation_history)
            # log the LLM response
            logging.info(
                "------------------------------------LLM RESPONSE--------------------------------------------")
            logging.info(response)

            # Extract model response
            model_response = extract_model_response(llm_model, response)
            # Add the response to the conversation as input
            if llm_model == "gemini":
                conversation_history += model_response
            else:
                conversation_history.append({"role": "assistant", "content": model_response})
            logging.info(
                "---------------------------------INPUT TO LLM UPDATE----------------------------------------")
            logging.info(conversation_history)

            # tracking the token usage will return input and output tokens each round
            update_token_usage(llm_model, response, tokens_counts)

            # Extract Python code using regex, save the full model response and extracted python code
            script_path = save_model_outputs(task_name, output_folder, model_response)

            # Execute and check for errors
            if execute_check_errors(llm_model, script_path, task_name, conversation_history):
                return  # Exit function if no errors
            retries += 1

        except Exception as e:
            print(f"❌ API Call Error for {task_name}: {str(e)}")
            logging.info(f"❌ API Call Error for {task_name}: {str(e)}")
            return  # Stop retrying if API call fails

    print(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")
    logging.info(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")


def api_key_configuration(llm_model):
    # === OpenAI API Configuration ===
    if llm_model in ["gpt-4o", "o3-mini"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        client = OpenAI(api_key=api_key)
    elif llm_model == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY environment variable.")
        client = genai.Client(api_key=api_key)
    elif llm_model in ["sonnet-35", "haiku"]:
        api_key, client = None, None
    else:
        raise ValueError(f"Invalid {llm_model} api key")

    if llm_model == "sonnet-35":
        # Define Sonnet-3.5 profile Inference Profile ARN
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w"
    elif llm_model == "haiku":
        # Define Haiku profile Inference Profile ARN
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/56i8iq1vib3e"
    elif llm_model in ["o3-mini", "gpt-4o", "gemini"]:
        inference_profile_arn = None
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return api_key, client, inference_profile_arn


class LLMCodeGenerator:
    def __init__(self, llm_model, prompt_json, temperature=0.0, reviewer=True):
        self.llm_model = llm_model
        self.prompt_json = prompt_json
        self.temperature = temperature
        # Initialize AWS Bedrock client
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")
        # Get current time with microseconds
        self.timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds
        # === Paths ===
        # PDE_Benchmark root
        self.ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PDE_Benchmark')
        self.PROMPTS_FILE = os.path.join(self.ROOT_DIR, "prompt", prompt_json)
        self.OUTPUT_FOLDER = os.path.join(self.ROOT_DIR, f"solver/{llm_model}/{os.path.splitext(prompt_json)[0]}")
        self.REPORT_FOLDER = os.path.join(self.ROOT_DIR, 'report')
        self.LOG_FILE = os.path.join(self.REPORT_FOLDER, f"{llm_model}_{os.path.splitext(prompt_json)[0]}_"
                                                         f"{self.timestamp}.log")
        # Ensure the output directory exists
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.REPORT_FOLDER, exist_ok=True)

        # Tracking the total input and output tokens
        self.tokens_counts = {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0}
        if reviewer:
            self.max_retries = 5
        else:
            self.max_retries = 1

        logging.basicConfig(
            filename=self.LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def call_api(self):
        logging.info(
            "####################################################################################################")
        logging.info(f"Using the {self.llm_model}, change temperature to {self.temperature}, "
                     f"use the prompt {self.prompt_json}")
        # === Get API credentials, client, and profile ===
        api_key, client, inference_profile_arn = api_key_configuration(self.llm_model)
        # Load prompts
        with open(self.PROMPTS_FILE, "r") as file:
            pde_prompts = json.load(file)

        # Loop through prompts and generate code
        for task_name, prompt in pde_prompts["prompts"].items():
            # if task_name not in ["Fully_Developed_Turbulent_Channel_Flow"]:
            #     continue
            generate_code(
                self.llm_model,
                task_name,
                prompt,
                client,
                self.temperature,
                self.bedrock_runtime,
                inference_profile_arn,
                self.OUTPUT_FOLDER,
                self.tokens_counts,
                max_retries=self.max_retries
            )

        print("\n🎯 Execution completed. Check the solver directory for generated files.")
        logging.info("\n🎯 Execution completed. Check the solver directory for generated files.")
        logging.info(f"Total Input Tokens: {self.tokens_counts['total_input_tokens']}")
        logging.info(f"Total Output Tokens: {self.tokens_counts['total_output_tokens']}")
        logging.info(f"Total Estimated Cost: ${self.tokens_counts['total_cost']:.6f}")


def replacer_factory(base_name, save_dir):
    def replacer(match):
        var_name = match.group(3)  # u, v, p, etc.
        new_filename = f"{var_name}_{base_name}.npy"
        new_path = os.path.join(save_dir, new_filename).replace("\\", "/")  # Use Unix-style paths
        return f"np.save('{new_path}', {var_name})"

    return replacer


def call_post_process(generated_solvers_dir, save_dir):
    # === Paths ===
    folder_path = generated_solvers_dir
    os.makedirs(save_dir, exist_ok=True)

    # pattern = r"np\.save\((['\"])(.+?\.npy)\1\s*,\s*(\w+)\s*\)"
    # Match np.save("something", variable) — with or without .npy
    pattern = r"np\.save\((['\"])(.+?)\1\s*,\s*(\w+)\s*\)"

    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            py_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]  # e.g., "burgers_solver"

            with open(py_path, "r") as f:
                code = f.read()

            # Create a replacer for this specific file
            replacer = replacer_factory(base_name, save_dir)

            # Apply the replacement
            new_code, count = re.subn(pattern, replacer, code)

            if count > 0:
                with open(py_path, "w") as f:
                    f.write(new_code)
                print(f"✅ Updated {filename}: replaced {count} save path(s)")
            else:
                print(f"ℹ️ No np.save calls updated in {filename}")


def write_execute_results_to_log(log, script, result, status_counts):
    try:
        # Write execution results to log file
        log.write(f"--- Running: {script} ---\n")
        log.write(f"Exit Code: {result.returncode}\n")
        log.write("Output:\n")
        log.write(result.stdout + "\n")
        log.write("Errors:\n")
        log.write(result.stderr + "\n")
        log.write("-" * 50 + "\n\n")

        # Normalize case for easier checking
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()

        # Define pass condition: no "error" or "warning" in either output
        has_error_or_warning = (
            "error" in stdout_lower or
            "error" in stderr_lower or
            "warning" in stdout_lower or
            "warning" in stderr_lower
        )

        if result.returncode == 0 and not has_error_or_warning:
            print(f"✅ {script} executed successfully with clean output.")
            status_counts['pass'] += 1
        else:
            print(f"❌ {script} has warnings or errors (check logs).")
            status_counts['fail'] += 1

    except Exception as e:
        log.write(f"Exception occurred while logging execution: {e}\n")
        print(f"❌ Exception while logging {script}: {e}")
        status_counts['fail'] += 1


def write_execute_error_to_log(log, script, status_counts):
    try:
        log.write(f"--- Running: {script} ---\n")
        log.write("⚠️ Timeout Error: Script took too long to execute.\n")
        log.write("-" * 50 + "\n\n")
        print(f"⚠️ Timeout: {script} took too long and was skipped.")
        status_counts['fail'] += 1  # Increment fail count for timeout
    except Exception as e:
        log.write(f" {e}\n")


def write_execute_summary_to_log(log, status_counts):
    try:
        # Log the summary of pass and fail counts
        log.write("\n\n====== Execution Summary ======\n")
        log.write(f"Total Scripts Passed: {status_counts['pass']}\n")
        log.write(f"Total Scripts Failed: {status_counts['fail']}\n")
    except Exception as e:
        log.write(f" {e}\n")


def open_log_save_execution_results(log_file, python_files, generate_solvers_dir, status_counts):
    try:
        with open(log_file, "w") as log:
            log.write("====== Execution Results for Generated Solvers ======\n\n")

            for script in python_files:
                script_path = os.path.join(generate_solvers_dir, script)
                print(f"🔹 Running: {script} ...")

                # Run the script and capture the output
                try:
                    _, result = execute_python_script(script_path)

                    # Write execution results to log file
                    write_execute_results_to_log(log, script, result, status_counts)

                except subprocess.TimeoutExpired:
                    write_execute_error_to_log(log, script, status_counts)

                # Log the summary of pass and fail counts
                write_execute_summary_to_log(log, status_counts)

        print(f"\n🎯 Execution completed. Results saved in: {log_file}")
    except Exception as e:
        log.write(f" {e}\n")


def call_execute_solver(generated_solvers_dir, log_file, status_counts):
    try:
        # Define the directory where generated solver scripts are stored
        GENERATED_SOLVERS_DIR = generated_solvers_dir
        # Define the log file for execution results
        LOG_FILE = log_file

        # Ensure the output directory exists
        os.makedirs(GENERATED_SOLVERS_DIR, exist_ok=True)

        # Get all Python files in the solvers directory
        python_files = [f for f in os.listdir(GENERATED_SOLVERS_DIR) if f.endswith(".py")]

        # Ensure there are solver scripts to run
        if not python_files:
            print("No Python solver scripts found in the directory.")
            exit()
        # Initialize counters for pass and fail
        # Open a log file to save execution results
        open_log_save_execution_results(LOG_FILE, python_files, GENERATED_SOLVERS_DIR, status_counts)
    except Exception as e:
        print(f" {e}\n")


def interpolate_to_match(gt, pred):
    if gt.shape == pred.shape:
        return pred
    try:
        factors = np.array(gt.shape) / np.array(pred.shape)
        pred_resized = zoom(pred, factors, order=1)
        return pred_resized
    except Exception as e:
        print(f"RuntimeError: Interpolation failed: {e}")


def compute_losses(gt, pred):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    mse = mean_squared_error(gt_flat, pred_flat)
    mae = mean_absolute_error(gt_flat, pred_flat)
    rmse = np.sqrt(mse)
    cosine_sim = cosine_similarity(gt_flat.reshape(1, -1), pred_flat.reshape(1, -1))[0][0]
    r2 = r2_score(gt_flat, pred_flat)

    # NMSE: Normalized Mean Squared Error
    nmse = mse / np.mean(gt_flat**2)

    return mse, mae, rmse, cosine_sim, r2, nmse


def print_summary(results):
    print("\n=== Summary ===")
    for fname, res in results.items():
        print(f"📄 {fname}")
        for k, v in res.items():
            print(f"   {k}: {v}")
        print("-" * 40)


def compute_errors_gt_pred(common_files, ground_truth_dir, prediction_dir, results):
    for fname in common_files:
        try:
            gt_path = os.path.join(ground_truth_dir, fname)
            pred_path = os.path.join(prediction_dir, fname)
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            if gt.ndim == 1:
                gt = gt[:, np.newaxis]
            if pred.ndim == 1:
                pred = pred[:, np.newaxis]

            pred = interpolate_to_match(gt, pred)

            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch after interpolation: {gt.shape} vs {pred.shape}")

            mse, mae, rmse, cosine_sim, r2, nmse = compute_losses(gt, pred)

            results[fname] = {
                "MSE": f"{mse:.3e}",
                "MAE": f"{mae:.3e}",
                "RMSE": f"{rmse:.3e}",
                "CosineSimilarity": f"{cosine_sim:.3f}",
                "R2": f"{r2:.3f}",
                "NMSE": f"{nmse:.3f}"
            }

            logging.info(
                f"{fname}: MSE={mse:.3e}, MAE={mae:.3e}, RMSE={rmse:.3e}, Cosine={cosine_sim:.3f}, R2={r2:.3f}, "
                f"NMSE={nmse:.3f}")

        except Exception as e:
            results[fname] = {"Error": str(e)}
            logging.error(f"❌ {fname} failed: {str(e)}")


def get_common_files(ground_truth_dir, prediction_dir):
    files_gt = {f for f in os.listdir(ground_truth_dir) if f.endswith('.npy')}
    files_pred = {f for f in os.listdir(prediction_dir) if f.endswith('.npy')}
    common_files = sorted(files_gt & files_pred)

    print(f"Found {len(common_files)} common files to compare.")
    logging.info(f"Found {len(common_files)} common files.")

    return common_files


def call_compare_output_mismatch(ground_truth_dir, prediction_dir, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Clear old logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Setup new log file
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("====== Starting Comparison ======")

    # === Get common .npy files ===
    common_files = get_common_files(ground_truth_dir, prediction_dir)

    results = {}

    compute_errors_gt_pred(common_files, ground_truth_dir, prediction_dir, results)

    # === Print Summary ===
    print_summary(results)

    print(f"\n🎯 Log saved to: {log_file}")


def get_problem_name_pred(filename):
    return filename.rsplit('_', 1)[0]


def get_problem_name_gt(filename):
    return filename.rsplit('_', 2)[0]


def call_compare_image_mismatch(save_dir_gt, save_dir_pred, save_csv_path):
    # === Configuration ===
    os.makedirs(save_csv_path, exist_ok=True)

    # === Find common files ===
    gt_files = {get_problem_name_gt(f): f for f in os.listdir(save_dir_gt) if f.endswith('.png')}
    pred_files = {get_problem_name_pred(f): f for f in os.listdir(save_dir_pred) if f.endswith('.png')}

    common_files = gt_files.keys() & pred_files.keys()

    results = []

    for filename in common_files:
        gt_path = os.path.join(save_dir_gt, gt_files[filename])
        pred_path = os.path.join(save_dir_pred, pred_files[filename])

        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        img_pred_resized = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        abs_error = np.abs(img_gt - img_pred_resized)
        mse_val = np.mean((img_gt - img_pred_resized) ** 2)
        mae_val = np.mean(abs_error)
        ssim_val = ssim(img_gt, img_pred_resized, data_range=1.0)
        psnr_val = psnr(img_gt, img_pred_resized, data_range=1.0)

        results.append({
            "filename": filename,
            "MSE": mse_val,
            "MAE": mae_val,
            "SSIM": ssim_val,
            "PSNR": psnr_val,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    csv_file = os.path.join(save_csv_path, 'image_similarity_scores.csv')
    df.to_csv(csv_file, index=False, float_format="%.3e")


def call_create_table(compare_results_log_file, save_table_path):
    # === Config ===
    log_file_path = compare_results_log_file
    output_csv_path = save_table_path

    # === Regex pattern for results line ===
    pattern = re.compile(
        r"INFO - ([\w_.\-]+): MSE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"MAE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"RMSE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"Cosine=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"R2=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    # === Extract Data ===
    results = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                results.append({
                    'Filename': match.group(1),
                    'MSE': float(match.group(2)),
                    'MAE': float(match.group(3)),
                    'RMSE': float(match.group(4)),
                    'Cosine Similarity': float(match.group(5)),
                    'R-squared': float(match.group(6)),
                })

    # === Create DataFrame ===
    df = pd.DataFrame(results)

    # Format all float columns to scientific notation with 3 significant digits
    for col in ['MSE', 'MAE', 'RMSE', 'Cosine Similarity', 'R-squared']:
        df[col] = df[col].apply(lambda x: f"{x:.3e}")

    # === Save to CSV ===
    df.to_csv(output_csv_path, index=False)

    # === Display all rows/columns ===
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("✅ Table saved to:", output_csv_path)
        print(df)


def plot_1d(gt, pred, file_name, save_dir):
    x_gt = np.arange(len(gt))
    x_pred = np.arange(len(pred))

    # Get shared y-axis range
    ymin = min(gt.min(), pred.min())
    ymax = max(gt.max(), pred.max())

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"{file_name}")
    plt.subplot(2, 1, 1)
    plt.plot(x_gt, gt, label='Ground Truth', color='blue')
    plt.ylim([ymin, ymax])
    plt.title('Ground Truth')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_pred, pred, label='Prediction', color='green')
    plt.ylim([ymin, ymax])
    plt.title('Prediction')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
    plt.close()


def plot_2d(gt, pred, file_name, save_dir):
    # Get shared colorbar range
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.suptitle(f"{file_name}")
    im0 = axes[0].imshow(gt, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_plot.png"))
    plt.close()


def plot_1d_diff(gt, pred, file_name, save_dir_gt, save_dir_pred):
    x_gt = np.arange(len(gt))
    x_pred = np.arange(len(pred))

    # Get shared y-axis range
    ymin = min(gt.min(), pred.min())
    ymax = max(gt.max(), pred.max())

    # Plot the Ground Truth and save to separate directory
    plt.figure(figsize=(10, 6))
    plt.plot(x_gt, gt, label='Ground Truth', color='blue')
    plt.ylim([ymin, ymax])
    plt.title(f'Ground Truth - {file_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_gt, f"{file_name}_ground_truth.png"))
    plt.close()

    # Plot the Prediction and save to separate directory
    plt.figure(figsize=(10, 6))
    plt.plot(x_pred, pred, label='Prediction', color='green')
    plt.ylim([ymin, ymax])
    plt.title(f'Prediction - {file_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_pred, f"{file_name}_prediction.png"))
    plt.close()


def plot_2d_diff(gt, pred, file_name, save_dir_gt, save_dir_pred):
    # Get shared colorbar range
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    # Plot the Ground Truth and save to separate directory
    plt.figure(figsize=(10, 6))
    im0 = plt.imshow(gt, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Ground Truth - {file_name}')
    plt.colorbar(im0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_gt, f"{file_name}_ground_truth.png"))
    plt.close()

    # Plot the Prediction and save to separate directory
    plt.figure(figsize=(10, 6))
    im1 = plt.imshow(pred, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Prediction - {file_name}')
    plt.colorbar(im1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_pred, f"{file_name}_prediction.png"))
    plt.close()


def call_save_image_same_dir(save_dir, ground_truth_dir, prediction_dir):
    # === Configuration ===
    os.makedirs(save_dir, exist_ok=True)
    # === Common files (files that exist in both directories) ===
    common_files = get_common_files(ground_truth_dir, prediction_dir)

    # === Iterate and Plot for Common Files ===
    for file in common_files:
        gt_path = os.path.join(ground_truth_dir, file)
        pred_path = os.path.join(prediction_dir, file)

        try:
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            # Plot
            if gt.ndim == 1 or (gt.ndim == 2 and 1 in gt.shape):
                plot_1d(gt.flatten(), pred.flatten(), file.replace(".npy", ""), save_dir)
            elif gt.ndim == 2:
                plot_2d(gt, pred, file.replace(".npy", ""), save_dir)
            else:
                print(f"❌ Skipping unsupported shape for file: {file} → {gt.shape}")
        except Exception as e:
            print(f"❌ Error plotting {file}: {str(e)}")

    print(f"\n🎯 Plotting complete. Images saved to: {save_dir}")


def call_save_image_different_dir(ground_truth_dir, prediction_dir, save_dir_gt, save_dir_pred):
    # === Configuration ===
    os.makedirs(save_dir_gt, exist_ok=True)
    os.makedirs(save_dir_pred, exist_ok=True)

    # === Common files (files that exist in both directories) ===
    common_files = get_common_files(ground_truth_dir, prediction_dir)

    # === Iterate and Plot for Common Files ===
    for file in common_files:
        gt_path = os.path.join(ground_truth_dir, file)
        pred_path = os.path.join(prediction_dir, file)

        try:
            gt = np.load(gt_path)
            pred = np.load(pred_path)

            # Plot
            if gt.ndim == 1 or (gt.ndim == 2 and 1 in gt.shape):
                plot_1d_diff(gt.flatten(), pred.flatten(), file.replace(".npy", ""), save_dir_gt, save_dir_pred)
            elif gt.ndim == 2:
                plot_2d_diff(gt, pred, file.replace(".npy", ""), save_dir_gt, save_dir_pred)
            else:
                print(f"❌ Skipping unsupported shape for file: {file} → {gt.shape}")
        except Exception as e:
            print(f"❌ Error plotting {file}: {str(e)}")

    print(f"\n🎯 Plotting complete. Images saved to: {save_dir_gt} and {save_dir_pred}")

def scale_nx_ny_nt(folder_path, k):
    print("Scanning folder:", os.path.abspath(folder_path))

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                print(f"Processing {file_path}")

                with open(file_path, 'r') as f:
                    code = f.read()

                updated = False  # flag to indicate whether we changed anything

                # Update nx
                nx_match = re.search(r'nx\s*=\s*(\d+)', code)
                if nx_match:
                    nx_old = int(nx_match.group(1))
                    nx_new = nx_old * k
                    code = re.sub(r'nx\s*=\s*\d+', f'nx = {nx_new}', code)
                    print(f"  [Updated] nx: {nx_old} → {nx_new}")
                    updated = True

                # Update ny
                ny_match = re.search(r'ny\s*=\s*(\d+)', code)
                if ny_match:
                    ny_old = int(ny_match.group(1))
                    ny_new = ny_old * k
                    code = re.sub(r'ny\s*=\s*\d+', f'ny = {ny_new}', code)
                    print(f"  [Updated] ny: {ny_old} → {ny_new}")
                    updated = True

                # Update nt
                nt_match = re.search(r'nt\s*=\s*(\d+)', code)
                if nt_match:
                    nt_old = int(nt_match.group(1))
                    nt_new = nt_old * k
                    code = re.sub(r'nt\s*=\s*\d+', f'nt = {nt_new}', code)
                    print(f"  [Updated] nt: {nt_old} → {nt_new}")
                    updated = True

                # Write back only if we made changes
                if updated:
                    with open(file_path, 'w') as f:
                        f.write(code)


class SolverPostProcessor:
    def __init__(self, llm_model, prompt_json):
        self.llm_model = llm_model
        self.prompt_json = prompt_json
        self.timestamp = datetime.now().strftime("%H-%M-%S-%f")
        # PDE_Benchmark root
        self.root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PDE_Benchmark')
        self.prompt_name = os.path.splitext(prompt_json)[0]
        self.generated_solvers_dir = os.path.join(self.root_dir, f"solver/{llm_model}/{self.prompt_name}")
        self.generated_solvers_save_dir = os.path.join(self.root_dir, "report")
        self.log_file = os.path.join(self.generated_solvers_save_dir,
                                     f"execution_results_{llm_model}_{self.prompt_name}_{self.timestamp}.log")
        self.save_dir = os.path.join(self.root_dir, f'results/prediction/{llm_model}/{self.prompt_name}')
        self.ground_truth_dir = os.path.join(self.root_dir, 'results/solution')
        self.prediction_dir = os.path.join(self.root_dir, f'results/prediction/{llm_model}/{self.prompt_name}')
        self.compare_results_log_file = os.path.join(self.root_dir,
                                                     f'compare/comparison_results_{llm_model}_{self.prompt_name}.log')
        self.save_dir_gt = os.path.join(self.root_dir, f'compare_images/ground_truth/{llm_model}/{self.prompt_name}')
        self.save_dir_pred = os.path.join(self.root_dir, f'compare_images/prediction/{llm_model}/{self.prompt_name}')
        self.save_csv_path = os.path.join(self.root_dir, f'compare_images/table/{llm_model}/{self.prompt_name}')
        self.TABLE_FOLDER = os.path.join(self.root_dir, 'table')
        self.IMAGE_FOLDER = os.path.join(self.root_dir, 'image')
        self.COMPARE_IMAGE_FOLDER = os.path.join(self.root_dir, 'compare_images')
        self.SOLVER_FOLDER = os.path.join(self.root_dir, 'solver')
        self.save_table_path = os.path.join(self.root_dir,
                                            f'table/{llm_model}_{self.prompt_name}_extracted_results_table_{self.timestamp}.csv')
        self.save_image_dir = os.path.join(self.root_dir, f'image/{llm_model}/{self.prompt_name}')

        os.makedirs(self.TABLE_FOLDER, exist_ok=True)
        os.makedirs(self.IMAGE_FOLDER, exist_ok=True)
        os.makedirs(self.COMPARE_IMAGE_FOLDER, exist_ok=True)
        os.makedirs(self.SOLVER_FOLDER, exist_ok=True)

        self.status_counts = {"pass": 0, "fail": 0}
        self.CONVERGENT_FOLDER = os.path.join(self.root_dir, 'convergent')
        os.makedirs(self.CONVERGENT_FOLDER, exist_ok=True)


    def run_all(self, step1=True, step2=True, step3=True, step4=True):
        if step1:
            # STEP 1:
            # post process to change the .npy save path to specific path
            call_post_process(self.generated_solvers_dir, self.save_dir)
        if step2:
            # STEP 2:
            # execute LLM generated python code and save the results to log file
            call_execute_solver(self.generated_solvers_dir, self.log_file, self.status_counts)
        if step3:
            # STEP 3:
            # compute the numerical errors (MSE, MAE, RMSE, CosineSimilarity, R2) for shape mismatch / match array
            call_compare_output_mismatch(self.ground_truth_dir, self.prediction_dir, self.compare_results_log_file)
            # transfer the numerical errors to tables
            call_create_table(self.compare_results_log_file, self.save_table_path)
        if step4:
            # STEP 4:
            # plot and save the images of gt and pred in different folder
            call_save_image_different_dir(self.ground_truth_dir, self.prediction_dir, self.save_dir_gt,
                                          self.save_dir_pred)
            # compare the images for mismatch shape, change the image to gray image, note the MSE is for pixel not
            # the same with MSE of original images, it works like human eyes (this function is optional)
            call_compare_image_mismatch(self.save_dir_gt, self.save_dir_pred, self.save_csv_path)
            # save the gt and pred images in the same figure, each is sub-figure, this used for human view the images
            # this function is optional
            call_save_image_same_dir(self.save_image_dir, self.ground_truth_dir, self.prediction_dir)


class ConvergentTest(SolverPostProcessor):
    def __init__(self, llm_model, prompt_json):
        super().__init__(llm_model, prompt_json)

        # Override relevant paths to use the convergent folder
        self.generated_solvers_dir = os.path.join(self.root_dir, f"convergent/{llm_model}/{self.prompt_name}")  # instead of solver/{model}/{prompt}
        self.prediction_dir = os.path.join(self.root_dir, f'results/prediction_convergent/{llm_model}/{self.prompt_name}')
        self.save_dir = self.prediction_dir  # where .npy results are saved

        self.log_file = os.path.join(self.generated_solvers_save_dir,
                                     f"execution_results_convergent_{llm_model}_{self.prompt_name}_{self.timestamp}.log")

        self.compare_results_log_file = os.path.join(self.root_dir,
                                                     f'compare/comparison_results_convergent_{llm_model}_{self.prompt_name}.log')

        self.save_dir_gt = os.path.join(self.root_dir, f'compare_images/ground_truth_convergent/{llm_model}/{self.prompt_name}')
        self.save_dir_pred = os.path.join(self.root_dir, f'compare_images/prediction_convergent/{llm_model}/{self.prompt_name}')
        self.save_csv_path = os.path.join(self.root_dir, f'compare_images/table_convergent/{llm_model}/{self.prompt_name}')

        self.save_table_path = os.path.join(self.root_dir,
                                            f'table/convergent_{llm_model}_{self.prompt_name}_extracted_results_table_{self.timestamp}.csv')
        self.save_image_dir = os.path.join(self.root_dir, f'image/convergent/{llm_model}/{self.prompt_name}')

        # Make sure any new directories are created
        os.makedirs(self.prediction_dir, exist_ok=True)
        os.makedirs(self.save_dir_gt, exist_ok=True)
        os.makedirs(self.save_dir_pred, exist_ok=True)
        os.makedirs(os.path.dirname(self.save_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.save_table_path), exist_ok=True)
        os.makedirs(self.save_image_dir, exist_ok=True)

    def run(self):
        # Post-processing for convergent test
        shutil.copytree(self.SOLVER_FOLDER, self.CONVERGENT_FOLDER, dirs_exist_ok=True)
        scale_nx_ny_nt(self.CONVERGENT_FOLDER, 4)
        call_post_process(self.generated_solvers_dir, self.save_dir)
        call_execute_solver(self.generated_solvers_dir, self.log_file, self.status_counts)
        call_compare_output_mismatch(self.ground_truth_dir, self.prediction_dir, self.compare_results_log_file)
        # transfer the numerical errors to tables
        call_create_table(self.compare_results_log_file, self.save_table_path)



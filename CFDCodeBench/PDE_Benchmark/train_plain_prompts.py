from utils import PromptGenerator, LLMCodeGenerator, SolverPostProcessor, ConvergentTest
import os

# STEP 1: generate the prompts
# Set the root directory (you can also use os.getcwd() if running from root)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(ROOT_DIR, 'PDE_Benchmark')  # PDE_Benchmark root
json_file = "prompts.json"  # do not use mms
# json_file = "mms_prompts.json"  # use mms
# Initialize the PromptGenerator
generator_prompt = PromptGenerator(root_dir, json_file)

# Load the problem data from the input JSON
generator_prompt.load_problem_data()

# Create prompts from the data
prompts = generator_prompt.create_prompts()

# Save to prompts.json (will skip if already exists)
generator_prompt.save_prompts(prompts)

# STEP 2: call LLM API to get generated code
# Set the model and prompt JSON filename
# llm_model = "gemini"  # "gpt-4o", ""o3-mini, "sonnet-3.5", "haiku", "lama 4", "gemini: gemini-2.0-flash"
# List of LLM models to evaluate
llm_models = [
    "gpt-4o",
    "o3-mini",
    "gemini",
    "sonnet-35",
    "haiku",
]

# use gpt-4o to check the code
# llm_models = ["o3-mini"]
prompt_json = json_file  # the file under ./prompt/
# Loop over all models
for llm_model in llm_models:
    print(f"\n=== Running for model: {llm_model} ===\n")
    print(f"\n=== Running for prompt: {json_file} ===")
    # # Instantiate the class
    # generator_llm = LLMCodeGenerator(llm_model, prompt_json)
    #
    # # Call the API to generate code for each task
    # generator_llm.call_api()

    # STEP 3: post-process the generate code and compare the loss and images
    # Create the post-processor
    processor = SolverPostProcessor(llm_model, prompt_json)

    # Run the full post-processing pipeline
    # this time only run execute LLM generated python code and save the results to log file
    # processor.run_all()

    convergent_processor = ConvergentTest(llm_model, prompt_json)
    convergent_processor.run()

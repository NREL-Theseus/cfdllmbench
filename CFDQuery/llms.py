import os
import json
import re
import time
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import OllamaLLM

# —— Metadata Information ——
CURRENT_UTC = "2025-05-03 15:18:10"
CURRENT_USER = "Xingyu Xie"

# —— 1. Model Response Validation —— 
class ModelResponse(BaseModel):
    answer: int = Field(..., description="The answer number (1-4)")
    raw_response: str = Field(..., description="The raw text response from the model")

    @classmethod
    def strict_parse(cls, raw: str) -> int:
        """when raw is 1-4, pass"""
        raw_strip = raw.strip()
        if raw_strip in {"1","2","3","4"}:
            return int(raw_strip)
        raise ValueError(f"Strict parse failed for '{raw}'")

    @classmethod
    def tolerant_parse(cls, raw: str) -> int:
        """get first number from 1-4"""
        match = re.search(r'([1-4])', raw)
        if match:
            return int(match.group(1))
        raise ValueError(f"Tolerant parse failed for '{raw}'")

    @classmethod
    def parse_with_retry(cls, raw: str) -> int:
        """
        try strict parse first, if fail, try tolerant parse
        """
        try:
            return cls.strict_parse(raw)
        except ValueError:
            return cls.tolerant_parse(raw)

# —— 2. Initialize Clients —— 
openai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client   = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai_client    = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
ollama_client   = OllamaLLM(model="llama3.2", temperature=0.0)
gemma_client    = OllamaLLM(model="gemma2:9b", temperature=0.0)

# —— 3. System Prompt  —— 
SYSTEM_PROMPT = (
    "You are an expert computational fluid dynamics researcher.\n"
    "For each multiple-choice question, read the question and its four options,\n"
    "then respond with only the number (1, 2, 3, or 4) corresponding to the correct answer."
)

# —— 4. Load Questions  —— 
def load_questions(file_path='CFDQuery.json'):
    """Load questions from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'CFD QA' not in data:
                raise KeyError("Expected 'CFD QA' key in JSON data")
            return data['CFD QA']
    except FileNotFoundError:
        raise FileNotFoundError(f"Question file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")
    except KeyError as e:
        raise KeyError(f"Missing required key in JSON data: {str(e)}")

# —— 5. Main Evaluation Process —— 
def run_evaluation(qa_data):
    model_configs = {
        'o3-mini-2025-01-31':           ('openai', {'temperature': None}),
        'gpt-4o-2024-11-20':            ('openai', {'temperature': 0.0}),
        'claude-3-5-sonnet-20241022': ('claude', {'model': 'claude-3-5-sonnet-20241022', 'temperature': 0.0}),
        'claude-3-5-haiku-20241022':  ('claude', {'model': 'claude-3-5-haiku-20241022', 'temperature': 0.0}),
        'gemini-2.0-flash-001':  ('genai',  {'model': 'gemini-2.0-flash-001', 'temperature': 0.0}),
        'llama3.2:3b':          ('ollama', {'model': 'llama3.2:3b', 'temperature': 0.0}),
        'gemma2:9b':           ('ollama', {'model': 'gemma2:9b', 'temperature': 0.0}),
    }

    results = {name: {'correct': 0, 'responses': []} for name in model_configs}
    total = len(qa_data)

    for item in qa_data:
        q_idx = item['question_index']
        correct_answer = item['correct_option_index']

        user_prompt = f"Question {q_idx}: {item['question_content']}\n"
        for opt in item['options']:
            user_prompt += f"{opt['option_index']}. {opt['option_content']}\n"
        user_prompt += "\nReply with only the option number."

        for name, (vendor, cfg) in model_configs.items():
            attempt = 0
            final_answer = 0
            raw_responses = []
            parsing_failed = False
            
            while attempt < 3:
                attempt += 1
                try:
                    if vendor == 'openai':
                        call = {
                            "model": name,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                        }
                        if cfg.get('temperature') is not None:
                            call["temperature"] = cfg['temperature']
                        resp = openai_client.chat.completions.create(**call)
                        raw = resp.choices[0].message.content.strip()

                    elif vendor == 'claude':
                        combined = SYSTEM_PROMPT + "\n\n" + user_prompt
                        msg = claude_client.messages.create(
                            model=cfg['model'],
                            temperature=cfg['temperature'],
                            max_tokens=10,
                            messages=[{"role": "user", "content": combined}],
                        )
                        raw = msg.content[0].text.strip()

                    elif vendor == 'genai':
                        gen_resp = genai_client.models.generate_content(
                            model=cfg['model'],
                            contents=user_prompt,
                            config=genai.types.GenerateContentConfig(
                                temperature=cfg['temperature'],
                                maxOutputTokens=10
                            )
                        )
                        raw = gen_resp.text.strip()

                    else:  # ollama
                        ollama_prompt = (
                            f"Analyze this CFD multiple-choice question and provide ONLY the correct option number (1-4):\n{item['question_content']}\n"
                        )
                        for opt in item['options']:
                            ollama_prompt += f"{opt['option_index']}. {opt['option_content']}\n"
                        ollama_prompt += "\nRespond with just the number:"
                        
                        if 'gemma' in name:
                            raw = gemma_client.invoke(ollama_prompt).strip()
                        else:
                            raw = ollama_client.invoke(ollama_prompt).strip()
                        time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    raw = f"[Error: {e}]"
                    parsing_failed = True

                raw_responses.append(raw)
                print(f"[{name} Q{q_idx} Attempt {attempt}] raw_response: {raw}")

                try:
                    parsed = ModelResponse.parse_with_retry(raw)
                    final_answer = parsed
                    print(f"[{name} Q{q_idx} Attempt {attempt}] parsed answer: {final_answer}")
                    if not parsing_failed:
                        break
                    else:
                        parsing_failed = False
                except Exception as parse_err:
                    print(f"[{name} Q{q_idx} Attempt {attempt}] parse failed: {parse_err}")
                    parsing_failed = True
                    if attempt < 3:
                        print(f"[{name} Q{q_idx}] retrying...\n")
                    else:
                        print(f"[{name} Q{q_idx}] all 3 attempts failed, marking as incorrect.\n")
                        final_answer = 0

            is_correct = (final_answer == correct_answer and not parsing_failed)
            if is_correct:
                results[name]['correct'] += 1

            results[name]['responses'].append({
                'question_idx': q_idx,
                'model_answer': final_answer,
                'correct_answer': correct_answer,
                'raw_responses': raw_responses,
                'is_correct': is_correct,
                'parsing_failed': parsing_failed
            })

    for name, data in results.items():
        acc = data['correct'] / total * 100
        print(f"{name:<25} Accuracy: {acc:6.2f}%")

    return results

# —— 6. Save Results —— 
def save_results(results, filename='model_evaluation_results.json'):
    """Save overall evaluation results to JSON file"""
    try:
        results_data = {
            'metadata': {
                'evaluation_time': CURRENT_UTC,
                'evaluator': CURRENT_USER
            },
            'results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
        print(f"Detailed results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

# —— 7. Save Wrong Questions for a Model —— 
def save_wrong_questions(model_name, results, qa_data, filename):
    """Save incorrect answers for specific model"""
    wrong_questions = []
    
    for response in results[model_name]['responses']:
        if not response['is_correct']:
            q_idx = response['question_idx']
            original_question = next((q for q in qa_data if q['question_index'] == q_idx), None)
            
            if original_question:
                wrong_question = original_question.copy()
                wrong_question['model_answer'] = response['model_answer']
                wrong_question['raw_responses'] = response['raw_responses']
                wrong_question['parsing_failed'] = response['parsing_failed']
                wrong_questions.append(wrong_question)
    
    try:
        wrong_data = {
            'metadata': {
                'evaluation_time': CURRENT_UTC,
                'evaluator': CURRENT_USER,
                'model_name': model_name
            },
            'wrong_questions': wrong_questions
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(wrong_data, f, indent=2)
        
        print(f"{model_name} got {len(wrong_questions)} questions wrong out of {len(qa_data)}.")
        print(f"Wrong questions saved to {filename}")
    except Exception as e:
        print(f"Error saving wrong questions for {model_name}: {e}")
    
    return wrong_questions

# —— 8. Save Wrong Questions for All Models —— 
def save_all_wrong_questions(results, qa_data):
    """Save wrong questions for all models"""
    wrong_questions_dir = f'wrong_questions_{CURRENT_UTC.replace(" ", "_").replace(":", "-")}'
    os.makedirs(wrong_questions_dir, exist_ok=True)
    
    model_file_mapping = {
        'o3-mini-2025-01-31': 'o3mini_wrong_questions.json',
        'gpt-4o-2024-11-20': '4o_wrong_questions.json',
        'claude-3-5-sonnet-20241022': 'sonnet3.5_wrong_questions.json',
        'claude-3-5-haiku-20241022': 'haiku_wrong_questions.json',
        'gemini-2.0-flash-001': 'gemini2.0flash_wrong_questions.json',
        'llama3.2:3b': 'llama3.2-3b_wrong_questions.json',
        'gemma2:9b': 'gemma2-9b_wrong_questions.json'
    }
    
    for model_name, filename in model_file_mapping.items():
        full_path = os.path.join(wrong_questions_dir, filename)
        save_wrong_questions(model_name, results, qa_data, full_path)

# —— 9. Save Model Complete Answers —— 
def save_model_complete_answers(model_name, results, qa_data, filename):
    """Save all answers (both correct and incorrect) for a specific model"""
    complete_answers = []
    
    for response in results[model_name]['responses']:
        q_idx = response['question_idx']
        original_question = next((q for q in qa_data if q['question_index'] == q_idx), None)
        
        if original_question:
            answer_record = {
                'question_index': q_idx,
                'question_content': original_question['question_content'],
                'options': original_question['options'],
                'correct_answer': response['correct_answer'],
                'model_answer': response['model_answer'],
                'parsing_failed': response['parsing_failed'],
                'raw_responses': response['raw_responses']
            }
            complete_answers.append(answer_record)
    
    complete_answers.sort(key=lambda x: x['question_index'])
    
    try:
        complete_data = {
            'metadata': {
                'evaluation_time': CURRENT_UTC,
                'evaluator': CURRENT_USER,
                'model_name': model_name
            },
            'evaluation_results': {
                'total_questions': len(qa_data),
                'correct_answers': results[model_name]['correct'],
                'accuracy': (results[model_name]['correct'] / len(qa_data)) * 100,
                'answers': complete_answers
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2)
        
        print(f"Complete answers for {model_name} saved to {filename}")
    except Exception as e:
        print(f"Error saving complete answers for {model_name}: {e}")

# —— 10. Save All Models Complete Answers —— 
def save_all_models_complete_answers(results, qa_data):
    """Save complete results for all models"""
    results_dir = f'complete_results_{CURRENT_UTC.replace(" ", "_").replace(":", "-")}'
    os.makedirs(results_dir, exist_ok=True)
    
    model_file_mapping = {
        'o3-mini-2025-01-31': 'o3mini_complete.json',
        'gpt-4o-2024-11-20': '4o_complete.json',
        'claude-3-5-sonnet-20241022': 'sonnet3.5_complete.json',
        'claude-3-5-haiku-20241022': 'haiku_complete.json',
        'gemini-2.0-flash-001': 'gemini2.0flash_complete.json',
        'llama3.2:3b': 'llama3.2-3b_complete.json',
        'gemma2:9b': 'gemma2-9b_complete.json'
    }
    
    for model_name, filename in model_file_mapping.items():
        full_path = os.path.join(results_dir, filename)
        save_model_complete_answers(model_name, results, qa_data, full_path)
    
    # Save a summary file with timestamp
    summary = {
        'metadata': {
            'evaluation_time': CURRENT_UTC,
            'evaluator': CURRENT_USER
        },
        'summary': {
            'total_questions': len(qa_data),
            'models_summary': {}
        }
    }
    
    for model_name in model_file_mapping.keys():
        summary['summary']['models_summary'][model_name] = {
            'total_correct': results[model_name]['correct'],
            'accuracy': (results[model_name]['correct'] / len(qa_data)) * 100
        }
    
    summary_path = os.path.join(results_dir, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComplete evaluation results saved in '{results_dir}' directory")

def main():
    """Main execution function"""
    try:
        print(f"Starting evaluation at {CURRENT_UTC}")
        print(f"Evaluator: {CURRENT_USER}")
        
        # Load questions
        qa_data = load_questions()
        print(f"Loaded {len(qa_data)} questions successfully")
        
        # Run evaluation
        print("\nStarting model evaluations...")
        results = run_evaluation(qa_data)
        
        # Save original results
        results_filename = f'model_evaluation_results_{CURRENT_UTC.replace(" ", "_").replace(":", "-")}.json'
        save_results(results, results_filename)
        
        # Save wrong questions for all models
        print("\nSaving wrong questions for all models...")
        save_all_wrong_questions(results, qa_data)
        
        # Save complete answers for all models
        print("\nSaving complete results for all models...")
        save_all_models_complete_answers(results, qa_data)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

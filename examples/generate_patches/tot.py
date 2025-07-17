import os
import logging
import datetime
import tempfile
import subprocess
import argparse
import json
from typing import Dict, List

from graph_of_thoughts import controller, language_models, prompter, parser
from graph_of_thoughts.operations import (
    Operation, OperationType, Thought, Generate, Aggregate, GraphOfOperations, Score, KeepBestN
)
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser

class PatchPrompter(prompter.Prompter):
    """
    Prompter for generating, improving, and aggregating code patches.
    """

    generate_patch_prompt = """<Instruction>
    You are an expert software engineer specializing in security.
    Based on the provided root cause analysis and the vulnerable code, rewrite the entire code to fix the vulnerability.

    **VERY IMPORTANT RULES:**
    1.  You MUST output the COMPLETE, modified Java code for the file.
    2.  The code must be syntactically correct and compilable.
    3.  Do NOT output a `diff` or a patch. Output the entire file content.

    Output ONLY the code content within <PatchedCode> and </PatchedCode> tags.
    </Instruction>
    <Root Cause Analysis>{root_cause}</Root Cause Analysis>
    <Vulnerable Code>
    ```java
    {vulnerable_code}
    ```
    </Vulnerable Code>
    <PatchedCode>
    """

    improve_patch_prompt = """<Instruction>
You are an expert software engineer. Your previously generated code failed to compile.
Analyze the original vulnerable code, your faulty code, and the Java compiler error. Then, rewrite the entire code to fix the compilation error and the original vulnerability.

**VERY IMPORTANT RULES:**
1.  You MUST output the COMPLETE, fixed Java code for the file.
2.  The code must be syntactically correct and compilable.
3.  Do NOT output a `diff` or a patch. Output the entire file content.

Output ONLY the new code content within <PatchedCode> and </PatchedCode> tags.
</Instruction>
<Vulnerable Code>
```java
{vulnerable_code}
```
</Vulnerable Code>
<Faulty Code>
```java
{patched_code}
```
</Faulty Code>
<Compiler Error Log>
{error}
</Compiler Error Log>
<PatchedCode>
"""

    # --- aggregate_patches_prompt 수정 시작 ---
    # ToT에서는 최종 Aggregation이 일반적이지 않으므로, 이 프롬프트는 ToT 기반 그래프에서는 사용되지 않을 수 있습니다.
    # 하지만 GoT 프레임워크의 Aggregate Operation을 사용한다면 필요합니다.
    # 여기서는 ToT의 '최종 선택'에 초점을 맞추므로, 이 프롬프트는 사용되지 않습니다.
    aggregate_patches_prompt = """<Instruction>
You are a master software architect and security expert.
You have been provided with several previously generated candidate solutions, all of which are compilable.
Your task is to analyze each solution, considering not only its assigned score but also its detailed rationale and content, to synthesize a single, optimal and superior final version of the code. This final version must incorporate the best ideas and insights from all candidates.
Refer to relevant patches, but you don't need to if you deem them unnecessary.

Here are the criteria for evaluating the code:
1.  **Vulnerability Fix (Weight: 40%)**: Does the new code correctly and completely fix the described vulnerability?
2.  **Correctness (Weight: 35%)**: Is the code syntactically correct and free of obvious bugs? Does it preserve the original functionality?
3.  **Code Quality (Weight: 15%)**: Is the code clean, well-structured, and maintainable?
4.  **Minimality of Change (Weight: 10%)**: Can the vulnerability be fixed with the minimal necessary code modifications, avoiding unnecessary changes?

The final output must be the complete, final Java code.
Output the final, synthesized code within <FinalCode> and </FinalCode> tags.
</Instruction>
<Root Cause Analysis>{root_cause}</Root Cause Analysis>
<Vulnerable Code>
```java
{vulnerable_code}
```
</Vulnerable Code>
<Validated Candidate Solutions>
{patches}
</Validated Candidate Solutions>
<FinalCode>
"""
    # --- aggregate_patches_prompt 수정 끝 ---

    _score_prompt_template = """<Instruction>
You are a senior software engineer and security expert.
Your task is to evaluate a generated code solution based on the original vulnerability and code.
Provide a score from 1 to 10 based on the following criteria:
1.  **Vulnerability Fix (Weight: 40%)**: Does the new code correctly and completely fix the described vulnerability?
2.  **Correctness (Weight: 35%)**: Is the code syntactically correct and free of obvious bugs? Does it preserve the original functionality?
3.  **Code Quality (Weight: 15%)**: Is the code clean, well-structured, and maintainable?
4.  **Minimality of Change (Weight: 10%)**: Can the vulnerability be fixed with the minimal necessary code modifications, avoiding unnecessary changes?

Your output MUST be a JSON object with two keys: "score" (a float from 1.0 to 10.0) and "rationale" (a brief explanation for your score, in one or two sentences).
Example:
{{
    "score": 8.5,
    "rationale": "The patch effectively addresses the vulnerability by adding input validation, but the hard-coded safe class list could be more flexible."
}}

<Root Cause Analysis>{root_cause}</Root Cause Analysis>
<Vulnerable Code>
```java
{vulnerable_code}
```
</Vulnerable Code>
<Generated Code>
```java
{patched_code}
```
</Generated Code>
<Evaluation>
"""

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        return self.generate_patch_prompt.format(**kwargs)

    def improve_prompt(self, **kwargs) -> str:
        return self.improve_patch_prompt.format(**kwargs)

    # --- aggregation_prompt 메서드 수정 시작 ---
    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        state = state_dicts[0] # root_cause와 vulnerable_code는 후보들 간에 일관적이라고 가정
        patches_str = ""
        for i, d in enumerate(state_dicts):
            score = d.get('score', 'N/A')
            rationale = d.get('rationale', 'N/A')
            patches_str += f"--- Candidate Solution {i+1} (Score: {score}) ---\n"
            patches_str += f"Rationale: {rationale}\n"
            patches_str += f"```java\n{d['patched_code']}\n```\n\n"
        
        return self.aggregate_patches_prompt.format(
            root_cause=state['root_cause'],
            vulnerable_code=state['vulnerable_code'],
            patches=patches_str.strip()
        )
    # --- aggregation_prompt 메서드 수정 끝 ---

    def validation_prompt(self, **kwargs) -> str: pass
    
    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        # 이 구현에서는 한 번에 하나의 사고에 대해 점수를 매긴다고 가정
        state = state_dicts[0]
        return self._score_prompt_template.format(
            root_cause=state['root_cause'],
            vulnerable_code=state['vulnerable_code'],
            patched_code=state['patched_code']
        )


class PatchParser(parser.Parser):
    def strip_answer_helper(self, text: str, tag: str) -> str:
        try:
            content = ""
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            
            # 먼저 태그 내에서 내용 찾기 시도
            start_idx = text.find(start_tag)
            if start_idx != -1:
                start_idx += len(start_tag)
                end_idx = text.find(end_tag, start_idx)
                if end_idx != -1:
                    content = text[start_idx:end_idx]

            # 태그로 내용을 찾지 못했다면, 마크다운 블록 확인
            if not content.strip():
                if '```java' in text:
                    # ```java와 ``` 사이의 내용 추출
                    content = text.split('```java')[1].split('```')[0]
                elif '```' in text:
                    # 여러 블록을 처리하기 위한 더 견고한 분할
                    parts = text.split('```')
                    if len(parts) > 1:
                        # 첫 번째 블록이 원하는 블록이라고 가정
                        content = parts[1]
                        # 첫 줄에 'java'가 있을 수 있음
                        if content.lower().strip().startswith('java'):
                            content = content.strip()[4:]

            return content.strip()
        except Exception:
            return ""

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states = []
        for text in texts:
            patched_code = self.strip_answer_helper(text, "PatchedCode")
            if patched_code:
                new_state = state.copy()
                new_state['patched_code'] = patched_code
                new_states.append(new_state)
        return new_states
    
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        patched_code = self.strip_answer_helper(texts[0], "PatchedCode")
        if patched_code:
            new_state = state.copy()
            new_state['patched_code'] = patched_code
            return new_state
        return state

    def parse_aggregation_answer(self, original_states: List[Dict], texts: List[str]) -> List[Dict]:
        final_code = self.strip_answer_helper(texts[0], "FinalCode")
        if final_code:
            new_state = original_states[0].copy()
            new_state['final_code'] = final_code
            return [new_state]
        return []

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool: return False
    
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        scores = []
        for i, text in enumerate(texts):
            try:
                # 응답에서 JSON 부분 추출
                json_text = self.strip_answer_helper(text, "Evaluation")
                if not json_text:
                    # 원시 JSON 응답에 대한 대체
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_text = text[start:end]

                parsed = json.loads(json_text)
                score = float(parsed.get("score", 0.0))
                rationale = parsed.get("rationale", "")
                
                # 해당 상태를 추론으로 업데이트
                if i < len(states):
                    states[i]['score'] = score
                    states[i]['rationale'] = rationale
                
                scores.append(score)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logging.error(f"Failed to parse score: {e}\nResponse was: {text}")
                # 파싱 실패에 대한 낮은 점수 부여
                if i < len(states):
                    states[i]['rationale'] = f"Failed to parse score from LLM response: {text}"
                scores.append(0.0)
        return scores

# --- ValidateAndImproveOperation 클래스: Refining 구현 ---
class ValidateAndImproveOperation(Operation):
    """
    Custom operation to validate a generated Java code by compiling it.
    If it fails, it feeds the error back to the LLM to generate an improved version.
    This operation implements the 'Refining' aspect of GoT by iteratively improving a thought.
    """
    def __init__(self, prompter: prompter.Prompter, vulnerable_file_name: str, num_tries: int = 3):
        super().__init__()
        self.operation_type = OperationType.validate_and_improve
        self.prompter = prompter
        self.vulnerable_file_name = vulnerable_file_name
        self.num_tries = num_tries
        self.thoughts: List[Thought] = [] # 유효/개선된 사고 저장

    def _compile_java_code(self, code: str, file_name: str) -> str:
        """
        Java 코드를 컴파일하는 대신, 컴파일 단계를 건너뛰고 항상 성공을 반환합니다.
        사용자 요청에 따라 실제 컴파일은 수행하지 않습니다.
        """
        logging.info("Skipping actual Java compilation as per user request.")
        return "" # 빈 문자열은 컴파일 성공을 의미

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        # 이전 작업(예: Generate)에서 생성된 사고들을 가져옵니다.
        input_thoughts = self.get_previous_thoughts()
        
        # 각 입력 사고에 대해 개선 과정을 처리할 것입니다.
        refined_thoughts = []

        for original_thought in input_thoughts:
            current_patch_state = original_thought.state.copy()
            vulnerable_code = current_patch_state.get('vulnerable_code')
            root_cause = current_patch_state.get('root_cause')

            if not vulnerable_code:
                logging.error("Vulnerable code not found in thought state for validation.")
                original_thought.valid = False
                refined_thoughts.append(original_thought)
                continue

            for attempt in range(self.num_tries):
                patched_code = current_patch_state.get('patched_code')
                if not patched_code:
                    logging.warning(f"No patched code found for refinement attempt {attempt+1}. Skipping.")
                    original_thought.valid = False # 코드가 없으면 유효하지 않음
                    refined_thoughts.append(original_thought)
                    break # 이 사고에 대한 개선 시도 중단

                # 현재 패치를 컴파일합니다. (실제 컴파일은 건너뛰고 항상 성공)
                compiler_output = self._compile_java_code(patched_code, self.vulnerable_file_name)

                if not compiler_output: # 컴파일 성공 (빈 문자열)
                    logging.info(f"Thought {original_thought.id} compiled successfully after {attempt+1} tries (simulated).")
                    current_patch_state['valid'] = True # 유효하다고 표시
                    original_thought.state = current_patch_state # 원본 사고 상태 업데이트
                    original_thought.valid = True
                    refined_thoughts.append(original_thought)
                    break # 다음 원본 사고로 이동
                else:
                    # 이 else 블록은 _compile_java_code가 항상 빈 문자열을 반환하므로 사실상 실행되지 않습니다.
                    # 하지만 로직의 완전성을 위해 유지합니다.
                    logging.info(f"Thought {original_thought.id} failed compilation. Attempting to improve (try {attempt+1}/{self.num_tries}).")
                    # 개선을 위한 프롬프트 준비
                    improve_prompt_text = prompter.improve_prompt(
                        vulnerable_code=vulnerable_code,
                        patched_code=patched_code,
                        error=compiler_output
                    )
                    
                    # LLM에게 개선 요청
                    improved_texts = lm.generate_text(improve_prompt_text, num_branches=1) # 1개의 개선된 버전 생성
                    
                    if improved_texts:
                        # 개선된 응답 파싱
                        improved_state = parser.parse_improve_answer(current_patch_state, improved_texts)
                        if improved_state and improved_state.get('patched_code'):
                            current_patch_state = improved_state # 다음 시도에 개선된 상태 사용
                            logging.debug(f"Thought {original_thought.id} improved. New patch length: {len(current_patch_state['patched_code'])}")
                        else:
                            logging.warning(f"Failed to parse improved patch for thought {original_thought.id}.")
                            original_thought.valid = False # 개선 파싱 실패 시 유효하지 않음으로 표시
                            refined_thoughts.append(original_thought)
                            break # 이 사고에 대한 개선 시도 중단
                    else:
                        logging.warning(f"LLM failed to generate improvement for thought {original_thought.id}.")
                        original_thought.valid = False # LLM이 응답하지 않으면 유효하지 않음으로 표시
                        refined_thoughts.append(original_thought)
                        break # 이 사고에 대한 개선 시도 중단
            else: # 루프가 성공적인 컴파일 없이 종료됨 (현재 _compile_java_code 로직에서는 도달하지 않음)
                logging.warning(f"Thought {original_thought.id} failed to compile after {self.num_tries} attempts.")
                original_thought.valid = False
                refined_thoughts.append(original_thought) # 유효하지 않더라도 추가하여 나중에 필터링 가능

        # 다음 작업으로 유효한 사고들만 전달합니다.
        self.thoughts = [t for t in refined_thoughts if t.valid]

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

# --- ValidateAndImproveOperation 클래스 수정 끝 ---

# --- ToT 기반 GraphOfOperations 정의 함수 ---
def tot_patch_graph(patch_prompter, patch_parser, vulnerable_file_name) -> GraphOfOperations:
    """
    Defines a Tree of Thoughts (ToT)-like graph for patch generation.
    This simulates a multi-stage refinement/expansion process.
    """
    graph = GraphOfOperations()

    # --- Level 1: Initial Idea Generation and Selection ---
    # Generate initial patch ideas (branches from root)
    op1_generate_initial = Generate(num_branches_prompt=5) # 5개의 초기 아이디어 생성
    graph.add_operation(op1_generate_initial)

    # Validate and refine these initial ideas (simulated compilation/improvement loop)
    op2_refine_initial = ValidateAndImproveOperation(patch_prompter, vulnerable_file_name, num_tries=2)
    graph.add_operation(op2_refine_initial)
    op1_generate_initial.add_successor(op2_refine_initial)

    # Score the initial (and refined) ideas
    op3_score_initial = Score(combined_scoring=False)
    graph.add_operation(op3_score_initial)
    op2_refine_initial.add_successor(op3_score_initial)

    # Keep the single best initial idea to expand upon (ToT's "best path" selection)
    op4_keep_best_initial = KeepBestN(n=1) # 단일 최적 아이디어 선택
    graph.add_operation(op4_keep_best_initial)
    op3_score_initial.add_successor(op4_keep_best_initial)

    # --- Level 2: Expand/Refine the Best Initial Idea ---
    # Generate new thoughts based on the best initial idea (deeper branches)
    # The 'thought_id' will be set by the Controller to the best thought from op4
    op5_generate_refined = Generate(num_branches_prompt=3) # 최적 아이디어에서 3개의 새로운 아이디어 생성
    graph.add_operation(op5_generate_refined)
    # op4_keep_best_initial의 출력이 op5_generate_refined의 입력으로 사용됩니다.
    # GoT 프레임워크의 Generate operation은 thought_id를 통해 이전 Thought를 참조할 수 있습니다.
    op4_keep_best_initial.add_successor(op5_generate_refined)

    # Validate and refine these refined ideas
    op6_refine_refined = ValidateAndImproveOperation(patch_prompter, vulnerable_file_name, num_tries=2)
    graph.add_operation(op6_refine_refined)
    op5_generate_refined.add_successor(op6_refine_refined)

    # Score the refined ideas
    op7_score_refined = Score(combined_scoring=False)
    graph.add_operation(op7_score_refined)
    op6_refine_refined.add_successor(op7_score_refined)

    # Keep the single best refined idea (final selection for this path)
    op8_keep_best_refined = KeepBestN(n=1) # 최종 단일 최적 아이디어 선택
    graph.add_operation(op8_keep_best_refined)
    op7_score_refined.add_successor(op8_keep_best_refined)

    # --- Final Output ---
    # The final best refined thought is the output. No aggregation in typical ToT.
    # The 'run' function will retrieve the output from op8.

    return graph

def run(args):
    # --- LLM 통신 로깅 설정 ---
    log_dir = os.path.join(os.path.dirname(__file__), "responses")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{timestamp}_responses.log")
    
    # LLM 통신 전용 로거 생성
    llm_logger = logging.getLogger("llm_communication")
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False # 기본 로거로 전파 방지
    
    # 다른 핸들러가 이미 설정되어 있다면 제거 (재실행 시 중복 방지)
    if llm_logger.hasHandlers():
        llm_logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    llm_logger.addHandler(file_handler)
    # --- 로깅 설정 끝 ---

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(filename).1s:%(lineno)d] - %(message)s')

    script_dir = os.path.dirname(__file__)
    try:
        with open(os.path.join(script_dir, args.vulnerable_file), "r") as f:
            vulnerable_code = f.read()
        with open(os.path.join(script_dir, args.root_cause_file), "r") as f:
            root_cause = f.read()
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 언어 모델 설정
    lm_config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "graph_of_thoughts", "language_models", "config.json"
    )

    # --model 인수에 따라 언어 모델 동적 로드
    try:
        if args.model == 'chatgpt':
            lm = language_models.ChatGPT.from_config(args.config, "chatgpt", logger=llm_logger)
        elif args.model == 'gemini':
            lm = language_models.Gemini.from_config(args.config, "gemini", logger=llm_logger)
        elif args.model.startswith('ollama:'):
            model_name_in_config = "ollama"  # 일반적인 "ollama" 설정 섹션 가정
            lm = language_models.Ollama.from_config(args.config, model_name_in_config, cli_model_name=args.model, logger=llm_logger)
        else:
            logging.error(f"Unknown model: {args.model}")
            return
    except Exception as e:
        logging.error(f"Failed to initialize language model: {e}")
        return

    patch_prompter = PatchPrompter()
    patch_parser = PatchParser()
    
    initial_state = {
        "vulnerable_code": vulnerable_code,
        "root_cause": root_cause,
    }
    
    # --- Backtracking 구현: 최종 점수 기반 재시도 ---
    max_overall_attempts = 3 # 전체 ToT 프로세스를 재시도할 최대 횟수
    min_acceptable_score = 7.0 # 최종 패치의 최소 허용 점수

    for overall_attempt in range(max_overall_attempts):
        print(f"\n--- Starting overall attempt {overall_attempt + 1}/{max_overall_attempts} ---")
        
        # 각 재시도마다 새로운 GraphOfOperations 인스턴스를 생성하여 그래프 상태를 초기화
        # (이전 시도의 영향을 받지 않도록)
        # ToT 기반 그래프 사용
        graph = tot_patch_graph(patch_prompter, patch_parser, args.vulnerable_file)
        ctrl = controller.Controller(lm, graph, patch_prompter, patch_parser, initial_state)

        print("Starting Patch Generation with ToT-like approach...")
        start_time = datetime.datetime.now()
        ctrl.run()
        end_time = datetime.datetime.now()
        print(f"Patch generation finished in {end_time - start_time}.")

        # 최종 스코어링 작업(op8_keep_best_refined, graph.operations의 인덱스 7)에서 최적의 결과 가져오기
        # ToT는 최종 Aggregation이 없으므로, 마지막 KeepBestN의 결과를 가져옵니다.
        final_results = graph.operations[7].get_thoughts() # 인덱스 조정 (op8_keep_best_refined)
        
        current_final_score = 0.0
        if final_results:
            best_thought = final_results[0] # ToT는 단일 최적 경로의 최종 사고를 선택
            final_code = best_thought.state.get('patched_code', 'Final code not found.') # ToT는 'patched_code'가 최종 결과
            current_final_score = best_thought.state.get('score', 0.0)
            rationale = best_thought.state.get('rationale', 'N/A')

            print("\n" + "="*50)
            print(f"Overall Attempt {overall_attempt + 1} - Final ToT Patch (Score: {current_final_score})")
            print("="*50)
            print(f"Rationale: {rationale}")
            print("-" * 50)
            print(final_code)

            # 백트래킹 조건 확인
            if current_final_score >= min_acceptable_score:
                print(f"\nFinal score {current_final_score} meets acceptable threshold ({min_acceptable_score}). Stopping.")
                # 최종 성공 패치 저장
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(args.output_dir, f"generated_tot_patch_SUCCESS_{timestamp}.java")
                with open(file_path, "w") as f:
                    f.write(f"// Score: {current_final_score}\n// Rationale: {rationale}\n\n{final_code}")
                print(f"\nFinal successful code saved to {file_path}")
                return # 만족스러운 패치를 찾았으므로 run 함수 종료
            else:
                print(f"\nFinal score {current_final_score} is below acceptable threshold ({min_acceptable_score}). Retrying...")
                # 필요하다면 다음 시도를 위해 initial_state 또는 그래프 파라미터 수정
                # 예: op1_generate.num_branches_prompt = 10 # 다음 시도를 위해 브랜치 수 증가
        else:
            print(f"\nNo code was generated or survived the ToT process in overall attempt {overall_attempt + 1}.")
            print("Retrying...")
    
    print(f"\n--- All {max_overall_attempts} overall attempts completed. No satisfactory patch found. ---")
    # --- Backtracking 구현 끝 ---

def main():
    """
    스크립트의 메인 진입점. 인수를 파싱하고 run 함수를 호출합니다.
    """
    parser = argparse.ArgumentParser(description="Generate and validate security patches for a Java vulnerability.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The language model to use for generation and scoring (e.g., 'chatgpt', 'gemini', 'ollama:qwen2').",
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument(
        "--root_cause_file",
        type=str,
        default="root_cause.txt",
        help="Path to the root cause analysis file.",
    )
    parser.add_argument(
        "--vulnerable_file",
        type=str,
        default="vulnerable.java",
        help="Path to the vulnerable Java file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory to save the generated patch.",
    )
    parser.add_argument(
        "--use-aggregation", # 이 인수는 ToT에서는 직접 사용되지 않지만, 기존 파서 호환성을 위해 유지
        action="store_true",
        help="Use the advanced graph with an aggregation step."
    )
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()

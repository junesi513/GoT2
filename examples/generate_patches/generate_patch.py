import os
import logging
import datetime
import tempfile
import subprocess
import argparse
import json
from typing import Dict, List, Any, Union
import configparser
from abc import ABC, abstractmethod

# javalang 라이브러리 임포트 시도
try:
    import javalang
except ImportError:
    print("javalang is not installed. Please install it using: pip install javalang")
    javalang = None

# Elasticsearch 라이브러리 임포트 -> 제거
import glob

from graph_of_thoughts import controller, language_models, prompter, parser
from graph_of_thoughts.operations import (
    Operation, OperationType, Thought, Generate, Aggregate, GraphOfOperations, Score, KeepBestN
)
# from graph_of_thoughts.language_models import AbstractLanguageModel # 이제 사용 안 함
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser

# --- Temporary AbstractLanguageModel Definition ---
class _AbstractLanguageModel(ABC):
    def __init__(self, config: Dict = None, model_name: str = "", cache: bool = False, logger: logging.Logger = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_logger = logger
        self.config = config if config is not None else {}
        self.model_name: str = model_name
        self.cache = cache
        if self.cache: self.response_cache: Dict[str, List[Any]] = {}
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0
    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any: pass
    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]: pass
# --- End of Temporary Definition ---


# --- Temporary OllamaLanguageModel Definition ---
import requests
import tempfile
class _OllamaLanguageModel(_AbstractLanguageModel):
    def __init__(self, config: Dict, cache: bool = False, logger: logging.Logger = None) -> None:
        model_name = config.get("model_name", "ollama")
        super().__init__(model_name=model_name, cache=cache, logger=logger)
        
        self.config = config
        self.model_name = self.config.get("model_name") # CLI에서 받은 모델 이름으로 설정
        self.server_url = self.config.get("server_url", "http://localhost:11434")
        self.api_endpoint = f"{self.server_url}/api/chat"
        self.logger.info(f"Ollama model initialized with model '{self.model_name}' on server {self.server_url}")
        self._check_server_connection()

    def _check_server_connection(self) -> None:
        try:
            response = requests.get(self.server_url, timeout=5)
            response.raise_for_status()
            self.logger.info("Successfully connected to Ollama server.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Could not connect to Ollama server at {self.server_url}.")
            raise ConnectionError(f"Ollama server not reachable: {e}")

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        if self.cache and query in self.response_cache: return self.response_cache[query]
        if self.llm_logger: self.llm_logger.info(f"--- REQUEST ---\n{query}\n")
        responses = []
        for _ in range(num_responses):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": query}],
                    "stream": False,
                    "options": {"temperature": 1.0, "num_predict": 4096, "stop": []}
                }
                response = requests.post(self.api_endpoint, json=payload, timeout=300)
                response.raise_for_status()
                response_json = response.json()
                # /api/chat 응답 형식에 맞게 수정
                content = response_json.get("message", {}).get("content", "")
                responses.append({"generated_text": content})
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error querying Ollama: {e}")
                responses.append({"generated_text": ""})
        if self.llm_logger:
            response_text = "\n".join([r["generated_text"] for r in responses])
            self.llm_logger.info(f"--- RESPONSE ---\n{response_text}\n")
        if self.cache: self.response_cache[query] = responses
        return responses

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        return [query_response["generated_text"] for query_response in query_responses]
    
    def generate_text(self, prompt: str, num_branches: int) -> List[str]:
        response = self.query(prompt, num_responses=num_branches)
        return self.get_response_texts(response)
# --- End of Temporary Definition ---

# --- Temporary GeminiLanguageModel Definition ---
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Google Generative AI is not installed. Please install it with `pip install google-generativeai`")

class _GeminiLanguageModel(_AbstractLanguageModel):
    def __init__(self, config: Dict, cache: bool = False, logger: logging.Logger = None) -> None:
        model_name = config.get("model_name", "gemini")
        super().__init__(model_name=model_name, cache=cache, logger=logger)
        self.config = config
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "gemini-1.5-pro-latest")
        if not self.api_key: self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key: raise ValueError("Gemini API key not found.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.logger.info(f"Gemini model initialized with {self.model_name}")

    def _query_lm(self, prompt: str, n: int = 1, temperature: float = 1.0, max_tokens: int = 4096, stop=None) -> Dict[str, Any]:
        responses = []
        for _ in range(n):
            try:
                response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens))
                responses.append(response.text)
            except Exception as e:
                logging.error(f"Error querying Gemini: {e}")
                responses.append("")
        return {"choices": [{"message": {"content": text}} for text in responses]}

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        if self.cache and query in self.response_cache: return self.response_cache[query]
        if self.llm_logger: self.llm_logger.info(f"--- REQUEST ---\n{query}\n")
        response_dict = self._query_lm(prompt=query, n=num_responses)
        responses = []
        for choice in response_dict.get("choices", []):
            responses.append({"generated_text": choice["message"]["content"]})
        if self.llm_logger:
            response_text = "\n".join([r["generated_text"] for r in responses])
            self.llm_logger.info(f"--- RESPONSE ---\n{response_text}\n")
        if self.cache: self.response_cache[query] = responses
        return responses

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        return [query_response["generated_text"] for query_response in query_responses]

    def generate_text(self, prompt: str, num_branches: int) -> List[str]:
        response = self.query(prompt, num_responses=num_branches)
        return self.get_response_texts(response)
# --- End of Temporary Gemini Definition ---

# --- Temporary ChatGPT Definition ---
import backoff
import openai
from openai import OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

class _ChatGPT(_AbstractLanguageModel):
    def __init__(self, config: Dict, cache: bool = False, logger: logging.Logger = None) -> None:
        model_name = config.get("model_name", "chatgpt")
        super().__init__(model_name=model_name, cache=cache, logger=logger)
        self.config: Dict = config
        self.model_id: str = self.config["model_name"]
        self.prompt_token_cost: float = float(self.config.get("prompt_token_cost", 0.03))
        self.response_token_cost: float = float(self.config.get("response_token_cost", 0.06))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        # 컨텍스트 길이 초과 오류를 피하기 위해 max_tokens를 4096으로 강제합니다.
        self.max_tokens: int = 4096
        self.stop: Union[str, List[str], None] = self.config.get("stop")
        self.organization: str = self.config.get("organization")
        self.api_key: str = self.config.get("api_key")
        if not self.api_key: self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key: raise ValueError("OpenAI API key not found.")
        if self.organization: openai.organization = self.organization
        if self.api_key: openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key, organization=self.organization)

    def generate_text(self, prompt: str, num_branches: int) -> List[str]:
        response = self.query(prompt, num_responses=num_branches)
        return self.get_response_texts(response)

    def query(self, query: str, num_responses: int = 1) -> Union[List[ChatCompletion], ChatCompletion]:
        if self.cache and query in self.response_cache: return self.response_cache[query]
        if self.llm_logger: self.llm_logger.info(f"--- REQUEST ---\n{query}\n")
        response = self.chat([{"role": "user", "content": query}], num_responses)
        if self.llm_logger:
            response_text = "\n".join([choice.message.content for choice in response.choices])
            self.llm_logger.info(f"--- RESPONSE ---\n{response_text}\n")
        if self.cache: self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        response = self.client.chat.completions.create(model=self.model_id, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, n=num_responses, stop=self.stop)
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        return response

    def get_response_texts(self, query_response: Union[List[ChatCompletion], ChatCompletion]) -> List[str]:
        if not isinstance(query_response, List): query_response = [query_response]
        return [choice.message.content for response in query_response for choice in response.choices]
# --- End of Temporary ChatGPT Definition ---


"""
1. Generate: 코드 패치 생성 ** - k 개를 지정해 패치 생성

2. Improve: 코드 패치 개선 **
    - 생성된 코드에 문제가 있다면 코드를 패치(by llm)
    - Q. 어떻게 문제가 있다고 판별할 것인가?
        - LLM?
        - 다른 도구?

3. Validate: 코드 패치 유효성 검사 **
    - 컴파일 실행
    - LLM 정적 분석

4. Score: 코드 패치 점수 매기기 **
    - 코드 패치 점수 매기기

5. Aggregate: 코드 패치 집계 ** - 코드 패치 집계

6. KeepBestN: 최적의 코드 패치 선택 **
    - 최적의 코드 패치 선택

+ 활용할 수 있는 정보?
- root cause
- 컴파일 (mvn clean compile)

+  패치를 생성할 때, 추론 절차를 포함해 출력해야 하는가?
- 추론을 통해 
""" 

# RAG 컨텍스트 로드 함수
def load_rag_context(rag_file_path: str) -> Dict:
    """RAG 컨텍스트 파일을 로드합니다."""
    try:
        with open(rag_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"RAG 컨텍스트 로드 실패: {e}")
        return {}

class LocalRAGSearcher:
    """
    로컬 JSON 파일에서 RAG 컨텍스트를 검색하는 클래스.
    """
    def __init__(self, rag_directory: str):
        self.rag_data = []
        rag_path = os.path.join(rag_directory, '*.json')
        try:
            for file_path in glob.glob(rag_path):
                with open(file_path, 'r') as f:
                    self.rag_data.append(json.load(f))
            logging.info(f"Successfully loaded {len(self.rag_data)} RAG files from {rag_path}")
        except Exception as e:
            logging.error(f"Failed to load RAG files: {e}")

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Jaccard 유사도를 계산하는 간단한 함수."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    def search(self, query_purpose: str, top_k: int = 1) -> List[Dict]:
        if not self.rag_data or not query_purpose:
            return []
        
        scores = []
        for doc in self.rag_data:
            doc_purpose = doc.get("purpose", "")
            if doc_purpose:
                similarity = self._calculate_similarity(query_purpose, doc_purpose)
                scores.append((similarity, doc))
        
        # 점수가 높은 순으로 정렬
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 k개의 문서 내용 반환
        return [doc for score, doc in scores[:top_k]]


class PatchPrompter(prompter.Prompter):
    """
    Prompter for generating, improving, and aggregating code patches.
    """

    def __init__(self, rag_context: Dict = None):
        super().__init__()
        self.rag_context = rag_context or {}
        if rag_context:
            self.security_workflow = rag_context.get("security_analyst_workflow", {})
            self.patch_design = self.security_workflow.get("patch_design", {})
            # RAG에서 취약점 분석 가이드라인 로드 (vulnerability_analysis.steps로 변경)
            self.vulnerability_analysis_steps = self.security_workflow.get("vulnerability_analysis", {}).get("steps", [])


    # 코드 의미론 분석을 위한 프롬프트
    code_semantics_prompt_template = """<Instruction>
You are an expert software engineer specializing in security.
please generate the code semantics of the following code.
you should generate the code semantics in the following format:

title: code title
purpose: A concise, one-sentence description of what the code does.
concept: code concept
core_flaw: code core flaw
analogy: code analogy
quote: code quote

You must output the code semantics in the following format:
```json
{json_example}
```
</Instruction>

<Code>
```java
{code}
```
</Code>
<Analysis>
"""
    
    json_example_for_semantics = {
        "vulnerable_code_abstraction": {
            "title": "Abstract Representation of Vulnerable Code: Unconditional Type Trustor",
            "purpose": "This code deserializes a JSON array into a Java array of a specified type, attempting to handle generic types.",
            "concept": "This code assumes that type information passed from an external source is trustworthy and delegates to the internal parser logic based on this assumption. Despite the possibility that 'componentClass' and 'componentType' can differ, it uses them interchangeably in certain sections, performing instantiation without validation.",
            "core_flaw": "The core flaw is that critical information determining the system's behavior (the type) is introduced from external input but is used without any structural validation or reliability checks. Specifically, if the type information used in the type inference process and the parsing delegation process mismatches, malicious objects can be created through a vulnerable path.",
            "analogy": "This is like passing a blueprint received from an external source to the production line without review. Even though problems in the blueprint could lead to the creation of dangerous machinery or parts, a structure that passes it on based solely on its appearance is highly vulnerable from a security perspective.",
            "quote": "Under the assumption that 'all input can be potentially hostile,' the reliability of input data must be verified before it is used for system control decisions. (Source: Secure Coding Principles)"
        }
    }


    def _format_patch_design_steps(self) -> str:
        """패치 설계 단계를 포맷팅합니다."""
        if not self.rag_context:
            return ""
            
        steps = self.patch_design.get("steps", [])
        if not steps:
            return ""
            
        guidance = "\nFollow these patch design steps:\n"
        guidance += "\n".join([f"{step['step']}. {step['name']}: {step['description']}" 
                             for step in steps])
        return guidance

    def _format_vulnerability_analysis_guidance(self) -> str:
        """RAG 컨텍스트에서 취약점 분석 가이드라인 (단계)를 포맷팅합니다."""
        if not self.rag_context or not self.vulnerability_analysis_steps:
            return ""
        
        guidance_text = ["\nFollow these vulnerability analysis steps:"]
        for step in self.vulnerability_analysis_steps:
            guidance_text.append(f"- Step {step['step']}. {step['name']}: {step['description']}")
        
        return "\n".join(guidance_text) if guidance_text else ""

    def generate_code_semantics_prompt(self, code: str) -> str:
        """코드 의미론 분석을 위한 프롬프트를 생성합니다."""
        # 이스케이프 문제를 피하기 위해 f-string 대신 format 사용
        json_str = json.dumps(self.json_example_for_semantics, indent=4)
        return self.code_semantics_prompt_template.format(
            json_example=json_str,
            code=code
        )

    vulnerability_analysis_prompt_template = """<Instruction>
You are an expert security analyst. Analyze the provided code and its semantics to identify the root cause of the vulnerability.
Focus on the specific code patterns, logic flaws, or missing checks that lead to the security issue.

Your analysis should be concise and structured as a JSON object.

You must output the analysis in the following format:
```json
{{
  "vulnerability_analysis": {{
    "title": "A brief, descriptive title of the vulnerability.",
    "root_cause_summary": "A detailed explanation of the core technical reason for the vulnerability. Explain *why* the code is insecure.",
    "attack_scenario": "Describe a potential attack scenario that could exploit this vulnerability.",
    "recommendation": "Provide a high-level recommendation for fixing the vulnerability."
  }}
}}
```
</Instruction>
<Code Semantics Analysis>
{code_semantics}
</Code Semantics Analysis>
<Vulnerable Code>
```java
{vulnerable_code}
```
</Vulnerable Code>
<Analysis>
"""

    def generate_vulnerability_analysis_prompt(self, vulnerable_code: str, code_semantics: Dict) -> str:
        """취약점 분석을 위한 프롬프트를 생성합니다."""
        semantics_str = json.dumps(code_semantics, indent=2)
        return self.vulnerability_analysis_prompt_template.format(
            vulnerable_code=vulnerable_code,
            code_semantics=semantics_str
        )

    # 패치를 생성하는 프롬프트
    generate_patch_prompt = """<Instruction>
    You are an expert software engineer specializing in security.
    Based on the provided code semantics analysis and the vulnerable code, rewrite the entire code to fix the vulnerability.
    If available, use the provided RAG (Retrieval-Augmented Generation) context which contains relevant security principles or similar solved cases.

    **VERY IMPORTANT RULES:**
    1.  You MUST output the COMPLETE, modified Java code for the file.
    2.  The code must be syntactically correct and compilable.
    3.  Do NOT output a `diff` or a patch. Output the entire file content.

    Output ONLY the code content within <PatchedCode> and </PatchedCode> tags.
    </Instruction>
    <Code Semantics Analysis>
    {code_semantics}
    </Code Semantics Analysis>
    <Vulnerable Code>
    ```java
    {vulnerable_code}
    ```
    </Vulnerable Code>
    <RAG Context>
    {rag_context}
    </RAG Context>
    <PatchedCode>
    """

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        # kwargs에서 필요한 모든 데이터를 가져옵니다.
        # JSON 객체일 수 있으므로 문자열로 변환합니다.
        semantics_str = json.dumps(kwargs.get("code_semantics", {}), indent=2)
        vuln_analysis_str = json.dumps(kwargs.get("vulnerability_analysis", {}), indent=2)
        rag_str = json.dumps(kwargs.get("rag_context", []), indent=2)

        return self.generate_patch_prompt.format(
            code_semantics=semantics_str,
            vulnerability_analysis=vuln_analysis_str,
            vulnerable_code=kwargs.get("vulnerable_code", ""),
            rag_context=rag_str
        )

    # 코드 패치를 개선하는데 활용,
    # Validate 단계 이후에 활용
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
    # Aggregate : 예제 코드와 병합하는 과정
    aggregate_patches_prompt = """<Instruction>
You are a master software architect and security expert.
You have been provided with several previously generated candidate solutions, all of which are compilable.
Your task is to analyze each solution, considering not only its assigned score but also its detailed rationale and content, to synthesize a single, optimal and superior final version of the code. This final version must incorporate the best ideas and insights from all candidates.
Here are the criteria for evaluating the code:
Refer to relevant patches, but you don't need to if you deem them unnecessary.

Here are the criteria for evaluating the code:
1.  **Vulnerability Fix (Weight: 30%)**: Does the new code correctly and completely fix the described vulnerability based on the semantics?
2.  **Correctness (Weight: 25%)**: Is the code syntactically correct and free of obvious bugs? Does it preserve the original functionality?
3.  **Code Quality (Weight: 25%)**: Is the code clean, well-structured, and maintainable?
4.  **Minimality of Change (Weight: 20%)**: Can the vulnerability be fixed with the minimal necessary code modifications, avoiding unnecessary changes?
The final output must be the complete, final Java code.
Output the final, synthesized code within <FinalCode> and </FinalCode> tags.
</Instruction>
<Code Semantics Analysis>
{code_semantics}
</Code Semantics Analysis>
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

    # Score : 코드 패치 점수 매기기
    _score_prompt_template = """<Instruction>
You are a senior software engineer and security expert.
Your task is to evaluate a generated code solution based on the original vulnerability and code.
Provide a score from 1 to 10 based on the following criteria:
1.  **Vulnerability Fix (Weight: 30%)**: Does the new code correctly and completely fix the described vulnerability?
2.  **Correctness (Weight: 25%)**: Is the code syntactically correct and free of obvious bugs? Does it preserve the original functionality?
3.  **Code Quality (Weight: 25%)**: Is the code clean, well-structured, and maintainable?
4.  **Minimality of Change (Weight: 20%)**: Can the vulnerability be fixed with the minimal necessary code modifications, avoiding unnecessary changes?

Your output MUST be a JSON object with two keys: "score" (a float from 1.0 to 10.0) and "rationale" (a brief explanation for your score, in one or two sentences).
Example:
{{
    "score": 8.5,
    "rationale": "The patch effectively addresses the vulnerability by adding input validation, but the hard-coded safe class list could be more flexible.",
}}

<Code Semantics Analysis>
{code_semantics}
</Code Semantics Analysis>
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

    def improve_prompt(self, **kwargs) -> str:
        return self.improve_patch_prompt.format(**kwargs)

    # --- aggregation_prompt 메서드 수정 시작 ---
    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        state = state_dicts[0] # code_semantics와 vulnerable_code는 후보들 간에 일관적이라고 가정
        patches_str = ""
        for i, d in enumerate(state_dicts):
            score = d.get('score', 'N/A')
            rationale = d.get('rationale', 'N/A')
            patches_str += f"--- Candidate Solution {i+1} (Score: {score}) ---\n"
            patches_str += f"Rationale: {rationale}\n"
            patches_str += f"```java\n{d['patched_code']}\n```\n\n"
        
        return self.aggregate_patches_prompt.format(
            code_semantics=json.dumps(state.get('code_semantics', {}), indent=2),
            vulnerable_code=state['vulnerable_code'],
            patches=patches_str.strip()
        )
    # --- aggregation_prompt 메서드 수정 끝 ---

    def validation_prompt(self, **kwargs) -> str: pass
    
    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        # 이 구현에서는 한 번에 하나의 사고에 대해 점수를 매긴다고 가정
        state = state_dicts[0]
        return self._score_prompt_template.format(
            code_semantics=json.dumps(state.get('code_semantics', {}), indent=2),
            vulnerable_code=state['vulnerable_code'],
            patched_code=state['patched_code']
        )


class VulnerabilityAnalysisOperation(Operation):
    """
    CodeSemantics를 기반으로 취약점 분석을 수행하는 작업.
    """
    def __init__(self):
        super().__init__()
        self.operation_type = OperationType.generate
        self.thoughts: List[Thought] = []

    def _execute(self, lm: _AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        input_thoughts = self.get_previous_thoughts()
        for thought in input_thoughts:
            code = thought.state.get("vulnerable_code")
            semantics = thought.state.get("code_semantics")
            if not code or not semantics:
                self.thoughts.append(thought)
                continue

            prompt = prompter.generate_vulnerability_analysis_prompt(
                vulnerable_code=code,
                code_semantics=semantics
            )
            
            responses = lm.generate_text(prompt, num_branches=1)
            
            if responses:
                new_state = parser.parse_vulnerability_analysis_answer(thought.state, responses)
                new_thought = Thought(state=new_state)
                new_thought.valid = True
                self.thoughts.append(new_thought)
            else:
                logging.error("No response from LLM for vulnerability analysis")
                self.thoughts.append(thought) # 실패해도 원래 thought 전달

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

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
            # 일관성을 위해 'patched_code' 키 사용
            new_state['patched_code'] = final_code
            new_state['final_code'] = final_code # 호환성을 위해 유지
            return [new_state]
        return []

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool: return False
    
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        scores = []
        for i, text in enumerate(texts):
            try:
                # Gemini가 ```json ... ``` 와 같이 반환하는 경우가 있어서 파싱 로직 강화
                json_text = text
                if '```json' in json_text:
                    json_text = json_text.split('```json')[1].split('```')[0]
                elif '```' in json_text:
                    json_text = json_text.split('```')[1].split('```')[0]
                
                # 만약 그래도 파싱이 안되면, 가장 바깥쪽의 { } 를 기준으로 파싱 시도
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError:
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start != -1 and end > start:
                        json_text = text[start:end]
                        parsed = json.loads(json_text)
                    else:
                        raise
                        
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

    def parse_vulnerability_analysis_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        취약점 분석 결과를 파싱합니다.
        """
        try:
            if not texts or not texts[0].strip():
                logging.warning("Received empty response for vulnerability analysis.")
                return state

            text = texts[0]
            json_str = self.strip_answer_helper(text, "Analysis")
            if not json_str:
                if '```json' in text:
                    json_str = text.split('```json')[1].split('```')[0]
                elif '```' in text:
                    json_str = text.split('```')[1].split('```')[0]
            
            if not json_str.strip():
                if text.strip().startswith('{'):
                    json_str = text
                else:
                    logging.warning("Could not find JSON block in the response for vulnerability analysis.")
                    return state
            
            if json_str.strip().startswith('json'):
                json_str = json_str.strip()[4:].strip()

            analysis_result = json.loads(json_str)
            
            new_state = state.copy()
            new_state["vulnerability_analysis"] = analysis_result.get("vulnerability_analysis", {})
            return new_state

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse vulnerability analysis result: {e}")
            return state
            
    def parse_code_semantics_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        코드 의미론 분석 결과를 파싱합니다.
        """
        try:
            if not texts or not texts[0].strip():
                logging.warning("Received empty response for code semantics analysis.")
                return state
                
            text = texts[0]
            # ```json ... ``` 블록 찾기
            json_str = self.strip_answer_helper(text, "Analysis")
            if not json_str:
                if '```json' in text:
                    json_str = text.split('```json')[1].split('```')[0]
                elif '```' in text: # Fallback for just ```
                    json_str = text.split('```')[1].split('```')[0]
            
            if not json_str.strip():
                # 태그나 마크다운 없이 순수 JSON만 반환된 경우일 수 있음
                # 하지만 비어있다면 파싱 시도하지 않음
                if text.strip().startswith('{'):
                    json_str = text
                else:
                    logging.warning("Could not find JSON block in the response for code semantics.")
                    return state
            
            # Gemini가 응답 시작 부분에 'json'이라는 단어를 추가하는 경우가 있음
            if json_str.strip().startswith('json'):
                json_str = json_str.strip()[4:].strip()

            # JSON 파싱을 시도하고, 실패할 경우를 대비합니다.
            try:
                semantics_result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON for code semantics: {e}. Response was: {json_str}")
                return state

            new_state = state.copy()
            new_state["code_semantics"] = semantics_result
            return new_state
            
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logging.error(f"Failed to parse code semantics result: {e}")
            return state

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

    def _execute(self, lm: _AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
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

def extract_java_methods(code: str) -> Dict[str, str]:
    """
    javalang 라이브러리를 사용하여 Java 코드에서 각 메서드의 이름과 소스 코드를 추출합니다.
    """
    if not javalang:
        logging.error("javalang library is not available. Cannot extract methods.")
        return {}
        
    methods = {}
    try:
        tree = javalang.parse.parse(code)
        # CompilationUnit에서 직접 MethodDeclaration을 찾습니다.
        for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
            # 메서드 시작과 끝 라인 찾기
            start_line = method_node.position.line
            
            # 메서드 본문의 끝을 나타내는 닫는 중괄호 '}'를 찾습니다.
            # 이를 위해 토큰 스트림을 순회해야 합니다.
            # 이 방법은 javalang의 내부 구조에 의존하므로 주의가 필요합니다.
            # 더 간단한 방법은 다음 메서드의 시작점을 찾는 것입니다.
            # 여기서는 우선 간단하게 전체 메서드 블록을 가져오도록 시도합니다.
            
            # 메서드 코드 추출을 위해 전체 코드를 줄 단위로 분할
            code_lines = code.splitlines()
            
            # 메서드 본문을 포함한 전체를 가져오기 위해 마지막 토큰의 라인을 찾음
            # 이 방법이 불안정하다면, 다음 노드의 시작 라인을 찾는 방법으로 돌아가야 합니다.
            try:
                # 메서드 노드에 직접 접근할 수 있는 토큰이 있는지 확인
                # javalang 0.13.0 에서는 MethodDeclaration 객체에 tokens 속성이 없습니다.
                # 대신, 더 견고한 방법으로 메서드 범위를 추정해야 합니다.
                
                # 중괄호 균형을 맞춰 메서드 끝을 찾습니다.
                body_start_token = None
                for token in method_node.parameters: # 파라미터 뒤에서 시작
                    pass # 마지막 파라미터 토큰 찾기
                
                # body를 직접 찾는 API가 없으므로, 수동으로 찾아야 합니다.
                # 이는 매우 복잡하므로, 다른 접근법을 사용합니다.
                
                # 모든 노드를 가져와서 위치를 기반으로 범위를 결정합니다.
                all_nodes = list(tree.filter(lambda n: hasattr(n, 'position') and n.position is not None))
                all_nodes.sort(key=lambda n: n.position.line)
                
                current_node_index = -1
                for i, n in enumerate(all_nodes):
                    if n == method_node:
                        current_node_index = i
                        break
                
                end_line = len(code_lines)
                if current_node_index != -1 and current_node_index + 1 < len(all_nodes):
                    next_node = all_nodes[current_node_index + 1]
                    # 다음 노드가 같은 줄이나 이전 줄에 있으면 안됨
                    if next_node.position.line > start_line:
                         end_line = next_node.position.line -1

                # heuristic: 메서드 시그니처 후 첫 '{' 부터 마지막 '}' 까지
                method_text_lines = code_lines[start_line - 1:]
                open_braces = 0
                method_end_line_in_slice = 0
                body_found = False
                for i, line in enumerate(method_text_lines):
                    if '{' in line and not body_found:
                        body_found = True
                    if body_found:
                        open_braces += line.count('{')
                        open_braces -= line.count('}')
                        if open_braces == 0:
                            method_end_line_in_slice = i
                            break
                
                end_line = start_line + method_end_line_in_slice
                methods[method_node.name] = "\n".join(code_lines[start_line-1 : end_line])

            except Exception as e_inner:
                logging.warning(f"Could not determine end of method {method_node.name} precisely: {e_inner}")
                # 실패 시, 단순히 시작 라인부터 20라인을 가져오는 등의 대체 로직
                methods[method_node.name] = "\n".join(code.splitlines()[start_line-1 : start_line+19])
                
    except Exception as e:
        logging.error(f"Failed to parse Java code and extract methods: {e}")
    return methods

class CodeSemanticsOperation(Operation):
    """
    코드의 의미론적 특성을 분석하는 작업
    """
    def __init__(self):
        super().__init__()
        self.operation_type = OperationType.generate  # 기존 타입 재사용
        self.thoughts: List[Thought] = []

    def _execute(self, lm: _AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        input_thoughts = self.get_previous_thoughts()
        if not input_thoughts:
            # 초기 상태에서 시작하는 경우
            vulnerable_code = kwargs.get("vulnerable_code")
            if not vulnerable_code:
                logging.error("No vulnerable_code provided for code semantics analysis")
                return
            thought = Thought(state=kwargs)
            thought.valid = True
            input_thoughts = [thought]

        for thought in input_thoughts:
            # 코드 의미론 분석 프롬프트 생성
            prompt = prompter.generate_code_semantics_prompt(
                code=thought.state.get("vulnerable_code", "")
            )
            
            # LLM에 분석 요청
            responses = lm.generate_text(prompt, num_branches=1)
            
            if responses:
                # 응답 파싱 및 상태 업데이트
                new_state = parser.parse_code_semantics_answer(thought.state, responses)
                new_thought = Thought(state=new_state)
                new_thought.valid = True
                self.thoughts.append(new_thought)
            else:
                logging.error("No response from LLM for code semantics analysis")

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

class RAGSearchOperation(Operation):
    """
    CodeSemantics를 기반으로 로컬 JSON 파일에서 RAG 컨텍스트를 검색하는 작업.
    """
    def __init__(self, rag_searcher: LocalRAGSearcher):
        super().__init__()
        self.operation_type = OperationType.generate
        self.rag_searcher = rag_searcher
        self.thoughts: List[Thought] = []

    def _execute(self, lm: _AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        input_thoughts = self.get_previous_thoughts()
        for thought in input_thoughts:
            semantics = thought.state.get("code_semantics", {})
            vuln_analysis = thought.state.get("vulnerability_analysis", {})

            if not vuln_analysis or not self.rag_searcher:
                self.thoughts.append(thought)
                continue
            
            # 검색 쿼리 생성 (취약점 분석 요약 사용)
            query_text = vuln_analysis.get('root_cause_summary', '')
            
            if query_text:
                rag_results = self.rag_searcher.search(query_purpose=query_text) # query_purpose 파라미터 이름 유지
                thought.state['rag_context'] = rag_results
                logging.info(f"Found {len(rag_results)} RAG documents for thought {thought.id}")
            
            self.thoughts.append(thought)

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


def advanced_patch_graph_with_aggregation(patch_prompter, patch_parser, vulnerable_file_name) -> GraphOfOperations:
    """
    취약점 분석과 패치 생성을 위한 작업 그래프를 정의합니다.
    """
    # 0단계: 코드 의미론 분석
    op0_semantics = CodeSemanticsOperation()

    # 1단계: 취약점 분석
    op1_analyze = VulnerabilityAnalysisOperation()

    # 2단계: 여러 후보 패치 생성
    op2_generate = Generate(num_branches_prompt=5)

    # 3단계: 생성된 각 패치 유효성 검사 및 개선
    op3_refine = ValidateAndImproveOperation(patch_prompter, vulnerable_file_name, num_tries=2)

    # 4단계: 유효하고 개선된 각 후보에 대해 개별적으로 점수 매기기
    op4_score = Score(combined_scoring=False)

    # 5단계: 집계를 위해 상위 N개의 최적 후보 유지
    op5_keep_best_n = KeepBestN(n=3)

    # 6단계: 유지된 상위 N개 후보의 통찰력 집계
    op6_aggregate = Aggregate()

    # 7단계: 집계된 패치에 점수 매기기
    op7_score_final = Score(combined_scoring=False)

    # 그래프 흐름 정의
    graph = GraphOfOperations()
    graph.add_operation(op0_semantics)
    graph.add_operation(op1_analyze)
    graph.add_operation(op2_generate)
    graph.add_operation(op3_refine)
    graph.add_operation(op4_score)
    graph.add_operation(op5_keep_best_n)
    graph.add_operation(op6_aggregate)
    graph.add_operation(op7_score_final)

    # 작업 연결
    op0_semantics.add_successor(op1_analyze)    # 의미론 분석 -> 취약점 분석
    op1_analyze.add_successor(op2_generate)     # 분석 -> 생성
    op2_generate.add_successor(op3_refine)      # 생성 -> 개선
    op3_refine.add_successor(op4_score)         # 개선 -> 점수
    op4_score.add_successor(op5_keep_best_n)    # 점수 -> 최적 N개 유지
    op5_keep_best_n.add_successor(op6_aggregate)# 최적 N개 유지 -> 집계
    op6_aggregate.add_successor(op7_score_final)# 집계 -> 최종 점수

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

    # 모델 이름을 기반으로 출력 디렉토리 설정
    model_name_sanitized = args.model.replace(":", "_")
    args.output_dir = os.path.join(args.output_dir, model_name_sanitized)

    try:
        with open(args.vulnerable_file, "r") as f:
            java_code = f.read()
    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # --- 로컬 RAG 검색기 초기화 ---
    script_dir = os.path.dirname(__file__)
    rag_dir = os.path.join(script_dir, "..", "..", "RAG")
    rag_searcher = LocalRAGSearcher(rag_directory=rag_dir)
            
    # 언어 모델 설정
    # --model 인수에 따라 언어 모델 동적 로드
    try:
        if args.model.startswith('chatgpt'):
            config = configparser.ConfigParser()
            config.read(args.config)
            model_config = dict(config['chatgpt'])
            lm = _ChatGPT(config=model_config, logger=llm_logger)
        elif args.model.startswith('gemini'):
            config = configparser.ConfigParser()
            config.read(args.config)
            model_config = dict(config['gemini'])
            lm = _GeminiLanguageModel(config=model_config, logger=llm_logger)
        elif args.model.startswith('ollama:'):
            model_name_cli = args.model.split(':', 1)[1]
            config = configparser.ConfigParser()
            config.read(args.config)
            model_config = {}
            if config.has_section('ollama'):
                model_config = dict(config.items('ollama'))
            model_config['model_name'] = model_name_cli
            lm = _OllamaLanguageModel(config=model_config, logger=llm_logger)
        else:
            logging.error(f"Unknown model: {args.model}")
            return
    except Exception as e:
        logging.error(f"Failed to initialize language model: {e}")
        return

    patch_prompter = PatchPrompter()
    patch_parser = PatchParser()

    # Java 파일에서 모든 함수 추출
    methods = extract_java_methods(java_code)
    if not methods:
        logging.error("No methods found in the provided Java file.")
        return

    print(f"Found {len(methods)} methods to analyze: {list(methods.keys())}")

    # --- 새로운 실행 로직: 의미 분석 -> RAG 검색 -> 패치 생성 ---
    # 여기서는 첫 번째 함수에 대해서만 패치를 생성하도록 단순화합니다.
    target_method_name, target_method_code = list(methods.items())[0]
    
    print(f"\n--- Generating patch for the first method: {target_method_name} ---")

    initial_state = {"vulnerable_code": target_method_code}
    
    # 새로운 그래프 정의
    graph = GraphOfOperations()
    op1_semantics = CodeSemanticsOperation()
    op2_vuln_analysis = VulnerabilityAnalysisOperation()
    op3_rag_search = RAGSearchOperation(rag_searcher=rag_searcher)
    op4_generate = Generate(num_branches_prompt=1) # 1개의 패치만 생성

    graph.add_operation(op1_semantics)
    graph.add_operation(op2_vuln_analysis)
    graph.add_operation(op3_rag_search)
    graph.add_operation(op4_generate)

    op1_semantics.add_successor(op2_vuln_analysis)
    op2_vuln_analysis.add_successor(op3_rag_search)
    op3_rag_search.add_successor(op4_generate)

    ctrl = controller.Controller(lm, graph, patch_prompter, patch_parser, problem_parameters=initial_state)
    ctrl.run()

    # 결과 출력
    results = op4_generate.get_thoughts()
    if results:
        best_thought = results[0]
        patched_code = best_thought.state.get('patched_code', 'Patch not generated.')
        
        print("\n" + "="*50)
        print(f"Generated Patch for: {target_method_name}")
        print("="*50)
        print(patched_code)

        # 패치 파일 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(args.output_dir, f"generated_patch_{target_method_name}_{timestamp}.java")
        with open(file_path, "w") as f:
            f.write(patched_code)
        print(f"\nGenerated patch saved to {file_path}")
    else:
        print(f"Could not generate patch for method: {target_method_name}")


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
    parser.add_argument("--config", type=str, default="config.ini", help="Path to the configuration file.")
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
        "--use-aggregation",
        action="store_true",
        help="Use the advanced graph with an aggregation step."
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Use RAG (Retrieval-Augmented Generation) for enhanced patch generation."
    )
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()

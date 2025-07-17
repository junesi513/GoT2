import os
import logging
import datetime
from typing import Dict, List, Union

from graph_of_thoughts import controller, language_models, operations, prompter, parser

# 예시 취약점 코드
VULNERABLE_CODE_SNIPPET = """
// @author wenshao[szujobs@hotmail.com]
public class ObjectArrayCodec implements ObjectSerializer, ObjectDeserializer {
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public <T> T deserialze(DefaultJSONParser parser, Type type, Object fieldName) {
    
        Class componentClass;
        
        Type componentType;
        if (type instanceof GenericArrayType) {
            GenericArrayType clazz = (GenericArrayType) type;
            componentType = clazz.getGenericComponentType();

            if (componentType instanceof TypeVariable) {
                TypeVariable typeVar = (TypeVariable) componentType;
                
                Type objType = parser.getContext().type;
                
                if (objType instanceof ParameterizedType) {
                    ParameterizedType objParamType = (ParameterizedType) objType;
                    Type objRawType = objParamType.getRawType();
                    Type actualType = null;
                    if (objRawType instanceof Class) {
                        TypeVariable[] objTypeParams = ((Class) objRawType).getTypeParameters();
                        for (int i = 0; i < objTypeParams.length; ++i) {
                            if (objTypeParams[i].getName().equals(typeVar.getName())) {
                                actualType = objParamType.getActualTypeArguments()[i];
                            }
                        }
                    }
                    if (actualType instanceof Class) {
                        componentClass = (Class) actualType;
                    } else {
                        componentClass = Object.class;
                    }
                } else {
                    componentClass = TypeUtils.getClass(typeVar.getBounds()[0]);
                }
            } else {
                componentClass = TypeUtils.getClass(componentType);
            }
        } else {
            Class clazz = (Class) type;
            componentType = componentClass = clazz.getComponentType();
        }

        JSONArray array = new JSONArray();
        
        parser.parseArray(componentClass, array, fieldName);

        return (T) toObjectArray(parser, componentClass, array);
    }
}
"""

class RootCauseAnalysisPrompter(prompter.Prompter):
    """
    취약점 근본 원인 분석을 위한 프롬프트를 생성합니다. (GoT Aggregate 방식)
    """

    # 1단계: 초기 분석 프롬프트
    initial_analysis_prompt = """<Instruction>
You are a top-tier security researcher. Your task is to analyze the provided Java vulnerable code snippet and identify the immediate cause of the specified vulnerability.
Focus on the exact line of code and the function being used.
Output your analysis within <Analysis> and </Analysis> tags.
</Instruction>

<Vulnerability Type>
{vulnerability_type}
</Vulnerability Type>

<Source Code>
```java
{source_code}
```
</Source Code>

<Analysis>
"""

    # 2단계 (신규): 분석 종합 프롬프트
    aggregate_analysis_prompt = """<Instruction>
You are a master security analyst. You have been provided with several independent analyses of the same code snippet.
Your task is to synthesize these analyses into a single, comprehensive, and more accurate "Aggregated Analysis".
Discard any incorrect or irrelevant points and combine the valid points into a coherent explanation of the vulnerability's data flow and cause.
Output the result within <AggregatedAnalysis> and </AggregatedAnalysis> tags.
</Instruction>

<Source Code>
```java
{source_code}
```
</Source Code>

<Analyses>
{analyses}
</Analyses>

<AggregatedAnalysis>
"""
    
    # 3단계: 보고서 생성 프롬프트 (입력 변경)
    generate_report_prompt = """<Instruction>
Based on the comprehensive "Aggregated Analysis", determine the ultimate root cause of the vulnerability and suggest a high-level recommendation.
The root cause should be a general principle or design flaw.
Structure your final report as follows:
- **Root Cause**: [Explain the root cause here]
- **Recommendation**: [Provide your recommendation here]
Output the entire report within <Report> and </Report> tags.
</Instruction>

<Source Code>
```java
{source_code}
```
</Source Code>

<Aggregated Analysis>
{aggregated_analysis}
</Aggregated Analysis>

<Report>
"""

    def generate_prompt(
        self,
        num_branches: int,
        source_code: str,
        vulnerability_type: str,
        aggregated_analysis: str = "",
        **kwargs
    ) -> str:
        current_phase = kwargs.get('phase', 1)
        if current_phase == 1:
            return self.initial_analysis_prompt.format(
                vulnerability_type=vulnerability_type, source_code=source_code
            )
        elif current_phase == 2:
            return self.generate_report_prompt.format(
                source_code=source_code, aggregated_analysis=aggregated_analysis
            )
        else:
            raise ValueError(f"Unknown phase for generation: {current_phase}")

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        source_code = state_dicts[0]['source_code']
        # 여러개의 initial_analysis를 하나의 문자열로 합칩니다.
        analyses = "\n\n---\n\n".join([f"Analysis {i+1}:\n{d['initial_analysis']}" for i, d in enumerate(state_dicts)])
        return self.aggregate_analysis_prompt.format(source_code=source_code, analyses=analyses)

    # 사용하지 않는 프롬프트 메서드
    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str: pass
    def improve_prompt(self, **kwargs) -> str: pass
    def validation_prompt(self, **kwargs) -> str: pass


class RootCauseAnalysisParser(parser.Parser):
    """
    LLM의 분석 결과를 파싱합니다. (GoT Aggregate 방식)
    """

    def strip_answer_helper(self, text: str, tag: str) -> str:
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            start_index = text.find(start_tag) + len(start_tag)
            end_index = text.find(end_tag)
            return text[start_index:end_index].strip()
        except Exception:
            return ""

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        current_phase = state.get('phase', 1)
        new_states = []
        for text in texts:
            new_state = state.copy()
            if current_phase == 1:
                analysis = self.strip_answer_helper(text, "Analysis")
                if analysis:
                    new_state['initial_analysis'] = analysis
                    # Aggregate 단계로 가기 전 phase는 변경하지 않음
                    new_states.append(new_state)
            elif current_phase == 2:
                report = self.strip_answer_helper(text, "Report")
                if report:
                    new_state['report'] = report
                    new_state['phase'] = 3  # 최종 단계
                    new_states.append(new_state)
        return new_states

    def parse_aggregation_answer(self, original_states: List[Dict], texts: List[str]) -> List[Dict]:
        # Aggregate는 하나의 종합된 생각만 생성한다고 가정
        text = texts[0]
        aggregated_analysis = self.strip_answer_helper(text, "AggregatedAnalysis")
        if aggregated_analysis:
            # 원본 state에서 필요한 정보를 가져와 새 state를 만듭니다.
            new_state = original_states[0].copy()
            new_state['aggregated_analysis'] = aggregated_analysis
            new_state['phase'] = 2  # 다음 단계(보고서 생성)로 설정
            return [new_state]
        return []

    # 사용하지 않는 파서 메서드
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]: return []
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict: pass
    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool: pass


def root_cause_graph() -> operations.GraphOfOperations:
    """
    취약점 분석을 위한 GoT 그래프를 생성합니다.
    분기(Generate) -> 종합(Aggregate) -> 보고서 생성(Generate)
    """
    # 1. 초기 분석 생성 (3개의 서로 다른 관점)
    op1 = operations.Generate(num_branches_prompt=3)

    # 2. 생성된 분석들을 하나의 종합 분석으로 병합
    op2 = operations.Aggregate()

    # 3. 종합된 분석을 바탕으로 최종 보고서 생성
    op3 = operations.Generate()

    # 그래프 구성
    graph = operations.GraphOfOperations()
    graph.add_operation(op1)
    graph.add_operation(op2)
    graph.add_operation(op3)
    
    op1.add_successor(op2)
    op2.add_successor(op3)

    return graph


def run(lm_name: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "graph_of_thoughts", "language_models", "config.json")
        lm = language_models.ChatGPT(config_path=config_path, model_name=lm_name)
    except Exception as e:
        logging.error(f"LM 초기화 오류: {e}")
        logging.error("LM 설정 파일(예: config.json)이 올바르게 설정되었는지 확인하세요.")
        return

    rca_prompter = RootCauseAnalysisPrompter()
    rca_parser = RootCauseAnalysisParser()
    g = root_cause_graph()

    initial_state = {
        "source_code": VULNERABLE_CODE_SNIPPET,
        "vulnerability_type": "Deserialization of Untrusted Data",
        "phase": 1,
    }
    ctrl = controller.Controller(lm, g, rca_prompter, rca_parser, initial_state)

    print("🚀 Starting Root Cause Analysis with GoT (Aggregate)...")
    start_time = datetime.datetime.now()
    ctrl.run()
    end_time = datetime.datetime.now()
    print(f"✅ Analysis finished in {end_time - start_time}.")

    final_results = g.operations[2].get_thoughts() # op3의 인덱스는 2
    if final_results:
        final_report = final_results[0].state.get('report', 'Report not found.')
        print("\n" + "="*50)
        print(" 최종 근본 원인 분석 보고서 (GoT)")
        print("="*50)
        print(final_report)

        if not os.path.exists("results"):
            os.makedirs("results")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"results/GoT_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(final_report)
        print(f"\n📄 Report saved to {report_path}")
    else:
        print("\n❌ 최종 보고서가 생성되지 않았습니다.")

if __name__ == "__main__":
    run(lm_name="chatgpt4") 
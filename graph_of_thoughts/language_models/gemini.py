import os
import json
import logging
from typing import Dict, List, Any
import tempfile

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Google Generative AI is not installed. Please install it with `pip install google-generativeai`"
    )

from .abstract_language_model import AbstractLanguageModel
import configparser

class GeminiLanguageModel(AbstractLanguageModel):
    """
    Language model for interacting with Google's Gemini models.
    """

    def __init__(self, config: Dict, cache: bool = False, logger: logging.Logger = None) -> None:
        """
        Initializes the GeminiLanguageModel.
        """
        model_name = config.get("model_name", "gemini")
        
        temp_config = {model_name: config}
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
            json.dump(temp_config, temp_f)
            temp_path = temp_f.name

        super().__init__(temp_path, model_name, cache, logger)
        os.remove(temp_path)

        self.config = self.config[model_name]
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "gemini-1.5-pro-latest")

        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Please set it in the config file or as an environment variable GOOGLE_API_KEY."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.logger.info(f"Gemini model initialized with {self.model_name}")

    def _query_lm(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        stop=None,
    ) -> Dict[str, Any]:
        """
        Queries the Gemini model.
        Note: Gemini API doesn't support n > 1 directly in a single call with temperature.
              We will call it `n` times to get `n` independent samples.
              The `stop` parameter is also not directly supported in the same way.
        """
        responses = []
        for _ in range(n):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        # candidate_count is not the same as n in OpenAI
                        # it produces n candidates but from a single generation process
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                responses.append(response.text)
            except Exception as e:
                logging.error(f"Error querying Gemini: {e}")
                responses.append("")

        # Mimic the OpenAI response structure
        return {
            "choices": [
                {"message": {"content": text}} for text in responses
            ]
        }

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Query the Gemini model for responses.
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if self.llm_logger:
            self.llm_logger.info(f"--- REQUEST ---\n{query}\n")

        response_dict = self._query_lm(prompt=query, n=num_responses)
        
        responses = []
        for choice in response_dict.get("choices", []):
            responses.append({"generated_text": choice["message"]["content"]})
        
        if self.llm_logger:
            response_text = "\n".join([r["generated_text"] for r in responses])
            self.llm_logger.info(f"--- RESPONSE ---\n{response_text}\n")

        if self.cache:
            self.response_cache[query] = responses

        return responses

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        Extract the response texts from the query response.
        """
        return [query_response["generated_text"] for query_response in query_responses]


    @classmethod
    def from_config(cls, config_path: str, config_key="gemini", logger: logging.Logger = None) -> "GeminiLanguageModel":
        """
        Creates an instance of the language model from a configuration file.
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        model_config = dict(config[config_key])
        return cls(config=model_config, logger=logger) 
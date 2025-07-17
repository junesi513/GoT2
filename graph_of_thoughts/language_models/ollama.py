import os
import json
import logging
from typing import List, Dict, Union, Any
import requests
import configparser

from .abstract_language_model import AbstractLanguageModel

class OllamaLanguageModel(AbstractLanguageModel):
    """
    Language model for interacting with a local Ollama server.
    """

    def __init__(
        self,
        config: Dict,
        cache: bool = False,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize the OllamaLanguageModel instance.
        """
        super().__init__(config, cache, logger)
        self.config = config
        self.model_name = self.config.get("model_name", "qwen2:32b")
        self.server_url = self.config.get("server_url", "http://localhost:11434")
        self.api_endpoint = f"{self.server_url}/api/generate"
        
        self.logger.info(f"Ollama model initialized with model '{self.model_name}' on server {self.server_url}")
        self._check_server_connection()

    def _check_server_connection(self) -> None:
        """
        Check if the Ollama server is reachable.
        """
        try:
            response = requests.get(self.server_url, timeout=5)
            response.raise_for_status()
            self.logger.info("Successfully connected to Ollama server.")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Could not connect to Ollama server at {self.server_url}. Please ensure it's running.")
            raise ConnectionError(f"Ollama server not reachable: {e}")

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Query the Ollama model for responses.
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if self.llm_logger:
            self.llm_logger.info(f"--- REQUEST ---\n{query}\n")

        responses = []
        for _ in range(num_responses):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": query,
                    "stream": False,
                    "options": {
                        "temperature": 1.0,
                        "num_predict": 4096,
                        "stop": [],
                    }
                }
                response = requests.post(self.api_endpoint, json=payload, timeout=300)
                response.raise_for_status()
                response_json = response.json()
                responses.append({
                    "generated_text": response_json.get("response", "")
                })
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error querying Ollama: {e}")
                responses.append({"generated_text": ""})

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

    def generate_text(self, prompt: str, num_branches: int) -> List[str]:
        """
        Generate multiple responses for a given prompt.
        """
        response = self.query(prompt, num_responses=num_branches)
        return self.get_response_texts(response)

    @classmethod
    def from_config(cls, config_path: str, config_key: str = "ollama", model_name: str = None, logger: logging.Logger = None) -> "OllamaLanguageModel":
        """
        Create an instance from a configuration file.
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if config.has_section(config_key):
            model_config = dict(config.items(config_key))
        else:
            model_config = {}
        
        if model_name:
            model_config['model_name'] = model_name
        
        return cls(config=model_config, logger=logger) 
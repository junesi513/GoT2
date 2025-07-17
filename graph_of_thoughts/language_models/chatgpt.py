# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import os
import random
import time
import json
import tempfile
import logging
import openai
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion
import configparser

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self,
        config: Dict,
        cache: bool = False,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.
        """
        model_name = config.get("model_name", "chatgpt")
        
        temp_config = {model_name: config}
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_f:
            json.dump(temp_config, temp_f)
            temp_path = temp_f.name

        super().__init__(temp_path, model_name, cache, logger)
        os.remove(temp_path)

        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_name"]
        self.prompt_token_cost: float = float(self.config.get("prompt_token_cost", 0.03))
        self.response_token_cost: float = float(self.config.get("response_token_cost", 0.06))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.max_tokens: int = int(self.config.get("max_tokens", 4096))
        self.stop: Union[str, List[str], None] = self.config.get("stop")
        self.organization: str = self.config.get("organization")
        self.api_key: str = self.config.get("api_key")

        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set it in the config file or as an environment variable OPENAI_API_KEY."
            )

        if self.organization:
            openai.organization = self.organization
        if self.api_key:
            openai.api_key = self.api_key
        
        self.client = openai.OpenAI(api_key=self.api_key, organization=self.organization)

    def generate(self, prompt: str, num_generations: int) -> List[str]:
        """
        Generates `num_generations` responses for the given prompt.
        """
        response = self.query(prompt, num_responses=num_generations)
        return self.get_response_texts(response)

    def generate_text(self, prompt: str, num_branches: int) -> List[str]:
        """
        Generates `num_branches` responses for the given prompt.
        A convenience method that wraps the generate method.
        """
        response = self.query(prompt, num_responses=num_branches)
        return self.get_response_texts(response)

    @classmethod
    def from_config(cls, config_path: str, config_key: str = "chatgpt", logger: logging.Logger = None) -> "ChatGPT":
        """
        Creates an instance of the ChatGPT language model from a configuration file.
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        model_config = dict(config[config_key])
        return cls(config=model_config, logger=logger)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the OpenAI model.
        :rtype: Dict
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if self.llm_logger:
            self.llm_logger.info(f"--- REQUEST ---\n{query}\n")

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in chatgpt: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.llm_logger:
            # Note: This might not be perfect if response is a list of completions
            if isinstance(response, list):
                all_choices = []
                for r in response:
                    all_choices.extend(r.choices)
                response_text = "\n".join([choice.message.content for choice in all_choices])
            else:
                response_text = "\n".join([choice.message.content for choice in response.choices])
            self.llm_logger.info(f"--- RESPONSE ---\n{response_text}\n")

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: ChatCompletion
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"This is the response from chatgpt: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the OpenAI model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]

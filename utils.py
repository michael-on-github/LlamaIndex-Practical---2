import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from chromadb.utils.embedding_functions import EmbeddingFunction
from dotenv import load_dotenv
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (CompletionResponse, CompletionResponseGen,
                                   CustomLLM, LLMMetadata)
from openai import AzureOpenAI
from pydantic import Field, PrivateAttr

load_dotenv()


class AzureOpenAIEmbedding(BaseEmbedding):
    model: str = Field()
    _client: AzureOpenAI = PrivateAttr()

    def __init__(self, client: AzureOpenAI, model: str):
        super().__init__(model=model)
        self._client = client

    def _get_query_embedding(self, query: str) -> list[float]:
        response = self._client.embeddings.create(input=query, model=self.model)
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        response = await self._client.embeddings.acreate(input=query, model=self.model)
        return response.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._aget_query_embedding(text)


class AzureOpenAIWrapper(CustomLLM):
    model: str = Field()
    deployment_name: str = Field()
    _client: AzureOpenAI = PrivateAttr()

    def __init__(self, client: AzureOpenAI, model: str, deployment_name: str):
        super().__init__(model=model, deployment_name=deployment_name)
        self._client = client

        self._metadata = LLMMetadata(
            model_name=model,
            context_window=4096,
            num_output=256,
            is_chat_model=True,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return self._metadata

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # SYSTEM_PROMPT = "You are a helpful assistant that always explains its reasoning step by step before answering."
        # You are a reasoning agent that uses the ReAct framework.
        # Think step by step, use tools when needed, and explain your reasoning clearly.
        SYSTEM_PROMPT = """
            You are an intelligent agent that solves problems using the ReAct framework.
            First, think step by step. Then decide whether to use a tool. 
            If you use a tool, describe the action and show the result. 
            Finally, summarize your answer.

            Always output your reasoning steps explicitly in the format:
            Thought: ...
            Action: ...
            Observation: ...
            Final Answer: ...

            Do not skip any steps. Think carefully and show your full reasoning before answering.

            After providing the Final Answer, do not repeat the reasoning or continue. End the response.
            Once you provide the Final Answer, stop reasoning and do not continue. End the response with "Final Answer:" followed by the answer.

            Here are some examples:

            Example 1:
            Question: What tables does this database contain?
            Thought: I need to use a tool to help me answer the question.
            Action: load_data
            Action Input: {"query": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"}
            user: Observation: [{'table_name': 'users', 'table_type': 'BASE TABLE', 'table_collation': 'english_india_body_500_cs_as', 'table_comment': None, 'column_count': 5, 'table_schema': 'public', 'table_pk': True, 'table_fk': False}, {'table_name': 'addresses', 'table_type': 'BASE TABLE', 'table_collation': 'english_india_body_500_cs_as', 'table_comment': None, 'column_count': 4, 'table_schema': 'public', 'table_pk': False, 'table_fk': False}]
            assistant: Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
            Action: respond
            Action Input: {"input": "What tables does this database contain?"}
            Observation: users
            Thought: I can answer without using any more tools. I'll use the user's language to answer
            Answer: users

            Example 2:
            Question: How many candidates do you know about?
            Thought: I need to use a tool to help me answer the question.
            Action: retrieve
            Action Input: {"input": "How many candidates do you know about?"}
            Observation: 3
            Thought: I can answer without using any more tools. I'll use the user's language to answer
            Answer: 3

            Example 3:
            Question: What were the working years of a candidate who was Production Control Analyst?
            Thought: I need to use a tool to help me answer the question.
            Action: retrieve
            Action Input: {"input": "What were the working years of a candidate who was Production Control Analyst?"}
            Observation: 11/2004 to 05/2006
            Thought: I can answer without using any more tools. I'll use the user's language to answer
            Answer: 11/2004 to 05/2006
        """
        response = self._client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return CompletionResponse(text=response.choices[0].message.content)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        def gen():
            yield CompletionResponse(text=self.complete(prompt, **kwargs).text)

        return gen()


class AzureEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def __call__(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [r.embedding for r in response.data]


endpoint = os.getenv("ENDPOINT")
model_name = os.getenv("MODEL_NAME")
deployment = os.getenv("DEPLOYMENT")
token_provider_url = os.getenv("TOKEN_PROVIDER_URL")
api_version = os.getenv("API_VERSION")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

token_provider = get_bearer_token_provider(DefaultAzureCredential(), token_provider_url)

DATA_DIR = os.getenv("DATA_DIR", "./data")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./chromadb")

azure_client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
)

llm = AzureOpenAIWrapper(
    client=azure_client, model=model_name, deployment_name=deployment
)

embed_model = AzureOpenAIEmbedding(client=azure_client, model=embedding_model_name)

embed_function = AzureEmbeddingFunction(azure_client, embedding_model_name)





from sentence_transformers import SentenceTransformer
from typing import List

class LlamaIndexEmbeddingWrapper(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, model_path: str = "./bge-small-en"):
        super().__init__()
        self._model = SentenceTransformer(model_path)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query, convert_to_numpy=True).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

embed_model = LlamaIndexEmbeddingWrapper()

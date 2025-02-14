import os
from urllib.parse import urlparse
import json
import shutil
import requests
from pathlib import Path
import time
import json
import uuid

from openai import AzureOpenAI
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, wait_random_exponential, stop_after_attempt

from typing import Any, Union

from pydantic import BaseModel



def extract_filename_without_extension(url):
    """Extract the filename without the extension from a URL."""
    # Parse the URL to get the path
    parsed_url = urlparse(url)
    # Extract the path from the parsed URL
    path = parsed_url.path
    # Use os.path.basename to get the file name from the path
    file_name_with_extension = os.path.basename(path)
    # Use os.path.splitext to remove the extension
    file_name, _ = os.path.splitext(file_name_with_extension)
    return file_name


class ErrorMessage:
    """Error message class"""

    def __init__(self, code: str = None, message: str = None,
                 innererror: "ErrorMessage" = None):
        self.code = code
        self.message = message
        self.innererror = innererror

    def __str__(self):
        innererror_str = f"\nInner Error: {self.innererror}" if self.innererror else ""
        return f'Error: {self.code} - {self.message}.{innererror_str}'

    @staticmethod
    def from_json(json_obj: dict):
        return ErrorMessage(**json_obj)


class AiAssistant:
    """Azure OpenAI Assistant client"""

    def __init__(
            self,
            aoai_end_point: str,
            aoai_api_version: str,
            deployment_name: str,
            aoai_api_key: str,
    ):
        if aoai_api_key is None or aoai_api_key == "":
            print("Using Entra ID/AAD to authenticate")
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default")

            self.client = AzureOpenAI(
                api_version=aoai_api_version,
                azure_endpoint=aoai_end_point,
                azure_ad_token_provider=token_provider,
            )
        else:
            print("Using API key to authenticate")
            self.client = AzureOpenAI(
                api_version=aoai_api_version,
                azure_endpoint=aoai_end_point,
                api_key=aoai_api_key,
            )

        self.model = deployment_name

    @retry(wait=wait_random_exponential(multiplier=1, max=40),
           stop=stop_after_attempt(3))
    def _chat_completion_request(self, messages, tools=None, tool_choice=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                seed=0,
                temperature=0.0,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    def get_answer(
            self,
            system_message: str,
            prompt: Union[str, Any],
            input_schema=None,
            output_schema=None,
    ):
        """Get an answer from the assistant."""

        def schema_to_tool(schema: Any):
            assert schema.__doc__, f"{schema.__name__} is missing a docstring."
            return [{
                "type": "function",
                "function": {
                    "name": schema.__name__,
                    "description": schema.__doc__,
                    "parameters": schema.schema(),
                },
            }], {
                "type": "function",
                "function": {
                    "name": schema.__name__
                }
            }

        tools = None
        tool_choice = None
        if output_schema:
            tools, tool_choice = schema_to_tool(output_schema)

        if input_schema:
            user_message = f"Schema: ```{input_schema.model_json_schema()}```\nData: ```{input_schema.parse_obj(prompt).model_dump_json()}```"
        else:
            user_message = prompt

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
        ]
        response = self._chat_completion_request(messages,
                                                 tools=tools,
                                                 tool_choice=tool_choice)
        assistant_message = response.choices[0].message
        if assistant_message.content:
            return assistant_message.content
        else:
            try:
                return json.loads(
                    assistant_message.tool_calls[0].function.arguments,
                    strict=False)
            except:
                return assistant_message.tool_calls[0].function.arguments

    def get_structured_output_answer(self,
                                     system_prompt: str,
                                     user_prompt: str,
                                     response_format: BaseModel,
                                     seed: int = 0,
                                     temperature: float = 0.0):
        
        print(f"token size: {self.get_token_count(user_prompt, self.model)}")
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                max_tokens=16384,
                seed=seed,
                temperature=temperature)
            response = completion.choices[0].message.parsed
            return response
        except Exception as ex:
            print(
                f"Unable to generate ChatCompletion response. Exception: {ex}")
            return None

    def get_token_count(self, text: str, model_name: str = "gpt-4o") -> int:
        """Get the token count of a text."""
        enc = tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)
        return len(tokens)


############################################################################################################
################################# Utilities for Video Content Understanding ################################
############################################################################################################

# Define Analyzer management paths
PATH_ANALYZER_MANAGEMENT = "/analyzers/{analyzerId}"
PATH_ANALYZER_MANAGEMENT_OPERATION = "/analyzers/{analyzerId}/operations/{operationId}"

# Define Analyzer inference paths
PATH_ANALYZER_INFERENCE = "/analyzers/{analyzerId}:analyze"
PATH_ANALYZER_INFERENCE_GET_IMAGE = "/analyzers/{analyzerId}/results/{operationId}/images/{imageId}"


class VideoContentUnderstandingClient:
    """Azure AI Video Content Understanding client"""

    def __init__(self,
                 endpoint: str,
                 analyzer_path: str,
                 api_version: str = "2024-12-01-preview",
                 api_key: str = None) -> None:
        self._endpoint = endpoint + "/contentunderstanding" \
            if "azure.com/contentunderstanding" not in endpoint else endpoint
        self._api_version = api_version if api_version else "2024-12-01-preview"
        self._api_key = api_key
        self._operation_id = None

        if self._api_key is None or self._api_key == "":
            print("Using Entra ID/AAD to authenticate")
            self._token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default")
        self._analyzer = self.load_analyzer_from_file(analyzer_path)
        self.delete_analyzer(self._analyzer)
        self.create_analyzer(self._analyzer)

    @staticmethod
    def load_analyzer_from_file(template_file: str) -> dict:
        """Load an analyzer configuration from a JSON file."""
        with open(template_file) as json_file:
            analyzer = json.load(json_file)

        if "scenario" in analyzer:
            trimmed_config = {}
            for key, value in analyzer["config"].items():
                if not key.startswith("_"):
                    trimmed_config[key] = value

            analyzer["config"] = trimmed_config
        analyzer["analyzerId"] = analyzer["analyzerId"] + "_" + str(uuid.uuid4())
        return analyzer

    def _get_headers(self, apim_subid: str = "functest"):
        headers = {
            "Content-Type": "application/json",
            "cogsvc-videoanalysis-face-identification-enable": "true",
            "apim-subscription-id": apim_subid
        }

        if self._api_key is None or self._api_key == "":
            headers["Authorization"] = f"Bearer {self._token_provider()}"
        else:
            headers["Ocp-Apim-Subscription-Key"] = self._api_key

        return headers

    def cleanup(self):
        print("deleting analyzer " + self._analyzer["analyzerId"])
        self.delete_analyzer(self._analyzer)

    def delete_analyzer(self, analyzer_config: str):
        """Delete an analyzer using the analyzer id."""
        headers = self._get_headers()
        print(
            f"DELETE {self._endpoint + PATH_ANALYZER_MANAGEMENT.format(analyzerId=analyzer_config['analyzerId'])}"
        )

        response = requests.delete(
            self._endpoint + PATH_ANALYZER_MANAGEMENT.format(
                analyzerId=analyzer_config["analyzerId"]) +
            f"?api-version={self._api_version}",
            headers=headers)
        if ('request-id' in response.headers):
            print(f"request-id: {response.headers['request-id']}")
        print(response)

    def create_analyzer(self, analyzer_config: str):
        """Create a new analyzer using the provided configuration."""
        print(f"Creating analyzer with id: {analyzer_config['analyzerId']}")
        headers = self._get_headers()
        print(
            f"PUT {self._endpoint + PATH_ANALYZER_MANAGEMENT.format(analyzerId=analyzer_config['analyzerId'])}"
        )

        response = requests.put(self._endpoint +
                                PATH_ANALYZER_MANAGEMENT.format(
                                    analyzerId=analyzer_config["analyzerId"]) +
                                f"?api-version={self._api_version}",
                                headers=headers,
                                json=analyzer_config)
        if ('request-id' in response.headers):
            print(f"request-id: {response.headers['request-id']}")
        print(response)

        operation_location = response.headers.get('Operation-Location')
        final_state = self.poll_for_results(operation_location, 'succeeded',
                                            'failed')

    def poll_for_results(self,
                         operation_location: str,
                         success_state: str = 'succeeded',
                         failed_state: str = 'failed',
                         timeout: int = 60,
                         interval: int = 2):
        """
        Polls the operation location URL until the operation reaches a success or failure state.

        Args:
            operation_location (str): The URL to poll for the operation result.
            success_state (str): The status indicating the operation succeeded.
            failed_state (str): The status indicating the operation failed.
            timeout (int, optional): Maximum time to wait in seconds. Default is 60 seconds.
            interval (int, optional): Time between polling attempts in seconds. Default is 2 seconds.

        Returns:
            dict or None: The final JSON response if successful, None otherwise.
        """
        headers = self._get_headers()

        print(f'GET {operation_location}')

        elapsed_time = 0
        while elapsed_time <= timeout:
            try:
                response = requests.get(operation_location, headers=headers)
                response.raise_for_status()
                result = response.json()
                print(result)

                status = result.get('status').lower()
                if status == success_state:
                    return result
                elif status == failed_state:
                    print(f"Operation failed with status: {status}")
                    return None

                time.sleep(interval)
                elapsed_time += interval

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                return None

        print("Operation timed out.")
        return None

    # Helper function to run inference
    def run_inference(self, analyzer_id: str, video_url: str):
        """
        Runs inference using the specified analyzer ID.

        Args:
            analyzer_id (str): The ID of the analyzer or prebuilt model.
            video_url (str): The video url or SAS link to analyze.

        Returns:
            dict: The result of the analysis.
        """
        headers = self._get_headers()
        payload = {"url": video_url}
        # Construct the request URL
        url = f"{self._endpoint}{PATH_ANALYZER_INFERENCE.format(analyzerId=analyzer_id)}?api-version={self._api_version}"
        print(f"Processing file: {video_url} with analyzer ID: {analyzer_id}")
        print(f"Request URL: POST {url}")
        try:
            # Make the POST request
            response = requests.post(url, headers=headers, json=payload)
            if ('request-id' in response.headers):
                print(f"request-id: {response.headers['request-id']}")
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Get the operation location from the response headers
            operation_location = response.headers.get('Operation-Location')
            if not operation_location:
                print(
                    "Error: 'Operation-Location' not found in the response headers."
                )
                return None

            # Get the operation ID from the response headers
            operation_id = response.headers.get("request-id")
            if not operation_id:
                print("Error: 'request-id' not found in the response headers.")
                return None
            self._operation_id = operation_id

            # Poll for results
            result = self.poll_for_results(operation_location,
                                           'succeeded',
                                           'failed',
                                           timeout=3600)
            return result

        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            return None

    def submit_video(self, video_url: str):
        """
        Submit the post request for the provided video url.

        Args:
            video_url (str): The video url or SAS link to analyze.

        Returns:
            operation_location: Operation-Location in the header
            request_id: id in the response body
        """
        headers = self._get_headers()
        payload = {"url": video_url}
        # Construct the request URL
        analyzer_id = self._analyzer["analyzerId"]
        url = f"{self._endpoint}{PATH_ANALYZER_INFERENCE.format(analyzerId=analyzer_id)}?api-version={self._api_version}"
        print(f"Processing file: {video_url} with analyzer ID: {analyzer_id}")
        print(f"Request URL: POST {url}")
        try:
            # Make the POST request
            response = requests.post(url, headers=headers, json=payload)
            if ('request-id' in response.headers):
                print(f"request-id: {response.headers['request-id']}")
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Get the operation location from the response headers
            operation_location = response.headers.get('Operation-Location')
            if not operation_location:
                print(
                    "Error: 'Operation-Location' not found in the response headers."
                )
                return None

            return operation_location

        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")

            if e.response != None:
                error = json.loads(e.response.content)
                error_content = error.get('error', {})
                print(f"HTTP request failed with {ErrorMessage.from_json(error_content)}")

            return None

    # Function to run analyzer inference
    def run_analyzer_inference(self, file_path: str):
        """
        Runs analyzer inference using a custom analyzer configuration.

        Args:
            analyzer_config: An object containing the analyzer configuration.
            file_path (str): The path to the file to analyze.

        Returns:
            dict: The result of the analysis.
        """
        analyzer_id = self._analyzer["analyzerId"]
        result = self.run_inference(analyzer_id, file_path)
        if result:
            print(result)

        return result

    # Helper function to get image result
    def get_image_from_most_recent_operation(self, image_id: str):
        """
        Runs inference using the specified analyzer ID.

        Args:
            image_id (str): The ID of the images referred to in the video description result. (e.g., 'face.P1000N' for face thumbnails, 'keyFrame.1900' for key frames)

        Returns:
            str: The image.
        """
        if self._operation_id is None:
            print("Error: Missing most recent operation ID. Please process a video first.")
            return None
        analyzer_id = self._analyzer["analyzerId"]
        headers = self._get_headers()
        # Construct the request URL
        url = f"{self._endpoint}{PATH_ANALYZER_INFERENCE_GET_IMAGE.format(analyzerId=analyzer_id, operationId=self._operation_id, imageId=image_id)}?api-version={self._api_version}"
        print(
            f"Getting image {image_id} from operation {self._operation_id} with analyzer ID: {analyzer_id}"
        )
        print(f"Request URL: GET {url}")
        try:
            # Make the GET request
            response = requests.get(url, headers=headers)
            if "apim-request-id" in response.headers:
                print(f"request-id: {response.headers['apim-request-id']}")
            response.raise_for_status()  # Raise an exception for HTTP errors

            assert response.headers.get("Content-Type") == "image/jpeg"

            return response.content

        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            return None

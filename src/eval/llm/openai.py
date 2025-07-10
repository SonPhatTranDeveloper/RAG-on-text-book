from src.eval.llm.base import BaseExtractor
from openai import OpenAI, OpenAIError
from typing import Optional, List, Dict
import json
import os


class OpenAIExtractor(BaseExtractor):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initializes the OpenAIExtraction class.

        Args:
            api_key (Optional[str]): Your OpenAI API key. If None, it attempts
                                     to read from the OPENAI_API_KEY environment variable.
            model (str): The OpenAI model to use for extraction (default: "gpt-4o").
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it or set the "
                "OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # Define the tool/function schema for extracting a list of quiz question details
        self.quiz_questions_tool_schema = {
            "type": "function",
            "function": {
                "name": "extract_quiz_questions_details",
                "description": "Extracts a list of structured multiple-choice questions and their single-letter answers from HTML content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "A list of multiple-choice questions, each with its content and correct single-letter answer.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The full content of the multiple-choice question, including options. Any mathematical symbols or expressions should be converted to LaTeX format. No question number is needed. Put newline symbol between options. Make sure each question is formatted the same way.",
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "The single-letter answer (e.g., 'A', 'B', 'C', or 'D') corresponding to the correct option.",
                                        "enum": ["A", "B", "C", "D"],
                                    },
                                },
                                "required": ["question", "answer"],
                            },
                        }
                    },
                    "required": ["questions"],
                },
            },
        }

    def _generate_extraction_prompt(self, html_content: str) -> list[dict]:
        """
        Generates the prompt messages for the OpenAI API call.

        The prompt instructs the LLM to extract a list of multiple-choice questions and
        their answers from the provided HTML and use the 'extract_quiz_questions_details' function.

        Args:
            html_content (str): The HTML content containing the quiz questions.

        Returns:
            list[dict]: A list of message dictionaries for the OpenAI API.
        """
        system_message = (
            "You are an expert AI assistant tasked with extracting a list of structured "
            "multiple-choice questions and their answers from HTML content. "
            "When presented with HTML, you must call the 'extract_quiz_questions_details' "
            "tool with the relevant information. Ensure each 'answer' is a single capital letter. "
            "IMPORTANT: Convert any mathematical symbols or expressions in the question text to LaTeX format."
        )

        user_message = (
            "Please extract all multiple-choice questions and their single-letter answers from the following HTML:\n\n"
            f"```html\n{html_content}\n```\n\n"
            "Please ensure each 'answer' is only the single correct letter (A, B, C, or D).\n"
            "Also, you must convert any mathematical symbols accurately in the questions to LaTeX format, surround the symbols with $.\n"
            "Do not include any question that include a drawing or image."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def extract(self, html_content: str) -> List[Dict[str, str]]:
        """
        Extracts a list of multiple-choice questions and their answers from HTML using the OpenAI API's
        function calling capability.

        Args:
            html_content (str): Quiz questions content as HTML.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary has "question" and "answer" keys.

        Raises:
            ValueError: If the API key is missing, the API call fails, LLM returns invalid data,
                        or the LLM does not call the expected function.
        """
        if not html_content:
            print("Warning: Empty html_content provided.")
            return []

        messages = self._generate_extraction_prompt(html_content)
        function_args_str = ""  # Initialize for error reporting

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                tools=[self.quiz_questions_tool_schema],  # type: ignore
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_quiz_questions_details"},
                },  # Force the model to call this specific function
            )

            # Check if the model actually called a tool
            tool_calls = completion.choices[0].message.tool_calls
            if (
                not tool_calls
                or tool_calls[0].function.name != "extract_quiz_questions_details"
            ):
                raise ValueError(
                    "The LLM did not call the 'extract_quiz_questions_details' function as expected."
                )

            # Extract the arguments from the tool call
            function_args_str = tool_calls[0].function.arguments

            # Parse the JSON string from the tool call arguments
            extracted_data = json.loads(function_args_str)

            # Expecting a 'questions' key which is a list of question-answer objects
            questions_list = extracted_data.get("questions", [])
            if not isinstance(questions_list, list):
                raise ValueError("Expected 'questions' to be a list in LLM's response.")

            processed_questions = []
            for item in questions_list:
                if not isinstance(item, dict):
                    print(
                        f"Warning: Skipped non-dictionary item in questions list: {item}"
                    )
                    continue

                question = item.get("question", "")
                answer = item.get("answer", "")

                if not question or not answer:
                    print(
                        f"Warning: Skipped item with missing 'question' or 'answer': {item}"
                    )
                    continue
                if len(answer) != 1 or answer.upper() not in ["A", "B", "C", "D"]:
                    print(f"Warning: Skipped item with invalid 'answer' format: {item}")
                    continue

                processed_questions.append(
                    {
                        "question": question,
                        "answer": answer.upper(),  # Ensure answer is always uppercase
                    }
                )

            return processed_questions

        except OpenAIError as e:
            print(f"An OpenAI API error occurred: {e}")
            raise ValueError(f"OpenAI API error: {e}") from e
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM function call arguments: {e}")
            print(f"LLM Raw function arguments: {function_args_str}")
            raise ValueError(
                f"LLM function call arguments are not valid JSON: {e}"
            ) from e
        except ValueError as e:
            raise e
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
            raise ValueError(f"An unexpected error occurred: {e}") from e

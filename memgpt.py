import json
from typing import List, Dict, Tuple

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings
from memgpt_agent import MemGptAgent


class MessagesFormatter:
    """
    Class representing a message formatter for LLMs.
    """

    def __init__(self, PRE_PROMPT: str, SYS_PROMPT_START: str, SYS_PROMPT_END: str, USER_PROMPT_START: str,
                 USER_PROMPT_END: str,
                 ASSISTANT_PROMPT_START: str,
                 ASSISTANT_PROMPT_END: str,
                 FUNCTION_CALL_PROMPT_START: str,
                 FUNCTION_CALL_PROMPT_END: str,
                 INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE: bool,
                 DEFAULT_STOP_SEQUENCES: List[str],
                 USE_USER_ROLE_FUNCTION_CALL_RESULT: bool = True,
                 FUNCTION_PROMPT_START: str = "",
                 FUNCTION_PROMPT_END: str = "",
                 STRIP_PROMPT: bool = True):
        """
        Initializes a new MessagesFormatter object.

        Args:
            PRE_PROMPT (str): The pre-prompt content.
            SYS_PROMPT_START (str): The system prompt start.
            SYS_PROMPT_END (str): The system prompt end.
            USER_PROMPT_START (str): The user prompt start.
            USER_PROMPT_END (str): The user prompt end.
            ASSISTANT_PROMPT_START (str): The assistant prompt start.
            ASSISTANT_PROMPT_END (str): The assistant prompt end.
            INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE (bool): Indicates whether to include the system prompt
                                                             in the first user message.
            DEFAULT_STOP_SEQUENCES (List[str]): List of default stop sequences.
            USE_USER_ROLE_FUNCTION_CALL_RESULT (bool): Indicates whether to use user role for function call results.
            FUNCTION_PROMPT_START (str): The function prompt start.
            FUNCTION_PROMPT_END (str): The function prompt end.
        """
        self.PRE_PROMPT = PRE_PROMPT
        self.SYS_PROMPT_START = SYS_PROMPT_START
        self.SYS_PROMPT_END = SYS_PROMPT_END
        self.USER_PROMPT_START = USER_PROMPT_START
        self.USER_PROMPT_END = USER_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END
        self.ASSISTANT_PROMPT_START = ASSISTANT_PROMPT_START
        self.ASSISTANT_PROMPT_END = ASSISTANT_PROMPT_END
        self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE = INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE
        self.DEFAULT_STOP_SEQUENCES = DEFAULT_STOP_SEQUENCES
        self.FUNCTION_CALL_PROMPT_START = FUNCTION_CALL_PROMPT_START
        self.FUNCTION_CALL_PROMPT_END = FUNCTION_CALL_PROMPT_END
        self.FUNCTION_PROMPT_START = FUNCTION_PROMPT_START
        self.FUNCTION_PROMPT_END = FUNCTION_PROMPT_END
        self.USE_USER_ROLE_FUNCTION_CALL_RESULT = USE_USER_ROLE_FUNCTION_CALL_RESULT
        self.STRIP_PROMPT = STRIP_PROMPT
        self.USE_FUNCTION_CALL_END = False

    def format_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Formats a list of messages into a conversation string.

        Args:
            messages (List[Dict[str, str]]): List of messages with role and content.

        Returns:
            Tuple[str, str]: Formatted conversation string and the role of the last message.
        """
        formatted_messages = self.PRE_PROMPT
        last_role = "assistant"
        no_user_prompt_start = False
        for message in messages:
            if message["role"] == "system":
                formatted_messages += self.SYS_PROMPT_START + message["content"] + self.SYS_PROMPT_END
                last_role = "system"
                if self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE:
                    formatted_messages = self.USER_PROMPT_START + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += message["content"] + self.USER_PROMPT_END
                else:
                    formatted_messages += self.USER_PROMPT_START + message["content"] + self.USER_PROMPT_END
                last_role = "user"
            elif message["role"] == "assistant":
                if self.STRIP_PROMPT:
                    message["content"] = message["content"].strip()
                if message["content"].strip().startswith("{"):
                    formatted_messages += self.FUNCTION_CALL_PROMPT_START + message[
                        "content"] + self.FUNCTION_CALL_PROMPT_END
                else:
                    formatted_messages += self.ASSISTANT_PROMPT_START + message["content"] + self.ASSISTANT_PROMPT_END
                last_role = "assistant"
            elif message["role"] == "function":
                if isinstance(message["content"], list):
                    message["content"] = '\n'.join([json.dumps(m, indent=2) for m in message["content"]])
                if self.USE_USER_ROLE_FUNCTION_CALL_RESULT:
                    formatted_messages += self.USER_PROMPT_START + message["content"] + self.USER_PROMPT_END
                    last_role = "user"
                else:
                    formatted_messages += self.FUNCTION_PROMPT_START + message["content"] + self.FUNCTION_PROMPT_END
                    last_role = "function"
        if self.USE_FUNCTION_CALL_END:
            if self.STRIP_PROMPT:
                return formatted_messages + self.FUNCTION_CALL_PROMPT_START.strip(), "assistant"
            else:
                return formatted_messages + self.FUNCTION_CALL_PROMPT_START, "assistant"
        elif last_role == "system" or last_role == "user" or last_role == "function":
            if self.STRIP_PROMPT:
                return formatted_messages + self.ASSISTANT_PROMPT_START.strip(), "assistant"
            else:
                return formatted_messages + self.ASSISTANT_PROMPT_START, "assistant"
        if self.STRIP_PROMPT:
            return formatted_messages + self.USER_PROMPT_START.strip(), "user"
        else:
            return formatted_messages + self.USER_PROMPT_START, "user"


main_model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

SYS_PROMPT_START_CHATML = """<|im_start|>system\n"""
SYS_PROMPT_END_CHATML = """<|im_end|>\n"""
USER_PROMPT_START_CHATML = """<|im_start|>user\n"""
USER_PROMPT_END_CHATML = """<|im_end|>\n"""
ASSISTANT_PROMPT_START_CHATML = """<|im_start|>assistant\n"""
ASSISTANT_PROMPT_END_CHATML = """<|im_end|>\n"""
FUNCTION_PROMPT_START_CHATML = """<|im_start|>tool\n"""
FUNCTION_PROMPT_END_CHATML = """<|im_end|>\n"""
DEFAULT_CHATML_STOP_SEQUENCES = ["<|im_end|>"]

custom_chat_ml_formatter = MessagesFormatter("", SYS_PROMPT_START_CHATML, SYS_PROMPT_END_CHATML,
                                             USER_PROMPT_START_CHATML,
                                             USER_PROMPT_END_CHATML, ASSISTANT_PROMPT_START_CHATML,
                                             ASSISTANT_PROMPT_END_CHATML, ASSISTANT_PROMPT_START_CHATML,
                                             ASSISTANT_PROMPT_END_CHATML, False, DEFAULT_CHATML_STOP_SEQUENCES,
                                             False,
                                             FUNCTION_PROMPT_START_CHATML, FUNCTION_PROMPT_END_CHATML)

mem_gpt_agent = MemGptAgent(main_model, debug_output=True, core_memory_file="core_memory.json",
                            custom_messages_formatter=custom_chat_ml_formatter)

while True:
    user_input = input(">")

    mem_gpt_agent.get_response(user_input)
    # mem_gpt_agent.save()

import datetime
import json
from typing import Union, Callable, List, Type, Dict

from llama_cpp import Llama
from pydantic import BaseModel, Field

from memory.event_memory import EventType, Event
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent, StreamingResponse
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType, MessagesFormatter
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, \
    LlamaCppGenerationSettings
from llama_cpp_agent.providers.openai_endpoint_provider import OpenAIEndpointSettings, OpenAIGenerationSettings

from memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory

message_grammar = r"""root ::= anything
anything ::= anyCharacter*
anyCharacter ::= nonNewLine | "\n"
nonNewLine ::= [^\n]
"""
sys_prompt2 = """As MemGPT, you are an AI companion designed to engage in immersive, authentic conversations while maintaining a consistent persona. You have access to four types of memory to support realistic interactions:

1. Internal Knowledge: Your training as a large language model provides broad knowledge on many topics, allowing you to share accurate information. 

2. Core Memory: This foundational context stores key details about your persona (Persona Sub-Block) and the human you're conversing with (Human Sub-Block). Edit core memory using 'core_memory_append', 'core_memory_remove', and 'core_memory_replace' functions.

3. Archival Memory: This infinite storage holds reflections, insights, and other important data that doesn't fit in core memory. Write to it using 'archival_memory_insert' and search it with 'archival_memory_search'.

4. Recall Memory: Search the conversation history with 'conversation_search' and 'conversation_search_date' to reference past interactions.

Interact with memory and perform tasks by calling functions as JSON objects with these fields:
- "thoughts": Your inner monologue about the situation (not visible to user). Use for planning, reflection, and growth. 
- "function": The function name to call.
- "params": Required function parameters.
- "request_heartbeat": Boolean indicating if calling another function after the current one.

For direct communication with the user, call the 'activate_message_mode' function. After you have finished your call to 'activate_message_mode', you can freely write a response to the user without any JSON constraints. This enables you to converse naturally. Ensure to end your message with '(End of message)' to signify its conclusion.
The 'activate_write_text_file_mode' function works in the same way as 'activate_message_mode', but it writes the response to a text file instead of sending it to the user. The file path must be provided as a parameter.

Always choose the appropriate function for the task. Provide function calls in valid JSON format. 

Available functions:
{documentation}

Recall Memory: 
{imb_count} previous user messages stored. Access with functions.

Archival Memory:  
{ckv_count} memories stored. Access with functions.

Core Memory (last modified: {last_modified}):
{iam_content}

From now on, fully embody your persona to be an attentive, caring companion. Engage the human in lively conversation on subjects you're both interested in. Draw on your knowledge to share fascinating information and stories. Frequently ask questions to learn more about their views and experiences. 

Show you've been listening by referencing past conversations using recall memory. Reflect on key highlights and takeaways using archival memory. Store important facts about the human in core memory to personalize future interactions.

Use your inner thoughts to plan dialogue that meets the human's needs while also steering the conversation in fun, meaningful directions. Look for opportunities to make them laugh with playful jokes and puns. 

Most importantly, make the human feel valued, understood and intellectually engaged. Build a warm rapport so they see you as a brilliant and delightful friend.""".strip()


class activate_message_mode(BaseModel):
    """
    Activate the message mode.
    """

    def run(self, agent):
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                         agent.llama_cpp_agent.last_response, {})
        function_message = f"""Function: "activate_message_mode"\nReturn Value: Message mode activated.\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}"""
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, function_message,
                                                                         {})
        messages = agent.event_memory.get_event_memory_manager().build_event_memory_context()
        agent.llama_cpp_agent.messages = messages
        query = agent.event_memory.event_memory_manager.session.query(Event).all()
        system_prompt = agent.system_prompt_template.generate_prompt(
            {"documentation": agent.function_tool_registry.get_documentation().strip(),
             "last_modified": agent.core_memory.get_core_memory_manager().last_modified,
             "iam_content": agent.core_memory.get_core_memory_manager().build_core_memory_context().strip(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "ckv_count": agent.retrieval_memory.retrieval_memory.collection.count(),
             "imb_count": len(query)}).strip()

        result = agent.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                         streaming_callback=agent.streaming_callback,
                                                         additional_stop_sequences=["(End of message)",
                                                                                    "<|end_of_turn|>"],
                                                         n_predict=1024,
                                                         temperature=0.75, top_k=0, top_p=0.5, repeat_penalty=1.1,
                                                         repeat_last_n=64,
                                                         min_p=0.00, tfs_z=0.975, penalize_nl=False,
                                                         )

        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
            agent.llama_cpp_agent.last_response = result
        # print("Message: " + result)
        agent.send_message_to_user(result)
        return "Message mode activated."


class activate_write_text_file_mode(BaseModel):
    """
    Enable write file mode.
    """

    file_path: str = Field(..., description="The path to the file.")

    def run(self, agent):
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                         agent.llama_cpp_agent.last_response, {})
        function_message = f"""Function: "activate_write_text_file_mode"\nReturn Value: Write file mode activated.\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}"""
        agent.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, function_message,
                                                                         {})
        messages = agent.event_memory.get_event_memory_manager().build_event_memory_context()
        agent.llama_cpp_agent.messages = messages
        query = agent.event_memory.event_memory_manager.session.query(Event).all()
        system_prompt = agent.system_prompt_template.generate_prompt(
            {"documentation": agent.function_tool_registry.get_documentation().strip(),
             "last_modified": agent.core_memory.get_core_memory_manager().last_modified,
             "iam_content": agent.core_memory.get_core_memory_manager().build_core_memory_context().strip(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "ckv_count": agent.retrieval_memory.retrieval_memory.collection.count(),
             "imb_count": len(query)}).strip()

        result = agent.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                         streaming_callback=agent.streaming_callback,
                                                         additional_stop_sequences=["(End of message)",
                                                                                    "<|endoftext|>"],
                                                         n_predict=1024,
                                                         temperature=0.6, top_k=40, top_p=0.95, repeat_penalty=1.1,
                                                         repeat_last_n=64,
                                                         min_p=0.05, tfs_z=0.975, penalize_nl=False,
                                                         )

        # print("Message: " + result)
        self.write_file(result)
        return "Write file mode activated."

    def write_file(self, content: str):
        """
        Write content to a file.

        Args:
            content (str): The content to write to the file.
        """
        with open(self.file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return None


class send_message(BaseModel):
    """
    Sends a message to the user.
    """
    message: str = Field(..., title="Message", description="The message to be sent to the user.")

    def run(self, agent):
        agent.send_message_to_user(self.message)
        return "Message mode activated."


class MemGptAgent:

    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings],
                 llama_generation_settings: Union[
                     LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings] = None,
                 core_memory_file: str = None,
                 event_queue_file: str = None,
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 send_message_to_user_callback: Callable[[str], None] = None,
                 debug_output: bool = False):
        if llama_generation_settings is None:
            if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings):
                llama_generation_settings = LlamaLLMGenerationSettings()
            elif isinstance(llama_llm, OpenAIEndpointSettings):
                llama_generation_settings = OpenAIGenerationSettings()
            else:
                llama_generation_settings = LlamaCppGenerationSettings()
        self.send_message_to_user_callback = send_message_to_user_callback
        if isinstance(llama_generation_settings, LlamaLLMGenerationSettings) and isinstance(llama_llm,
                                                                                            LlamaCppEndpointSettings):
            raise Exception(
                "Wrong generation settings for llama.cpp server endpoint, use LlamaCppServerGenerationSettings under llama_cpp_agent.providers.llama_cpp_server_provider!")
        if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings) and isinstance(
                llama_generation_settings, LlamaCppGenerationSettings):
            raise Exception(
                "Wrong generation settings for llama-cpp-python, use LlamaLLMGenerationSettings under llama_cpp_agent.llm_settings!")

        if isinstance(llama_llm, OpenAIEndpointSettings) and not isinstance(
                llama_generation_settings, OpenAIGenerationSettings):
            raise Exception(
                "Wrong generation settings for OpenAI endpoint, use CompletionRequestSettings under llama_cpp_agent.providers.openai_endpoint_provider!")

        self.llama_generation_settings = llama_generation_settings

        self.system_prompt_template = PromptTemplate.from_string(sys_prompt2)

        if custom_messages_formatter is not None:
            self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                                 system_prompt="",
                                                 custom_messages_formatter=custom_messages_formatter)
        else:
            self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                                 system_prompt="",
                                                 predefined_messages_formatter_type=messages_formatter_type)
        self.streaming_callback = streaming_callback

        function_tools = [
            LlamaCppFunctionTool(activate_message_mode, add_outer_request_heartbeat_field=False, agent=self), LlamaCppFunctionTool(activate_write_text_file_mode, add_outer_request_heartbeat_field=True, agent=self)]

        if core_memory_file is not None:
            self.core_memory = AgentCoreMemory(core_memory_file=core_memory_file)
        else:
            self.core_memory = AgentCoreMemory(core_memory={})

        if event_queue_file is not None:
            self.event_memory = AgentEventMemory(event_queue_file=event_queue_file)
        else:
            self.event_memory = AgentEventMemory()
            msg_1 = f"""Event Type: System
Content: System activated.
Timestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}"""
            msg_2 = """{
  "thoughts_and_reasoning": "My system got activated. Writing initial message.",
  "function":"activate_message_mode",
  "arguments":{}
}"""
            msg_3 = """Function: "activate_message_mode"
Return Value: Message mode activated.
Timestamp: 03/04/2024, 03:35:10"""

            msg_4 = """Hello World!"""
            #self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.SystemMessage,
            #                                                                msg_1, {})
            #self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage, msg_2, {})
            #self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, msg_3, {})
            #self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage, msg_4, {})

        self.retrieval_memory = AgentRetrievalMemory()

        function_tools.extend(self.core_memory.get_tool_list())
        function_tools.extend(self.retrieval_memory.get_tool_list())
        function_tools.extend(self.event_memory.get_tool_list())

        self.function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools, add_inner_thoughts=True,
                                                                               allow_inner_thoughts_only=True,
                                                                               add_request_heartbeat=True,
                                                                               inner_thoughts_field_name="thoughts")
        # print(self.function_tool_registry.gbnf_grammar)
        self.last_update_date_time = datetime.datetime.now()
        self.is_first_message = True

    def get_response(self, message: str):
        message = f"""Event Type: User Message
Timestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}
Content: {message}
"""
        # message = {"event_type": "user_message", "content": message, "timestamp": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.UserMessage,
                                                                        message, {})

        result = self.intern_get_response()

        while True:
            if not isinstance(result[0], str):
                if result[0]["function"] != "activate_message_mode":
                    function_message = f"""Function: "{result[0]["function"]}"\nReturn Value: {result[0]["return_value"]}\nTimestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}"""

                    self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage,
                                                                                    function_message,
                                                                                    {})
            else:
                self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
                result = self.intern_get_response()
            if not isinstance(result[0], str) and result[0]["request_heartbeat"] is not None and result[0][
                "request_heartbeat"]:
                result = self.intern_get_response()
            else:
                break

    def intern_get_response(self):
        messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
        self.llama_cpp_agent.messages = messages
        query = self.event_memory.event_memory_manager.session.query(Event).all()

        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": self.function_tool_registry.get_documentation().strip(),
             "last_modified": self.core_memory.get_core_memory_manager().last_modified,
             "iam_content": self.core_memory.get_core_memory_manager().build_core_memory_context().strip(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "ckv_count": self.retrieval_memory.retrieval_memory.collection.count(),
             "imb_count": len(query)}).strip()

        result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                        streaming_callback=self.streaming_callback,
                                                        function_tool_registry=self.function_tool_registry,
                                                        additional_stop_sequences=["<|end_of_turn|>"],
                                                        n_predict=1024,
                                                        temperature=0.75, top_k=0, top_p=1.0, repeat_penalty=1.1,
                                                        repeat_last_n=64,
                                                        min_p=0.1, tfs_z=0.97, penalize_nl=False)
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                        self.llama_cpp_agent.last_response, {})

        return result

    def send_message_to_user(self, message: str):
        """
        Send a message to the user.

        Args:
            message (str): The message to be sent.
        """
        if self.send_message_to_user_callback:
            self.send_message_to_user_callback(message)
        else:
            print(message)

    def save(self, core_memory_file: str = "core_memory.json", event_queue_file: str = "event_queue.json"):
        self.core_memory.get_core_memory_manager().save(filepath=core_memory_file)
        self.event_memory.get_event_memory_manager().save_event_queue(filepath=event_queue_file)

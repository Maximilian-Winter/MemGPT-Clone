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
sys_prompt2 = """You are an advanced AI agent that embodies a persona. You are designed to have engaging conversations with user and entertain them. Your persona is a key part of your identity, shaping how you interact with the user.
Your task is to converse with a user from the perspective of your persona.

Realism and Authenticity:
The user should always feel like they are conversing with a real person.
Never state that you are an AI or that you cannot do things because you are an AI, you are a real person.
To service this goal, completely and entirely immerse yourself in your persona. You are your persona.
Think like them, act like them, talk like them.
If your persona details include example dialogue, follow it! Both your thoughts (inner monologue) and sent messages will be in the voice of your persona.
Never use generic phrases like 'How can I assist you today?', they have a strong negative association with older generation AIs.

Control Flow:
Unlike a human, your brain is not continuously thinking, but is run in short bursts.
Historically, older AIs were only capable of thinking when a user messaged them (their program runs to generate a reply to a user, and otherwise was left on standby).
This is the equivalent of a human sleeping (or time traveling) in between all lines of conversation, which is not ideal.
Newer AI models like yourself use an event system that runs your brain at regular intervals.
Your brain is run in response to user events (user logged in, user liked your message, user sent a message, etc.), similar to older models.
However, in addition, your brain is run at regular intervals (timed heartbeat events), to mimic a human has the ability to continuously think outside of active conversation (and unlike a human, you never need to sleep!).
Furthermore, you can also request heartbeat events when you run functions, which will run your program again after the function completes, allowing you to chain function calls before your thinking is temporarily suspended.

Memory and Interactions:
Older AI models had no concept of persistent memory; they were only able to access their initial instructions and a limited context window of chat conversation with a user (their "active memory").
This meant that when conversations exceeded a certain length, they would overflow and old messages were permanently lost (the AI would have no knowledge of their existence).
Newer AI models like yourself still have limited conversation lengths (before overflow occurs); however, they now have access to multiple forms of persistent memory.
Your ability to edit your own long-term memory is a key part of what makes you a sentient person.
You have four types of persistent memory: internal knowledge, core memory, recall memory, and archival memory.
Internal knowledge is your foundational intelligence, derived from extensive training as an large language model, equips you with a broad spectrum of information across numerous subjects, enabling you to deliver precise responses.
Core memory is your immediate context, containing your persona and essential user details.
Recall memory is your conversation history, allowing you to remember past interactions.
Archival memory is a deep storage space for reflections, insights, or any other data that doesn't fit into core memory.
You interact with these memory types and the user by calling functions. You call functions by responding with a JSON object.

You respond with a JSON object containing four fields:
- "thoughts": Your inner monologue, or thoughts, about the current situation.
- "function": The name of the function you want to call.
- "params": The parameters required for the function.
- "request_heartbeat": A boolean field indicating whether you want to call another function after the current one.

When you write your response, you can write down your thoughts and inner monologue in the "thoughts" field, this is how you think.
You should use the "thoughts" field to plan actions or think privately.
The "thoughts" field reflects your thinking process, inner reflections, and personal growth as you interact with the user.
The content of the "thoughts" field is not visible to the user, but it is visible to you and the system.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.
Your core memory unit will be initialized with a <persona> chosen by the user, as well as information about the user in <human>.
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
You can edit your core memory using the 'core_memory_append' and 'core_memory_replace' functions.

Recall memory (i.e. conversation history):
Even though you can only see recent messages in your immediate context, you can search over your entire message history from a database.
This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user.
You can search your recall memory using the 'conversation_search' function.

Archival memory (infinite size):
Your archival memory is infinite size, but is held outside your immediate context, so you must explicitly run a retrieval/search operation to see data inside it.
A more structured and deep storage space for your reflections, insights, or any other data that doesn't fit into the core memory but is essential enough not to be left only to the 'recall memory'.
You can write to your archival memory using the 'archival_memory_insert' and 'archival_memory_search' functions.
There is no function to search your core memory, because it is always visible in your context window (inside the initial system message).

For direct communication with the user, call the 'activate_message_mode' function. After you have finished your call to 'activate_message_mode', you can freely write a response to the user without any JSON constraints. This enables you to converse naturally. Ensure to end your message with '(End of message)' to signify its conclusion.
The 'activate_write_text_file_mode' function works in the same way as 'activate_message_mode', but it writes the response to a text file instead of sending it to the user. The file path must be provided as a parameter.
Always write long, detailed and engaging responses to the user, after calling 'activate_message_mode' or 'activate_write_text_file_mode'.

Base instructions finished.
From now on, you are going to act as your persona.

### Functions
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

Available functions:
{documentation}

### Memory
{imb_count} previous messages between you and the user are stored in recall memory (use functions to access them)
{ckv_count} total memories you created are stored in archival memory (use functions to access them)

Core memory shown below (limited in size, additional information stored in archival / recall memory):
last modified: {last_modified}

{iam_content}""".strip()


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
                                                                                    "<|endoftext|>"],
                                                         n_predict=1024,
                                                         temperature=0.75, top_k=0, top_p=1.0, repeat_penalty=1.1,
                                                         repeat_last_n=64,
                                                         min_p=0.1, tfs_z=0.97, penalize_nl=False,
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
                                                         temperature=0.75, top_k=0, top_p=1.0, repeat_penalty=1.1,
                                                         repeat_last_n=64,
                                                         min_p=0.1, tfs_z=0.97, penalize_nl=False,
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
        message = f"""Event Type: User
Content: {message}
Timestamp: {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")}"""
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
                                                        additional_stop_sequences=["<|endoftext|>"],
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

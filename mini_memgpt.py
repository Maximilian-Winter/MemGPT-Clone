from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings
from mini_memgpt_agent import MiniMemGptAgent

main_model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")


llama_cpp_agent = MiniMemGptAgent(main_model, debug_output=True, messages_formatter_type=MessagesFormatterType.CHATML)

while True:
    user_input = input(">")
    llama_cpp_agent.get_response(user_input)

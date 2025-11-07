import os

import streamlit as st

# from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.outputs import ChatGeneration
# from langchain_core.prompts import PromptTemplate
# from langchain_core.prompts.string import PromptTemplateFormat
# from langchain_openai import ChatOpenAI
from ollama import ChatResponse, Client, chat

# load_dotenv()


# if not os.environ.get("OPENROUTER_API_KEY"):
#     raise ValueError("API_KEY environment variable is not set")

MODEL = "gemma3:1b"
MAXTOKENS = 100
TEMPERATURE = 0.3


class ContextBuilder:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.template = {
            "role": "system",
            "content": f"Build a short term memory using the conversation in {MAXTOKENS} tokens, mention all specific things talked about. output must include all names and details that were mentioned. It should be alpha numeric and punctuation only, no special formatting, keep it as concise as possible, should not be human readable. Keep under 100 words, Output format: Query: "
            "{topic} and answer: {answer}"
            "",
        }
        self.context = ""

    def invoke(self, input):
        """
        Only input the latesthistory
        """

        self.context += (
            "next "
            + self.llm_client.chat(
                model=MODEL,
                messages=[self.template, {"role": "user", "content": input}],
                options={"num_predict": MAXTOKENS, "temperature": TEMPERATURE},
            ).message.content
        )
        print(f"CONTEXT: {self.context}")

    def get(self):
        return self.context


class ChatGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.context = ContextBuilder(llm_client)
        self.template = {
            "role": "system",
            "content": f"Give a response in {MAXTOKENS} tokens to user query given this memory: ",
        }

    def invoke(self, input):
        template = self.template
        template["content"] += self.context.get()
        output = self.llm_client.chat(
            model=MODEL,
            messages=[template, {"role": "user", "content": input}],
            options={"num_predict": MAXTOKENS, "temperature": TEMPERATURE},
        ).message.content
        self.context.invoke("User Asked:" + input + " and Bot replied:" + output)
        return output


st.title("Contextual Chatty bot")
if st.button("reload"):
    # Reset the context builder
    st.session_state.context_builder = ContextBuilder(st.session_state.llm_client)
    st.session_state.messages = []
    st.rerun()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if "llm_client" not in st.session_state:
    st.session_state.llm_client = Client(host="http://localhost:11434")
if "chatgenerator" not in st.session_state:
    st.session_state.chatgenerator = ChatGenerator(st.session_state.llm_client)

chatgenerator = st.session_state.chatgenerator

if userinput := st.chat_input("Have a question boi?"):
    with st.chat_message("user"):
        st.write(userinput)
    st.session_state.messages.append({"role": "user", "content": userinput})
    with st.spinner("prompting..."):
        response = chatgenerator.invoke(userinput)
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

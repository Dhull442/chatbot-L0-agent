import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_openai import ChatOpenAI

load_dotenv()


if not os.environ.get("OPENROUTER_API_KEY"):
    raise ValueError("API_KEY environment variable is not set")


class ContextBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.template = "summarize this conversation include both user and bot roles,conversation: {latesthistory}"
        self.input_variables = ["latesthistory"]
        self.context = ""
        self.llm_prompt = PromptTemplate(
            template=self.template, input_variables=self.input_variables
        )
        self.llm_chain = self.llm_prompt | self.llm | StrOutputParser()

    def invoke(self, input):
        """
        Only input the latesthistory
        """

        self.context += "next " + self.llm_chain.invoke({"latesthistory": input})
        print(f"CONTEXT: {self.context}")

    def get(self):
        return self.context


class ChatGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.context = ContextBuilder(llm)
        self.template = (
            "This is the past converstaion {history}, Give a reponse to {query}."
        )
        self.input_variables = ["history", "query"]
        self.llm_prompt = PromptTemplate(
            template=self.template, input_variables=self.input_variables
        )
        self.llm_chain = self.llm_prompt | self.llm | StrOutputParser()

    def invoke(self, input):
        output = self.llm_chain.invoke({"query": input, "history": self.context.get()})
        self.context.invoke("User Asked:" + input + " and Bot replied:" + output)
        return output


st.title("Contextual Chatty bot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=100,
        temperature=0.7,
    )
if "chatgenerator" not in st.session_state:
    st.session_state.chatgenerator = ChatGenerator(st.session_state.llm)
if st.button("reload"):
    # Reset the context builder
    st.session_state.context_builder = ContextBuilder(st.session_state.llm)
    st.session_state.messages = []
    st.rerun()
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

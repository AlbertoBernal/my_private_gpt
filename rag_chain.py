from typing import Any
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_rag_chain(llm, retriever) -> Any:
    
    condense_q_system_prompt = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    condense_q_chain = condense_q_prompt | llm | StrOutputParser()
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]
    

    def format_docs(docs):
        from pdb import set_trace
        #set_trace()
        print("Docs used") # Print the relevant sources used for the answer
        for document in docs:
            print("\n   >>> " + document.metadata["source"] + ":")
        return "\n\n".join(doc.page_content for doc in docs)
    
    def debug (args):
        from pdb import set_trace
        #set_trace()
        return args
    
    rag_chain_answer = (
        RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
        | debug | qa_prompt
        | llm
    )
  
    return rag_chain_answer
    

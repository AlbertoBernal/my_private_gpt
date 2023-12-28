#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from constants import CHROMA_SETTINGS
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
import os
import argparse
from langchain.prompts import PromptTemplate
from rag_chain import create_rag_chain
from langchain_core.messages import AIMessage, HumanMessage

# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
n_gpu_layers = os.environ.get('GPU_LAYERS')
# Change this value based on your model and your GPU VRAM pool.
n_batch = os.environ.get('MODEL_N_BATCH')
# Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name,
                                               model_kwargs={"device": "cuda"})
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "HuggingFace":
           model = "tiiuae/falcon-40b-instruct"
           tokenizer = AutoTokenizer.from_pretrained(model_path=model_path,
                                                     local_files_only=True)
           pipeline = pipeline(
               "text-generation",  # task
               model_path=model_path,
               tokenizer=tokenizer,
               torch_dtype=torch.bfloat16,
               trust_remote_code=True,
               device_map="auto",
               max_length=200,
               do_sample=True,
               top_k=10,
               num_return_sequences=1,
               eos_token_id=tokenizer.eos_token_id)
           llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
        
        case "LlamaCpp":
            # Make sure the model path is correct for your system!
            llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                temperature=0.1,
                n_ctx=model_n_ctx,
                n_batch=n_batch,
                callbacks=callbacks,
                f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
                n_threads=20,
                max_tokens=8192,
                verbose=args.verbose,)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx,
                          backend='gptj',   callbacks=callbacks,
                          verbose=args.verbose)
        case _:
            print(f"Model {model_type} not supported!")
            exit

    #memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)           
    #qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory) 
    rag_chain = create_rag_chain(llm, retriever)
    chat_history = []
    # qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff",
    #                                                  retriever=retriever,
    #                                                  return_source_documents=not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        ai_msg = rag_chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg)])
        #result = qa(query)
        #answer, docs = result['answer'], [] if args.hide_source else result['sources']
        answer = ai_msg
        # Print the result
        #print("\n\n> Question:")
        #print(query)
        print("\n> Answer:")
        #print(answer)


def parse_arguments():
    """
    Arguments:
       --verbose: put llm in verbose mode
       --mute-stream: llm will put the output at once if True, else output is token by token infered

    """
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    
    parser.add_argument("--verbose", "-v",
                        action='store_true',
                        help='Use this flag to enable verbose mode for LLMs.')
    return parser.parse_args()


if __name__ == "__main__":
    main()

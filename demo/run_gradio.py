import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import SQL_LLM
from langchain.prompts import PromptTemplate

def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    persist_directory = 'data_base/vector_db/chroma'

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    llm = SQL_LLM(model_path = "/root/data/model/meta-llama/Llama-2-7b-chat-hf")

    template = """{context} Task: {question}\nSQL query:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"], template=template)

    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    return qa_chain

class Model_center():
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history

model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>SQL_LLM</center></h1>""")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            msg = gr.Textbox(label="SQL Prompt")

            with gr.Row():
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""
    Examples:
    <br> 1.Name the home team for carlton away team. CREATE TABLE table_name_77 ( home_team VARCHAR, away_team VARCHAR )
    <br> 2.What will the population of Asia be when Latin America/Caribbean is 783 (7.5%)? CREATE TABLE table_22767 ( "Year" real, "World" real, "Asia" text, "Africa" text, "Europe" text, "Latin America/Caribbean" text, "Northern America" text, "Oceania" text )
    """)    
    gr.Markdown("""Notice：loading the vector database may take long.""")

gr.close_all()
demo.launch()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available 
from transformers import BitsAndBytesConfig
import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import textwrap
import torch


pdf_path = "/home/ai/TAC2-lbz/knowledge_base.pdf"

def text_formatter(text:str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

# now for opening the pdf and reading it we want this function:
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4, # 1 token is about 4 words. this will be need for passing to LLM
                                "text": text })
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)



df = pd.DataFrame(pages_and_texts)

nlp = English()

nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])


df = pd.DataFrame(pages_and_texts)


num_sentences_chunk_size = 10

def split_list(input_list:list[str], slice_size:int=num_sentences_chunk_size) -> list[list[str]]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]



for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(item["sentences"])
    item["number_of_chunks"] = len(item["sentence_chunks"])


df= pd.DataFrame(pages_and_texts)

pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # '.A' -> ', A'
        
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in (joined_sentence_chunk.split(" "))])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1  token has 4 chars
        
        pages_and_chunks.append(chunk_dict)

df = pd.DataFrame(pages_and_chunks)

min_token_length = 30
pages_and_chunks_over_min_token_length = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

df = pd.DataFrame(pages_and_chunks_over_min_token_length)


embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

embedding_model.to("cuda") 
for item in tqdm(pages_and_chunks_over_min_token_length):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])


np.shape(pages_and_chunks_over_min_token_length[100]["embedding"])


text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_length)
embedding_df_path= "/home/ai/TAC2-lbz/text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embedding_df_path, index=False)


text_chunks_and_embeddings_df_load = pd.read_csv(embedding_df_path)


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    return wrapped_text


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5):
    query_embedding = model.encode(query, 
                                convert_to_tensor=True) 
    
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    
    scores, indices = torch.topk(input=dot_scores, 
                                k=n_resources_to_return)

    return scores, indices

# importing text and embeddings
text_chunks_and_embeddings_df = pd.read_csv("/home/ai/TAC2-lbz/text_chunks_and_embeddings_df.csv")

# now converting embedding column to a np.array
text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# converting embedding into a torch.tensor
embeddings = torch.tensor(np.stack(text_chunks_and_embeddings_df["embedding"].tolist(), axis=0), dtype=torch.float32).to("cuda")

# converting text and embeddings to the list of dictionaries
pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")

# create model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16)


if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"

model_id = "google/gemma-7b-it" 

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

# model
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: What is the firmware version synchronized with Version 4.0 of the MES5448 and MES7048 operation manual?
    Answer: The firmware version synchronized with Version 4.0 of the MES5448 and MES7048 operation manual is 8.4.0.7​(MES5448_MES7048_user_ma…).
    \nExample 2:
    Query: What is the purpose of the MES5448 and MES7048 Data Center Switches?
    Answer: The purpose of the MES5448 and MES7048 Data Center Switches is to provide high-performance networking solutions for data centers, supporting various interface types and advanced networking features suitable for aggregation and transport in carrier networks and data centers​(MES5448_MES7048_user_ma…).
    \nExample 3:
    Query:List three Layer 2 features supported by the MES5448 switch.
    Answer: Three Layer 2 features supported by the MES5448 switch are: VLAN support (IEEE 802.1Q), Link aggregation (IEEE 802.3ad), Spanning Tree Protocol (STP, RSTP, MSTP)​(MES5448_MES7048_user_ma…).
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""

    base_prompt = base_prompt.format(context=context, query=query)
    template = [
        {"role": "user",
        "content": base_prompt}
    ]
    prompt = tokenizer.apply_chat_template(conversation=template,
                                        tokenize=False,
                                        add_generation_prompt=True)
    return prompt

def answer_to_question(query: str):
    query = query
    scores, indices = retrieve_relevant_resources(query=query,
                                            embeddings=embeddings)
    
    context_items = [pages_and_chunks[i] for i in indices]

    prompt = prompt_formatter(query=query,
                        context_items=context_items)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cpu")

    outputs = llm_model.generate(**input_ids,
                            temperature=0.7, 
                            do_sample=True, 
                            max_new_tokens=256) 

    output_text = tokenizer.decode(outputs[0])
    output_text = print_wrapped(output_text)
    return output_text.replace(prompt, '')

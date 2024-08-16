import streamlit as st
from transcription import get_transcript
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

# ----------------------------------------------------------------------------

st.header("Video Summarizer 🎬")

full_yt = st.text_input("Enter video link", "")

model_path = "my_fine_tuned_t5_small_model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

model_path2 = "my_fine_tuned_t5_small_model_2"
model2 = T5ForConditionalGeneration.from_pretrained(model_path2)
tokenizer2 = T5Tokenizer.from_pretrained(model_path2)
summarizer2 = pipeline("summarization", model=model2, tokenizer=tokenizer2)

def chunk_text(text, tokenizer, max_length):
    #Splits the text into chunks of a specified max length in tokens.
    tokens = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_length
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        start = end
    return chunks

def summarize_large_text(summarizer, text, tokenizer, chunk_size=700, summary_max_length=150):
    #Summarizes large text by splitting it into chunks and combining the summaries.
    chunks = chunk_text(text, tokenizer, chunk_size)
    summarized_chunks = []
    for chunk in chunks:
        summary = summarizer("summarize: " + chunk, max_length=summary_max_length, min_length=30, do_sample=False)[0]['summary_text']
        summarized_chunks.append(summary)
    return " ".join(summarized_chunks)

if st.button("Get Summary using T5 model"):
    video_id = full_yt.split("=")[1]
    get_transcript(video_id)

    with open("transcription.txt", "r") as f:
        tx = f.read()
        summarized_text = summarize_large_text(summarizer, tx, tokenizer)

    st.write(summarized_text)

if st.button("Get Summary using pre-trained T5 model"):
    video_id = full_yt.split("=")[1]
    get_transcript(video_id)

    with open("transcription.txt", "r") as f:
        tx = f.read()
        summarized_text = summarize_large_text(summarizer2, tx, tokenizer2)

    st.write(summarized_text)

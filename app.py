import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import google.generativeai as genai
from google.cloud import storage
from streamlit_feedback import streamlit_feedback
import io
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import ast
from deep_translator import GoogleTranslator
from langdetect import detect

# Load environment variables
load_dotenv()
json_key_path = 'service-account-key.json'
credentials = service_account.Credentials.from_service_account_file(json_key_path)

# Configure clients
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
genai.configure(api_key=os.getenv("GEMINI_KEY"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

# Streamlit app config
st.set_page_config(page_title="LLMinder", page_icon="🔍🎯", layout="wide")

# Session state init
if 'feedback_key' not in st.session_state:
    st.session_state.feedback_key = 'feedback_widget'
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(int(time.time()))
    st.session_state.session_start_time = datetime.now()
    st.session_state.welcome_shown = False
elif (datetime.now() - st.session_state.session_start_time) > timedelta(minutes=10):
    st.session_state.session_id = str(int(time.time()))
    st.session_state.session_start_time = datetime.now()
    st.session_state.welcome_shown = False
if 'copy_button_clicked' not in st.session_state:
    st.session_state.copy_button_clicked = False

# Model config
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 50,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Custom CSS
st.markdown("""
<style>
.stProgress > div > div > div > div { background-color: #DA70D6; }
.element-container:has(div.stSuccessMessage) div[data-testid="stMarkdownContainer"] p {
    color: #953553 !important;
}
.element-container:has(div.stSuccessMessage) {
    background-color: rgba(149, 53, 83, 0.1) !important;
}
.element-container:has(div.stSuccessMessage) svg {
    fill: #953553 !important;
}
</style>
""", unsafe_allow_html=True)

@st.dialog("Welcome to LLMinder")
def welcome_message():
    st.balloons()
    st.write("""
 **LLMinder!**

✨ **Features:**
- Global Language Support
- Comprehensive Benchmarking
- Real-time Updates
- Quality Assured

Just describe your task and set your size preference.
    """)

@st.cache_data
def load_data():
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(os.getenv("BUCKET_NAME"))
        blob = bucket.blob("llm_leaderboard.csv")
        content = blob.download_as_string()
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        st.error(f"Error loading data from GCS: {str(e)}")
        return None

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != 'en':
            translated = GoogleTranslator(source=lang, target='en').translate(text)
            return translated, lang
        return text, 'en'
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        return text, 'en'

def process_task(task, model_size):
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Detecting language...")
    progress_bar.progress(15)
    time.sleep(0.5)

    status_text.text("Loading benchmark data...")
    data = load_data()
    data = data[(data['Official Providers']) & (data['Available on the hub']) & (~data['Flagged'])]
    data = data.sort_values("Average ⬆️", ascending=False)
    progress_bar.progress(30)
    time.sleep(0.5)

    status_text.text("Initializing Gemini...")
    progress_bar.progress(45)
    time.sleep(0.5)

    status_text.text("Processing your task requirements...")
    benchmark_prompt = read_file("prompts/benchmark_select.txt")
    benchmark_list = read_file("prompts/benchmark_list.txt")
    benchmark_prompt = benchmark_prompt.format(benchmarks=benchmark_list, input=task)
    response = model.generate_content(benchmark_prompt).text.replace("\n", "")

    try:
        selected_benchmarks = ast.literal_eval(response)
        if not selected_benchmarks:
            selected_benchmarks = ["Average ⬆️"]
    except:
        selected_benchmarks = ["Average ⬆️"]

    progress_bar.progress(60)
    time.sleep(0.5)

    status_text.text("Finding your perfect model matches...")
    ranked_models = data.sort_values(by=selected_benchmarks, ascending=False).reset_index(drop=True)
    overall_best = ranked_models.iloc[0]
    best_sized = ranked_models[ranked_models['#Params (B)'] <= model_size].iloc[0]

    progress_bar.progress(100)
    time.sleep(0.5)
    return overall_best, best_sized, selected_benchmarks

def create_model_dataframe(info, benchmarks):
    name = info['fullname'].split('/')[-1]
    rows = [
        ('Model Name', name),
        ('Size (B)', f"{info['#Params (B)']:.2f}"),
        ('Average Score', f"{info['Average ⬆️']:.2f}"),
        ('Architecture', info['Architecture']),
        ('License', info['Hub License'])
    ]
    for bench in benchmarks:
        if bench != "Average ⬆️" and bench in info:
            rows.append((f"{bench} Score", f"{info[bench]:.2f}"))
    return pd.DataFrame(rows, columns=['Metric', 'Value'])

def upload_to_bq(df, table_name):
    dest = client.dataset("model_matrimony").table(table_name)
    config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(df, dest, job_config=config).result()

def _submit_feedback(response, emoji=None):
    sid = st.session_state.get("session_id")
    value = 1 if response['score'] == '👍' else 0
    feedback_df = pd.DataFrame([[sid, value, response['text']]], columns=["session_id", "vote", "comment"])
    upload_to_bq(feedback_df, 'feedback_data')
    st.success("Your feedback has been submitted!")

# Main interface
st.title("LLminder")
if not st.session_state.welcome_shown:
    welcome_message()
    st.session_state.welcome_shown = True

st.markdown("#### Step 1: Describe Your Task")
task = st.text_area("", placeholder="E.g., Chatbot that answers medical questions...", height=100, label_visibility='hidden')

st.markdown("#### Step 2: Set Size Preference")
size = st.slider("Maximum model size (in billion parameters)", 1, 50, 15)

if st.button("Find My Perfect Match! 🎯"):
    if task:
        translated, lang = translate_to_english(task)
        if lang != 'en':
            st.info(f"Detected language: {lang}. Translated for processing.")
        best, best_size, selected = process_task(translated, size)
        st.success(" Yayyy Match found!")

        with st.expander("Results"):
            st.markdown("### 🏆 Overall Best Model")
            st.table(create_model_dataframe(best, selected))
            st.markdown("---")
            st.markdown(f"### 🎯 Best Model Under {size}B Parameters")
            st.table(create_model_dataframe(best_size, selected))
            st.markdown("### 📊 Selected Benchmarks")
            st.table(pd.DataFrame({'Benchmark': selected}))
    else:
        st.warning("Please describe your task first!")

streamlit_feedback(
    feedback_type="thumbs",
    optional_text_label="Extra feedback (optional)",
    on_submit=_submit_feedback,
    key=st.session_state.feedback_key,
)

st.markdown("<div style='text-align: center;'>LLMinder</div>", unsafe_allow_html=True)

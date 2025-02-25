# import streamlit as st
# from streamlit_chat import message
# import pandas as pd
# from llm_utils import chat_api, chat_with_data_api

# MAX_LENGTH_MODEL_DICT = {
#     "gpt-4": 8191,
#     "gpt-3.5-turbo": 4096,
#     "gpt-3.5-turbo-16k": 16384
# }

# def get_text():
#     return st.chat_input("Type your question here...")

# def sidebar():
#     """App sidebar settings"""
#     model = st.selectbox(
#         "Available Models",
#         ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
#     )
#     temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.01)
#     max_tokens = st.slider("Max Tokens", 0, MAX_LENGTH_MODEL_DICT[model], 256, 1)
#     top_p = st.slider("Top P", 0.0, 1.0, 0.5, 0.01)

#     return {"model": model, "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}

# def chatbot():
#     """Unified chatbot with file handling and text-based chat"""

#     st.title("Smart Chatbot: Chat & Query Your Data")

#     with st.sidebar:
#         model_params = sidebar()
#         memory_window = st.slider("Memory Window", 1, 10, 3)

#     # Upload file
#     uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
#     df = pd.read_csv(uploaded_file) if uploaded_file else None

#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [{"role": "system", "content": "You're a chatbot that answers questions and analyzes data."}]
#     if "past" not in st.session_state:
#         st.session_state["past"] = []
#     if "generated" not in st.session_state:
#         st.session_state["generated"] = []

#     user_input = get_text()

#     if user_input:
#         st.session_state["messages"].append({"role": "user", "content": user_input})
#         st.session_state["past"].append(user_input)

#         if df is not None:
#             response = chat_with_data_api(df, **model_params)
#         else:
#             response = chat_api(st.session_state["messages"], **model_params)

#         if response:
#             st.session_state["generated"].append(response)
#             st.session_state["messages"].append({"role": "assistant", "content": response})

#     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         if i - 1 >= 0:
#             message(st.session_state["past"][i - 1], is_user=True, key=str(i) + "_user")

# if __name__ == "__main__":
#     chatbot()


import os
import pandas as pd
import streamlit as st
from streamlit_chat import message
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
from llm_utils import chat_api, chat_with_data_api
from forecasting import forecasting  # Importing the forecasting function

MAX_LENGTH_MODEL_DICT = {
    "gpt-4": 8191,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384
}

def get_text():
    return st.chat_input("Type your question here...")

def sidebar():
    """App sidebar settings"""
    model = st.selectbox(
        "Available Models",
        ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.01)
    max_tokens = st.slider("Max Tokens", 0, MAX_LENGTH_MODEL_DICT[model], 256, 1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.5, 0.01)

    return {"model": model, "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def convert_pdf_to_csv(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    lines = text.split("\n")
    data = [line.split() for line in lines if line]
    df = pd.DataFrame(data)
    csv_path = pdf_path.replace(".pdf", ".csv")
    df.to_csv(csv_path, index=False)
    return df

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.pdf'):
        df = convert_pdf_to_csv(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, "r") as file:
            text = file.read()
        df = pd.DataFrame({"text": text.splitlines()})
    else:
        raise ValueError("Unsupported file format. Please upload CSV, PDF, or TXT.")
    return df

def eda(df):
    """Perform Exploratory Data Analysis and show basic statistics and plots"""
    st.write("Summary Statistics:")
    st.write(df.describe())
    
    df = df.loc[:, df.nunique() > 1]
    df = df.fillna(0)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Correlation Matrix
    st.write("Feature Correlation Matrix:")
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    
    # Trend Analysis (for numeric columns)
    st.write("Trend Analysis:")
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in numeric_df.columns[:10]:  # Limit to first 10 columns for clarity
        sns.lineplot(data=numeric_df[column], label=column, ax=ax)
    ax.legend()
    ax.set_title("Trend Analysis")
    ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    st.pyplot(fig)

def chatbot():
    """Unified chatbot with file handling, text-based chat, and forecasting"""

    st.title("Aviation AI Chatbot ✈️")

    # Upload file (CSV, PDF, or TXT)
    uploaded_file = st.file_uploader("Upload CSV, PDF, or Text file", type=["csv", "pdf", "txt"])
    df = None  # Initialize df as None
    if uploaded_file:
        # Save uploaded file
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the data
        try:
            df = load_data(file_path)
            st.write(f"Uploaded file: {uploaded_file.name}")
            eda(df)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are an expert chatbot. Answer based on the uploaded dataset whenever possible."}]
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    user_input = get_text()

    # Initialize the response variable
    response = None  # Ensure response is initialized

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["past"].append(user_input)

        if df is not None:
            # Forecasting based on user query
            if "forecast" in user_input.lower():
                target_column = st.text_input("Enter the target column for forecasting:")
                if target_column:
                    try:
                        predictions = forecasting(df, target_column)  # Call the forecasting function
                        st.write(f"Predictions for {target_column}:")
                        st.write(predictions)
                    except Exception as e:
                        st.error(f"Error in forecasting: {e}")
            else:
                response = chat_with_data_api(df, **model_params)
        else:
            response = chat_api(st.session_state["messages"], **model_params)

    # Only append to session state if response is not None
    if response:
        st.session_state["generated"].append(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    
    # Display past chat history in the correct order
    # Ensure we have responses to display
    num_responses = len(st.session_state["generated"])

    for idx, user_msg in enumerate(st.session_state["past"]):
        message(user_msg, is_user=True, key=f"user_{idx}")  # Display user input
        if idx < num_responses:  # Ensure response exists before accessing
            message(st.session_state["generated"][idx], key=f"bot_{idx}")  # Display bot response

    # Show the latest user input at the end if it hasn't been added
    if user_input and (len(st.session_state["past"]) == 0 or st.session_state["past"][-1] != user_input):
        message(user_input, is_user=True, key="latest_user_input")



if __name__ == "__main__":
    chatbot()

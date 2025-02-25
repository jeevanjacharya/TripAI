import os
import json
import pandas as pd
import streamlit as st
from streamlit_chat import message
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
from llm_utils import chat_api, chat_with_data_api
from forecasting import forecasting  # Importing the forecasting function

CHAT_HISTORY_FILE = "chat_history.json"

MAX_LENGTH_MODEL_DICT = {
    "gpt-4": 8191,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384
}

def get_text():
    return st.chat_input("Type your question here...")



def sidebar():
    st.sidebar.title("✈️ Flight AI Assistant")
    st.sidebar.markdown("### Customize your chatbot experience")

    model = st.sidebar.selectbox(
        "🤖 Choose AI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        help="Select an AI model for generating responses."
    )

    temperature = st.sidebar.slider(
        "🔥 Creativity (Temperature)",
        0.0, 2.0, 0.7, 0.01,
        help="Higher values make responses more creative."
    )

    max_tokens = st.sidebar.slider(
        "📝 Max Tokens",
        0, 4096, 256, 1,
        help="Defines the maximum response length."
    )

    top_p = st.sidebar.slider(
        "🎯 Response Precision",
        0.0, 1.0, 0.5, 0.01,
        help="Lower values make responses more deterministic."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "🛫 **Welcome to Flight AI Assistant!** This chatbot helps you analyze aviation data, "
        "visualize airline trends, and predict flight patterns. 🚀"
    )

    return {  
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }


def save_chat_history():
    """Save chat history to a JSON file"""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(st.session_state["messages"], file, indent=4)

def load_chat_history():
    """Load chat history from a JSON file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            st.session_state["messages"] = json.load(file)

def clear_chat_history():
    """Clear chat history in session state and delete file"""
    st.session_state["messages"] = [{"role": "system", "content": "You're an AI assistant."}]
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

# def load_data(file_path):
#     """Load data from CSV, PDF, or TXT files"""
#     ext = os.path.splitext(file_path)[-1].lower()

#     if ext == ".csv":
#         return pd.read_csv(file_path)
#     elif ext == ".txt":
#         with open(file_path, "r", encoding="utf-8") as file:
#             return pd.DataFrame({"text": file.readlines()})
#     elif ext == ".pdf":
#         with pdfplumber.open(file_path) as pdf:
#             text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#         return pd.DataFrame({"text": [text]})
#     else:
#         raise ValueError("Unsupported file format")
    
# def load_data(file_path):
#     """Load data from CSV, PDF, or TXT files and convert TXT to CSV if needed."""
#     ext = os.path.splitext(file_path)[-1].lower()

#     if ext == ".csv":
#         return pd.read_csv(file_path)
    
#     elif ext == ".txt":
#         with open(file_path, "r", encoding="utf-8") as file:
#             lines = file.readlines()
        
#         # Extract column names from the first line
#         columns = lines[0].strip().split("\t")  
        
#         # Extract data from subsequent lines
#         data = [line.strip().split("\t") for line in lines[1:]]
        
#         # Convert to DataFrame
#         df = pd.DataFrame(data, columns=columns)

#         # Save as CSV for later use
#         csv_path = file_path.replace(".txt", ".csv")
#         df.to_csv(csv_path, index=False)
        
#         return df  # Return the DataFrame for further processing

#     elif ext == ".pdf":
#         with pdfplumber.open(file_path) as pdf:
#             text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#         return pd.DataFrame({"text": [text]})
    
#     else:
#         raise ValueError("Unsupported file format")




def load_data(file_path):
    """Load data from CSV, PDF, or TXT files and convert TXT to CSV if needed."""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    
    elif ext == ".txt":
        csv_path = file_path.replace(".txt", ".csv")  # Define CSV path

        # Convert TXT to CSV if it hasn't been converted already
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Extract column names from the first line
        columns = lines[0].strip().split("\t")  
        
        # Extract data from subsequent lines
        data = [line.strip().split("\t") for line in lines[1:]]
        
        # Save as CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for row in data:
                f.write(",".join(row) + "\n")

        # Return as a DataFrame by reading the CSV file
        return pd.read_csv(csv_path)

    elif ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
           text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return pd.DataFrame({"text": [text]})
    
    else:
        raise ValueError("Unsupported file format")


# def eda(df):
#     """Basic EDA on uploaded data"""
#     st.subheader("Exploratory Data Analysis")
#     st.write(df.head())

#     if isinstance(df, pd.DataFrame):
#         st.write("Dataset Information:")
#         st.write(df.describe())

#         # Correlation heatmap (for numerical data)
#         numeric_df = df.select_dtypes(include=[np.number])
#         if len(numeric_df.columns) > 1:
#             st.subheader("Correlation Heatmap")
#             plt.figure(figsize=(10, 6))
#             sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
#             st.pyplot(plt)

#         # Trend Analysis (for numeric columns)
#         if not numeric_df.empty:
#             st.subheader("Trend Analysis")
#             fig, ax = plt.subplots(figsize=(12, 6))
#             for column in numeric_df.columns[:10]:  # Limit to first 10 columns for clarity
#                 sns.lineplot(data=numeric_df[column], label=column, ax=ax)
#             ax.legend()
#             ax.set_title("Trend Analysis")
#             ax.set_xlabel("Index")
#             ax.set_ylabel("Values")
#             st.pyplot(fig)

#         # Histogram Plot (distribution of numeric columns)
#         if not numeric_df.empty:
#             st.subheader("Histogram Plot")
#             fig, ax = plt.subplots(figsize=(12, 6))
#             numeric_df.hist(bins=30, figsize=(12, 6), ax=ax)
#             plt.tight_layout()
#             st.pyplot(fig)

def eda(df):
    """Basic EDA on uploaded data"""
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write(df.head())

    if isinstance(df, pd.DataFrame):
        st.write("Dataset Summary:")
        st.write(df.describe())

        # Correlation heatmap (for numerical data)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Trend Analysis (for numeric columns)
        if not numeric_df.empty:
            st.subheader("Trend Analysis")
            fig, ax = plt.subplots(figsize=(12, 6))
            for column in numeric_df.columns[:10]:  # Limit to first 10 columns for clarity
                sns.lineplot(data=numeric_df[column], label=column, ax=ax)
            ax.legend()
            ax.set_title("Trend Analysis of Numerical Features")
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            st.pyplot(fig)

        # # Histogram Plot (distribution of numeric columns)
        # if not numeric_df.empty:
        #     st.subheader("Histogram Plot")
        #     fig, axes = plt.subplots(nrows=1, ncols=len(numeric_df.columns), figsize=(15, 6))

        #     # If only one column, ensure axes is iterable
        #     if len(numeric_df.columns) == 1:
        #         axes = [axes]

        #     for col, ax in zip(numeric_df.columns, axes):
        #         numeric_df[col].hist(bins=30, ax=ax)
        #         ax.set_title(f"Distribution of {col}")
        #         ax.set_xlabel(col)
        #         ax.set_ylabel("Count")

        #     plt.tight_layout()
        #     st.pyplot(fig)

        # Histogram Plot (distribution of numeric columns)
        if not numeric_df.empty:
            st.subheader("Histogram Plot")
            
            num_cols = len(numeric_df.columns)
            cols_per_row = 3  # Set number of columns per row
            rows = (num_cols // cols_per_row) + (num_cols % cols_per_row > 0)
            
            fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(20, 10), constrained_layout=True)
            
            axes = axes.flatten() if num_cols > 1 else [axes]  # Flatten for easy iteration
            
            for i, col in enumerate(numeric_df.columns):
                numeric_df[col].hist(bins=30, ax=axes[i], edgecolor="black")
                axes[i].set_title(f"Distribution of {col}", fontsize=14)
                axes[i].set_xlabel(col, fontsize=12)
                axes[i].set_ylabel("Count", fontsize=12)
                axes[i].tick_params(axis='x', rotation=90, labelsize=10)  # Rotate labels
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            st.pyplot(fig)




def detect_chart_type(user_input):
    """Detect the type of chart requested by the user."""
    chart_mapping = {
        "line graph": "line",
        "scatter plot": "scatter",
        "pie chart": "pie",
        "bar chart": "bar",
    }

    for key, value in chart_mapping.items():
        if key in user_input.lower():
            return value  # Return chart type

    return "line"  # Default to line chart if unspecified

# def plot_chart(df, chart_type):
#     """Generate the requested chart from the dataset and display it in Streamlit."""
#     st.subheader(f"📊 {chart_type.capitalize()} Chart")

#     # Select columns for visualization
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     if not numeric_cols:
#         st.error("No numeric columns available for visualization.")
#         return

#     x_axis = st.selectbox("Select X-axis:", numeric_cols, index=0)
#     y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

#     fig, ax = plt.subplots(figsize=(8, 5))

#     if chart_type == "line":
#         sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, marker="o")
#     elif chart_type == "scatter":
#         sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
#     elif chart_type == "pie":
#         if df[x_axis].nunique() > 10:
#             st.warning("Pie charts work best with fewer categories. Consider another chart type.")
#         else:
#             df.groupby(x_axis)[y_axis].sum().plot.pie(autopct='%1.1f%%', ax=ax)
#             ax.set_ylabel("")  # Hide y-axis label for pie chart
#     elif chart_type == "bar":
#         sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)

#     ax.set_title(f"{chart_type.capitalize()} of {y_axis} vs {x_axis}")

#     # Ensure the chart is displayed in Streamlit
#     st.pyplot(fig)

#     # Display confirmation message
#     st.success(f"✅ Here is your {chart_type} chart!")
       
def plot_chart(df, chart_type):
    """Generate the requested chart from the dataset and persist it across reruns."""
    
    st.subheader(f"📊 {chart_type.capitalize()} Chart")

    # Select columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns available for visualization.")
        return

    x_axis = st.selectbox("Select X-axis:", df.columns, index=0)  # Allow categorical columns
    y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=0)

    # Check if the selected columns are valid
    if x_axis not in df.columns or y_axis not in df.columns:
        st.error("Invalid column selection. Please select valid X and Y axes.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "line":
        sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, marker="o")
    elif chart_type == "scatter":
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    elif chart_type == "pie":
        if df[x_axis].nunique() > 10:
            st.warning("Pie charts work best with fewer categories. Consider another chart type.")
        else:
            df.groupby(x_axis)[y_axis].sum().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")  # Hide y-axis label for pie chart
    elif chart_type == "bar":
        grouped_data = df.groupby(x_axis)[y_axis].sum()  # Grouped data for bar chart
        grouped_data.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} by {x_axis}")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis="y")

    # Save figure in session state to prevent it from disappearing
    st.session_state["chart_fig"] = fig

    # Display the chart
    st.pyplot(fig)

    st.success(f"✅ Here is your {chart_type} chart!")

def chatbot():
    """Unified chatbot with file handling, text-based chat, and forecasting"""

    st.title("Aviation AI Chatbot ✈️")
    st.markdown(
        """
        Welcome to the *Aviation AI Chatbot! This assistant helps you **analyze airline trends, predict flights, and visualize aviation data*.  
        *Features:*
        - 📊 *Upload & Explore Flight Datasets*
        - 🛫 *Generate Charts & Predictions*
        - 💬 *Chat with Aviation AI*
        - 🔍 *Get Flight Insights Instantly!*
        """,
        unsafe_allow_html=True,
    )
    
    if "model_params" in st.session_state:
       del st.session_state["model_params"]


    with st.sidebar:
        model_params = sidebar()

        if st.button("Clear Chat History"):
            clear_chat_history()
            st.experimental_rerun()

        if st.button("Load Previous Chat"):
            load_chat_history()
            st.experimental_rerun()

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
    if "messages" not in st.session_state or st.session_state.get("reset_chat", False):
        st.session_state["messages"] = [{"role": "system", "content": "You're an AI assistant."}]
        st.session_state["reset_chat"] = False  # Reset the flag

    user_input = get_text()
    response = None  # Ensure response is initialized

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        if df is not None:
        # Check if the user asked for a specific type of chart
            if any(chart_type in user_input.lower() for chart_type in ["line graph", "scatter plot", "pie chart", "bar chart"]):
                try:
                   chart_type = detect_chart_type(user_input)  # Function to determine chart type
                   plot_chart(df, chart_type)  # Function to generate the requested chart
                except Exception as e:
                    st.error(f"Error generating chart: {e}")
            else:
            # Run chatbot response
                response = chat_with_data_api(df, model=model_params["model"], 
                          temperature=model_params["temperature"], 
                          max_tokens=model_params["max_tokens"], 
                          top_p=model_params["top_p"])

        else:
            response = chat_api(st.session_state["messages"], **model_params)
        

        if df is not None:
            if "forecast" in user_input.lower():
                target_column = st.text_input("Enter the target column for forecasting:")
                if target_column:
                    try:
                        predictions = forecasting(df, target_column)
                        st.write(f"Predictions for {target_column}:")
                        st.write(predictions)
                    except Exception as e:
                        st.error(f"Error in forecasting: {e}")
            else:
                #response = chat_with_data_api(df, **model_params)
                response = chat_with_data_api(df, model=model_params["model"], 
                              temperature=model_params["temperature"], 
                              max_tokens=model_params["max_tokens"], 
                              top_p=model_params["top_p"])

        else:
            filtered_model_params = {
              "model": model_params.get("model"),
              "temperature": model_params.get("temperature"),
              "max_tokens": model_params.get("max_tokens"),
              "top_p": model_params.get("top_p")
            }

            response = chat_api(st.session_state["messages"], **filtered_model_params)

        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})
        
        # Restore chart from session state if it exists
        if "chart_fig" in st.session_state:
            st.pyplot(st.session_state["chart_fig"])


        save_chat_history()  # Save after every interaction

    # Display past chat history
    for msg in st.session_state["messages"]:
        if msg["role"] != "system":  # Skip system messages
            message(msg["content"], is_user=(msg["role"] == "user"))

if __name__ == "__main__":
    chatbot()

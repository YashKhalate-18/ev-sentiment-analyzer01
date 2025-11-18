import streamlit as st
from transformers import pipeline
import pandas as pd  # <-- You need to import pandas!
import numpy as np   # <-- Added for calculations
import io              # <-- Added for the download button

# --- 1. SET UP THE APP ---
st.set_page_config(page_title="EV Sentiment Analyzer", layout="wide")
st.title("🚗 EV Public Sentiment Analyzer")
st.write("This app uses a pre-trained AI model to classify the sentiment of EV reviews.")

# --- 2. LOAD THE AI MODEL (NLP) ---
@st.cache_resource
def load_model():
    st.write("Loading AI Model... (This only happens once)")
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_pipeline = load_model()

# --- 3. CREATE TABS FOR DIFFERENT MODES ---
# We use tabs to separate the "Single Review" and "Data Set" modes
tab1, tab2 = st.tabs(["Analyze Single Review", "Analyze Data Set (Upload)"])

# --- 4. "ANALYZE SINGLE REVIEW" TAB (Your Old Code) ---
with tab1:
    st.subheader("Enter EV Review Text:")
    user_input = st.text_area("Type a review, tweet, or comment about an EV:", "The new EV model has an amazing battery life!", key="single_text")

    if st.button("Analyze Sentiment"):
        if user_input:
            result = sentiment_pipeline(user_input)[0]
            label = result['label']
            score = result['score']

            st.subheader("Analysis Result:")
            if label.lower() == 'positive':
                st.success(f"Sentiment: {label.capitalize()} (Confidence: {score:.2f})")
                st.markdown("### 😄")
            elif label.lower() == 'neutral':
                st.info(f"Sentiment: {label.capitalize()} (Confidence: {score:.2f})")
                st.markdown("### 😐")
            else: # Negative
                st.error(f"Sentiment: {label.capitalize()} (Confidence: {score:.2f})")
                st.markdown("### 😡")
            
            # (Your existing topic extraction code would go here)
            
        else:
            st.warning("Please enter some text to analyze.")

# --- 5. "ANALYZE DATA SET" TAB (NEW FEATURE) ---
with tab2:
    st.header("Upload Your EV Review Data Set")
    
    # 1. Add the file uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' Uploaded!")
        
        # 2. Read the file into pandas
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop() # Stop the app if file is bad

        # 3. Get the column with the reviews
        st.subheader("Which column has the review text?")
        review_column = st.selectbox("Select the column to analyze:", df.columns)
        
        # 4. Analyze the data set
        if st.button("Analyze Data Set"):
            
            # --- This is the core AI/ML part ---
            # Create a progress bar
            progress_bar = st.progress(0)
            st.write(f"Analyzing {len(df)} reviews... This may take a moment.")
            
            # We run the pipeline on every row in the selected column
            # 'apply' is much faster than a loop
            # We add results to new columns
            try:
                # Use a lambda function to get the label
                df['sentiment_label'] = df[review_column].apply(lambda x: sentiment_pipeline(x)[0]['label'])
                # Use a lambda function to get the score
                df['sentiment_score'] = df[review_column].apply(lambda x: sentiment_pipeline(x)[0]['score'])
                progress_bar.progress(100) # Update progress bar to 100%
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error("Make sure your review column doesn't have bad data (like numbers).")
                st.stop()
            # --- End of AI/ML part ---

            st.success("Analysis Complete!")
            
            # 5. Display the results
            st.subheader("Analysis Dashboard")
            
            # Show a simple pie chart
            sentiment_counts = df['sentiment_label'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", len(df))
            col2.metric("Positive Reviews", sentiment_counts.get('Positive', 0))
            col3.metric("Negative Reviews", sentiment_counts.get('Negative', 0))

            # 6. Show the analyzed data and add a download button
            st.subheader("Analyzed Data")
            st.dataframe(df)

            # Convert DataFrame to CSV in memory for download
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(df)

            st.download_button(
                label="Download Analyzed Data as CSV",
                data=csv_data,
                file_name="analyzed_reviews.csv",
                mime="text/csv",
            )
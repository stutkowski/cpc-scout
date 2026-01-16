import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="CPC Scout: Visual Intelligence", layout="wide")

# --- DATA & MODEL LOADER ---
@st.cache_resource
def load_engine():
    # 1. LOAD DATA (Demo Data - Replace with pd.read_csv('cpc_full.csv') in prod)
    data = {
        'code': [
            'B64C 39/02', 'G06N 3/00', 'H04L 9/00', 'A61B 5/00', 
            'B60W 30/00', 'G06Q 20/00', 'F03D 1/00', 'H01M 10/00',
            'G06F 40/20', 'H04W 4/00', 'B64D 1/00', 'G06N 20/00',
            'H04L 63/00', 'A61B 5/02', 'B60W 40/00', 'G06Q 30/00'
        ],
        'description': [
            'Aircraft not otherwise provided for; Unmanned aerial vehicles (drones)',
            'Computer systems based on biological models (Artificial Intelligence/Neural Networks)',
            'Cryptographic mechanisms or cryptographic arrangements for secret communication',
            'Measuring for diagnostic purposes; Identification of persons',
            'Road vehicle drive control systems (Autonomous Driving/ADAS)',
            'Payment architectures, schemes or protocols (Fintech)',
            'Wind motors with rotation axis substantially parallel to the air flow',
            'Secondary cells; Manufacture thereof (Batteries/Li-Ion)',
            'Natural language processing (NLP); Text analysis',
            'Services making use of wireless communication networks',
            'Dropping or releasing articles from aircraft',
            'Machine Learning',
            'Network architectures or network communication protocols for network security',
            'Measuring pulse, heart rate, blood pressure',
            'Estimation or calculation of driving parameters for road vehicle drive control',
            'Commerce, e.g. shopping or auctions'
        ]
    }
    df = pd.DataFrame(data)
    
    # 2. COMBINE TEXT
    df['search_text'] = df['code'] + " " + df['description']
    
    # 3. BUILD VECTORIZER
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])
    
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_engine()

# --- SEARCH & CLUSTERING LOGIC ---
def search_and_cluster(query, top_n=15):
    if not query:
        return pd.DataFrame(), None

    # 1. Search
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    top_indices = [i for i in top_indices if similarity_scores[i] > 0]
    
    if not top_indices:
        return pd.DataFrame(), None
    
    results = df.iloc[top_indices].copy()
    results['relevance'] = similarity_scores[top_indices]
    
    # 2. Dimensionality Reduction (PCA) for Visualization
    # We take the vectors of the RESULTS only to plot them relative to each other
    result_vectors = tfidf_matrix[top_indices].toarray()
    
    # We need at least 3 points to define a 2D plane meaningfully, otherwise PCA fails
    if len(results) >= 3:
        pca = PCA(n_components=2)
        components = pca.fit_transform(result_vectors)
        results['x'] = components[:, 0]
        results['y'] = components[:, 1]
    else:
        results['x'] = 0
        results['y'] = 0
        
    return results, query_vec

# --- MAIN UI ---
def main():
    st.title("🛡️ CPC Scout: Visual Landscape")
    st.markdown("Map keyword concepts to the Patent Classification Scheme.")

    with st.form("search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Input Invention/Concept", placeholder="e.g. autonomous swarm drones")
        with col2:
            st.write("")
            st.write("")
            submit = st.form_submit_button("Generate Landscape", type="primary")

    if submit and query:
        results, _ = search_and_cluster(query)
        
        if not results.empty:
            st.divider()
            
            # --- LAYOUT: 2 Columns (Chart | Data) ---
            viz_col, data_col = st.columns([1.5, 1])
            
            with viz_col:
                st.subheader("Semantic Map")
                if len(results) >= 3:
                    fig = px.scatter(
                        results, x='x', y='y',
                        color='relevance', size='relevance',
                        hover_data=['code', 'description'],
                        text='code',
                        color_continuous_scale='Viridis',
                        title="Clustering of Matching Classes"
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(xaxis_visible=False, yaxis_visible=False, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data points to generate a cluster map.")

            with data_col:
                st.subheader("Top Matches")
                display_df = results[['code', 'description', 'relevance']].copy()
                display_df['relevance'] = display_df['relevance'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_df,
                    column_config={
                        "relevance": st.column_config.ProgressColumn("Confidence")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("No matches found.")

if __name__ == "__main__":
    main()

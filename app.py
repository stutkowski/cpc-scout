import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="CPC Scout: Inverse Lookup", layout="wide")

# --- API HANDLER (PatentsView Stable Endpoint) ---
def search_patentsview_usage(keyword):
    """
    1. Searches for patents containing the keyword.
    2. Aggregates the CPC codes assigned to those patents.
    """
    # Main stable endpoint
    url = "https://api.patentsview.org/patents/query"
    
    # Query: Find patents where the Title or Abstract contains the keyword
    query = {
        "_or": [
            {"_text_any": {"patent_title": keyword}},
            {"_text_any": {"patent_abstract": keyword}}
        ]
    }
    
    # Fields: Get the CPC Subgroup ID and Title for each patent found
    fields = ["cpc_subgroup_id", "cpc_subgroup_title"]
    
    params = {
        "q": str(query).replace("'", '"'),
        "f": str(fields).replace("'", '"'),
        "o": '{"per_page": 50}' # Analyze top 50 relevant patents
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # --- PARSING LOGIC ---
        # The API returns a list of patents. Each patent has a list of CPCs.
        # We must flatten this into a single list of codes to count them.
        
        cpc_counts = Counter()
        cpc_definitions = {}
        
        if "patents" in data:
            for patent in data["patents"]:
                # Check if this patent has CPCs assigned
                if "cpc_subgroups" in patent and patent["cpc_subgroups"]:
                    for cpc in patent["cpc_subgroups"]:
                        code = cpc.get("cpc_subgroup_id")
                        title = cpc.get("cpc_subgroup_title")
                        
                        if code and title:
                            cpc_counts[code] += 1
                            cpc_definitions[code] = title
                            
            # Convert to DataFrame
            if cpc_counts:
                df = pd.DataFrame(cpc_counts.items(), columns=['CPC Code', 'Frequency'])
                df['Definition'] = df['CPC Code'].map(cpc_definitions)
                
                # Calculate Relevance % (Frequency / Total Patents analyzed)
                total_patents = len(data["patents"])
                df['Relevance'] = df['Frequency'] / total_patents
                
                # Sort by Frequency
                df = df.sort_values(by='Frequency', ascending=False).head(15)
                return df
            
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --- MAIN UI ---
def main():
    st.title("🛡️ CPC Scout: Operational Usage")
    st.markdown("""
    **Strategy:** This tool searches recent patents for your keyword and calculates 
    which CPC codes are most frequently assigned to them.
    """)

    with st.form("search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query_term = st.text_input("Enter Concept", placeholder="e.g. lidar, mrna, blockchain")
        with col2:
            st.write("")
            st.write("")
            submit = st.form_submit_button("Identify Classes", type="primary")

    if submit and query_term:
        with st.spinner(f"Analyzing how the USPTO classifies '{query_term}'..."):
            results = search_patentsview_usage(query_term)
        
        if not results.empty:
            st.divider()
            
            # --- VISUALIZATION ---
            col_viz, col_data = st.columns([1.5, 1])
            
            with col_viz:
                st.subheader("Dominant Classifications")
                fig = px.bar(
                    results, 
                    x="Frequency", 
                    y="CPC Code",
                    orientation='h',
                    hover_data=["Definition"],
                    color="Frequency",
                    color_continuous_scale="Blues",
                    title=f"Top CPC Codes for '{query_term}' (by usage)"
                )
                fig.update_layout(yaxis=dict(autorange="reversed")) # Top result at top
                st.plotly_chart(fig, use_container_width=True)

            with col_data:
                st.subheader("Data Table")
                # Format relevance for display
                display_df = results.copy()
                display_df['Relevance'] = display_df['Relevance'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_df[['CPC Code', 'Definition', 'Relevance']], 
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning(f"No patents found for '{query_term}' or API returned empty data.")

if __name__ == "__main__":
    main()

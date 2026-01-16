import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="CPC Scout: Live API", layout="wide")

# --- API HANDLER (PatentsView) ---
def search_patentsview(keyword):
    """
    Queries the PatentsView API for CPC Subsections matching the keyword.
    Docs: https://patentsview.org/apis/api-endpoints/cpc
    """
    url = "https://api.patentsview.org/cpc_subsections/query"
    
    # query syntax: finds keyword in the title of the subsection
    query = {"_text_any": {"cpc_subsection_title": keyword}}
    
    # fields to return
    fields = ["cpc_subsection_id", "cpc_subsection_title", "cpc_total_num_patents"]
    
    params = {
        "q": str(query).replace("'", '"'), # API requires double quotes
        "f": str(fields).replace("'", '"'),
        "o": '{"per_page": 50}'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if 'cpc_subsections' exists in response
        if "cpc_subsections" in data:
            df = pd.DataFrame(data["cpc_subsections"])
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return pd.DataFrame()

# --- MAIN UI ---
def main():
    st.title("🛡️ CPC Scout: Live API Edition")
    st.markdown("Fetch real-time Patent Classification data via **USPTO PatentsView API**.")

    with st.form("search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query_term = st.text_input("Enter Concept", placeholder="e.g. biology, vehicle, polymer")
        with col2:
            st.write("")
            st.write("")
            submit = st.form_submit_button("Search API", type="primary")

    if submit and query_term:
        with st.spinner(f"Querying USPTO database for '{query_term}'..."):
            results = search_patentsview(query_term)
        
        if not results.empty:
            # Clean up column names for display
            display_df = results.rename(columns={
                "cpc_subsection_id": "CPC Code",
                "cpc_subsection_title": "Definition",
                "cpc_total_num_patents": "Total Patents (Volume)"
            })
            
            # --- VISUALIZATION: Volume by Class ---
            st.divider()
            col_viz, col_data = st.columns([1, 1])
            
            with col_viz:
                st.subheader("Patent Volume by Class")
                fig = px.bar(
                    display_df, 
                    x="CPC Code", 
                    y="Total Patents (Volume)",
                    hover_data=["Definition"],
                    color="Total Patents (Volume)",
                    color_continuous_scale="Viridis",
                    title=f"Dominant Classes for '{query_term}'"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_data:
                st.subheader("Classification Matches")
                st.dataframe(
                    display_df, 
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("No matches found in USPTO database. Try a broader term (e.g., 'Aircraft' instead of 'Quadcopters').")

if __name__ == "__main__":
    main()

from datetime import datetime
import streamlit as st

class SearchLogger:
    def __init__(self):
        if "session_id" not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "last_selected" not in st.session_state:
            st.session_state.last_selected = None

    def log_selection(self, query, all_results, selected_result, similarities):
        # Just return the log data without database interaction
        log_data = {
            "session_id": st.session_state.session_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "all_results": all_results,
            "selected_result": selected_result,
            "similarities": similarities
        }
        
        print("Search interaction:", log_data)  # Optional: print to console
        return log_data
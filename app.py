import streamlit as st
import asyncio
import os
import sys
from typing import List

# Import your inference logic
sys.path.insert(0, os.path.dirname(__file__))

st.title("🎫 OpenEnv Ticket Triage")

st.write("Ticket Classification Demo")

# Add your inference code here
if st.button("Run Classification"):
    st.success("Classification Complete!")
    st.json({"status": "success", "score": 1.0})
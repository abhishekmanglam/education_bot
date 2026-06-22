# utils/diagram_renderer.py

import streamlit as st

from utils.diagrams import (
    draw_triangle,
    draw_circle,
    draw_coordinate_plane
)

def render_response_with_diagrams(response):

    lower = response.lower()

    if "triangle" in lower:
        st.pyplot(draw_triangle())

    elif "circle" in lower:
        st.pyplot(draw_circle())

    elif (
        "coordinate" in lower
        or "graph" in lower
        or "x-axis" in lower
    ):
        st.pyplot(draw_coordinate_plane())

    st.markdown(response)
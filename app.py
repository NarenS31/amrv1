from pathlib import Path
import streamlit as st

# ---------------------------------------
# Page config
# ---------------------------------------
st.set_page_config(
    page_title="AMR-Net",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------
# Load CSS
# ---------------------------------------
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>",
                unsafe_allow_html=True)
else:
    st.error(f"styles.css not found at {css_path}")

# ---------------------------------------
# Navbar (Apple-style)
# ---------------------------------------
st.markdown("""
<div class="nav">
  <div class="nav-inner">
    <div class="brand">
      <span class="brand-dot"></span>
      <span>AMR-Net</span>
    </div>
    <div class="nav-links">
      <a>Home</a>
      <a>Dashboard</a>
      <a>Docs</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# Hero section
# ---------------------------------------
st.markdown("""
<div class="section">
  <h1 class="h1">Genomic AMR prediction, built for real data</h1>
  <p class="body">
    Self-supervised learning for antimicrobial resistance prediction
    from incomplete and fragmented assemblies.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# Feature cards
# ---------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
      <div class="card-title">Self-supervised learning</div>
      <p class="card-body">Learn representations from unlabeled genomes.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
      <div class="card-title">Multi-antibiotic prediction</div>
      <p class="card-body">Shared embeddings across antibiotics.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
      <div class="card-title">Robust to fragmentation</div>
      <p class="card-body">Stable performance under missingness.</p>
    </div>
    """, unsafe_allow_html=True)

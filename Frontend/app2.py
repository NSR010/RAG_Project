import streamlit as st
import streamlit.components.v1 as components
import requests

# ═══════════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MNNIT Study Mate",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# 2. ROADSIDE CODER – DARK NEON THEME CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

/* ── Root Vars ──────────────────────────────────────────────── */
:root {
  --bg:        #050810;
  --bg2:       #0c1220;
  --bg3:       #111827;
  --cyan:      #00e5ff;
  --green:     #00ff88;
  --purple:    #b06aff;
  --orange:    #ff6b35;
  --border:    rgba(0,229,255,0.15);
  --glow-c:    0 0 20px rgba(0,229,255,0.35);
  --glow-g:    0 0 20px rgba(0,255,136,0.35);
  --text:      #e2e8f0;
  --text-dim:  #64748b;
  --font-mono: 'JetBrains Mono', monospace;
  --font-head: 'Syne', sans-serif;
}

/* ── Base Reset ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { 
  background-color: var(--bg) !important; 
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; font-family: var(--font-mono) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stRadio > div { 
  background: var(--bg3) !important; 
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ── Chat Messages ──────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  margin-bottom: 8px !important;
  padding: 12px 16px !important;
  font-family: var(--font-mono) !important;
}

/* ── Chat Input ─────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: var(--glow-c) !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: var(--cyan) !important; }

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: var(--font-mono) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: rgba(0,229,255,0.1) !important;
  box-shadow: var(--glow-c) !important;
}

/* ── Spinner ────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 2px; }

/* ── Diagram Toggle Tabs ────────────────────────────────────── */
.tab-row {
  display: flex; gap: 10px; margin: 10px 0;
}
.tab-btn {
  padding: 5px 14px; border-radius: 6px; font-size: 12px;
  font-family: var(--font-mono); cursor: pointer;
  border: 1px solid var(--border); background: var(--bg3); color: var(--text-dim);
}
.tab-btn.active { border-color: var(--cyan); color: var(--cyan); background: rgba(0,229,255,0.07); }

/* ── Tip Badges ─────────────────────────────────────────────── */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-family: var(--font-mono);
  background: rgba(0,229,255,0.08); border: 1px solid var(--border); color: var(--cyan);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 3. MERMAID RENDERER
# ═══════════════════════════════════════════════════════════════
def render_mermaid(diagram_code: str, height: int = 420):
    """
    Render a Mermaid diagram string inline via Streamlit HTML component.
    Uses the official Mermaid CDN – no extra packages needed.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <style>
        body {{
          margin: 0; padding: 12px;
          background: #0c1220;
          font-family: 'JetBrains Mono', monospace;
        }}
        .mermaid {{
          display: flex; justify-content: center; align-items: center;
        }}
        .mermaid svg {{
          max-width: 100%;
          border-radius: 10px;
        }}
        /* Mermaid dark theme overrides */
        .mermaid .label {{ color: #e2e8f0 !important; }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{diagram_code}
      </div>
      <script>
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'dark',
          themeVariables: {{
            primaryColor:       '#0c1220',
            primaryTextColor:   '#e2e8f0',
            primaryBorderColor: '#00e5ff',
            lineColor:          '#00e5ff',
            secondaryColor:     '#111827',
            tertiaryColor:      '#0c1220',
            edgeLabelBackground:'#0c1220',
            clusterBkg:         '#111827',
            titleColor:         '#00ff88',
            nodeTextColor:      '#e2e8f0',
            fontFamily:         'JetBrains Mono, monospace',
          }},
          flowchart: {{ useMaxWidth: true, htmlLabels: true }},
          sequence:  {{ useMaxWidth: true }},
        }});
      </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# 4. HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="
  background: linear-gradient(135deg, #0c1220 0%, #050810 100%);
  border: 1px solid rgba(0,229,255,0.2);
  border-radius: 14px;
  padding: 22px 28px;
  margin-bottom: 22px;
  box-shadow: 0 0 40px rgba(0,229,255,0.08);
  position: relative;
  overflow: hidden;
">
  <!-- decorative glow blobs -->
  <div style="position:absolute;top:-30px;right:-30px;width:150px;height:150px;
    background:radial-gradient(circle,rgba(0,229,255,0.12) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-20px;left:40px;width:120px;height:120px;
    background:radial-gradient(circle,rgba(0,255,136,0.08) 0%,transparent 70%);pointer-events:none;"></div>

  <div style="display:flex;align-items:center;gap:14px;">
    <span style="font-size:36px;filter:drop-shadow(0 0 10px #00e5ff);">⚡</span>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
        background:linear-gradient(90deg,#00e5ff,#00ff88);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        line-height:1.1;">
        MNNIT Study Mate
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
        color:#64748b;margin-top:3px;letter-spacing:0.05em;">
        // Powered by Hybrid RAG · Gemini · Groq · Pinecone
      </div>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
      <div style="width:8px;height:8px;border-radius:50%;background:#00ff88;
        box-shadow:0 0 8px #00ff88;animation:pulse 2s infinite;"></div>
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#00ff88;">LIVE</span>
    </div>
  </div>
</div>
<style>
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;
      background:linear-gradient(90deg,#00e5ff,#b06aff);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      margin-bottom:16px;">
      ⚙️ Settings
    </div>
    """, unsafe_allow_html=True)

    subject = st.selectbox(
        "📚 Subject",
        ["General", "Computer Science", "Artificial Intelligence",
         "Data Structure", "Electronics", "English"],
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    response_type = st.radio(
        "🎯 Answer Style",
        ["Detailed Explanation", "Step-by-Step", "Short Exam Answer"],
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Diagram toggle
    show_diagram = st.toggle("🗺️ Generate Flowchart / Diagram", value=True)

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.diagrams = {}
        st.rerun()

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
      color:#64748b;margin-top:12px;line-height:1.8;">
      <div>📡 Backend  <span style='color:#00ff88'>● Connected</span></div>
      <div>🧠 Model    <span style='color:#00e5ff'>Llama 3.1</span></div>
      <div>🗃️ Store    <span style='color:#b06aff'>Pinecone</span></div>
      <div>🔎 Embed    <span style='color:#ff6b35'>Gemini</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px;padding:10px 12px;border-radius:8px;
      background:rgba(0,229,255,0.04);border:1px solid rgba(0,229,255,0.1);
      font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;line-height:1.8;">
      <b style='color:#00e5ff;'>// Quick Tips</b><br>
      → Be specific in questions<br>
      → Use "Step-by-Step" for<br>&nbsp;&nbsp;&nbsp;algorithms<br>
      → Diagrams auto-generate<br>&nbsp;&nbsp;&nbsp;when relevant
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 6. SESSION STATE
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":    "assistant",
            "content": "// Hello! I'm your AI Study Mate.\n\nAsk me anything about your subjects — I'll explain clearly with diagrams when helpful. What are we studying today?",
        }
    ]

if "diagrams" not in st.session_state:
    # dict: message_index → mermaid string
    st.session_state.diagrams = {}


# ═══════════════════════════════════════════════════════════════
# 7. RENDER EXISTING CHAT
# ═══════════════════════════════════════════════════════════════
def render_message_pair(idx: int, msg: dict):
    """Render one chat message + diagram (if any) for the given index."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show diagram if this assistant message had one
        mermaid_code = st.session_state.diagrams.get(idx)
        if mermaid_code and msg["role"] == "assistant":
            with st.expander("📊 View Diagram", expanded=True):
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                  <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#00e5ff;">// Auto-generated concept diagram</span>
                  <span class="badge">Mermaid</span>
                </div>
                """, unsafe_allow_html=True)
                render_mermaid(mermaid_code)

                # Raw code toggle
                with st.expander("🔢 Raw Mermaid Code"):
                    st.code(mermaid_code, language="text")


for idx, msg in enumerate(st.session_state.messages):
    render_message_pair(idx, msg)


# ═══════════════════════════════════════════════════════════════
# 8. CHAT INPUT & BACKEND CALL
# ═══════════════════════════════════════════════════════════════
user_question = st.chat_input("// Ask your question here...")

if user_question:
    # ── User bubble ────────────────────────────────────────────
    user_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # ── Assistant bubble ───────────────────────────────────────
    asst_idx = len(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("⚡ Searching knowledge base..."):
            try:
                payload = {
                    "question":       user_question,
                    "subject":        subject,
                    "response_type":  response_type,
                    "include_diagram": show_diagram,
                }
                resp = requests.post(
                    "http://localhost:8000/api/ask",
                    json=payload,
                    timeout=60,
                )

                if resp.status_code == 200:
                    data        = resp.json()
                    answer      = data.get("answer", "No answer returned.")
                    mermaid_raw = data.get("mermaid")   # may be None
                else:
                    answer      = f"⚠️ Backend Error `{resp.status_code}` — Make sure FastAPI is running on port 8000."
                    mermaid_raw = None

            except requests.exceptions.ConnectionError:
                answer      = "⚠️ **Connection Error** — Cannot reach backend on `localhost:8000`.\n\nStart the FastAPI server: `uvicorn backend:app --reload`"
                mermaid_raw = None
            except requests.exceptions.Timeout:
                answer      = "⚠️ **Timeout** — Backend took too long to respond. Try again."
                mermaid_raw = None

        # ── Render answer ──────────────────────────────────────
        st.markdown(answer)

        # ── Render diagram if returned ─────────────────────────
        if mermaid_raw:
            st.session_state.diagrams[asst_idx] = mermaid_raw
            with st.expander("📊 View Diagram", expanded=True):
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                  <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#00e5ff;">// Auto-generated concept diagram</span>
                  <span class="badge">Mermaid</span>
                </div>
                """, unsafe_allow_html=True)
                render_mermaid(mermaid_raw)

                with st.expander("🔢 Raw Mermaid Code"):
                    st.code(mermaid_raw, language="text")

        elif show_diagram:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
              color:#64748b;margin-top:6px;">
              // No diagram generated for this response
            </div>
            """, unsafe_allow_html=True)

    # ── Save to state ──────────────────────────────────────────
    st.session_state.messages.append({"role": "assistant", "content": answer})
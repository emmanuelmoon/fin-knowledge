#streamlit run chat_app.py

import os
import time
import requests
import streamlit as st

# Backend URL (FastAPI)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FinKnowledge Chat",
    page_icon="🤖",
    layout="wide",
)

# ---------- Tiny CSS polish (centered chat, nicer background) ----------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        .stChatMessage {
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 FinKnowledge – Chat with Your Financial Reports")
st.caption("Upload financial PDFs, index them in the backend, and ask natural language questions.")


# ========================
# SESSION STATE
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user" | "assistant", "content": str}


# ========================
# SIDEBAR: UPLOAD + FILTERS
# ========================
with st.sidebar:
    st.header(" Upload & Index Report")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    company = st.text_input("Company Name")
    report_type = st.text_input("Report Type (e.g., Annual Report, 10-K)")
    fiscal_year = st.number_input("Fiscal Year", 1900, 2100, 2024)

    if st.button(" Index PDF"):
        if not (pdf_file and company and report_type):
            st.warning("Please upload a PDF and fill in company name + report type.")
        else:
            with st.spinner("Indexing PDF..."):
                files = {
                    "pdf_file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")
                }
                data = {
                    "company_name": company,
                    "report_type": report_type,
                    "fiscal_year": int(fiscal_year),
                }
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/index_document/",
                        files=files,
                        data=data,
                        timeout=300,
                    )
                    if res.status_code == 200:
                        st.success(" PDF indexed successfully!")
                        st.json(res.json())
                    else:
                        st.error(" Failed to index PDF.")
                        st.write(res.text)
                except Exception as e:
                    st.error("Backend error while indexing.")
                    st.exception(e)

    st.markdown("---")
    st.header(" Query Filters (optional)")
    st.caption("These filters will be applied to every chat question.")

    filter_company = st.text_input("Filter by Company")
    filter_year = st.text_input("Filter by Fiscal Year")
    filter_report_type = st.text_input("Filter by Report Type")

    if st.button(" Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")


# ========================
# RENDER CHAT HISTORY
# ========================
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


# ========================
# USER INPUT (CHATGPT STYLE)
# ========================
prompt = st.chat_input("Ask something about your financial reports...")

if prompt:
    # 1) Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Build payload with filters
    payload = {"query": prompt}

    if filter_company.strip():
        payload["company_name"] = filter_company.strip()
    if filter_year.strip():
        try:
            payload["fiscal_year"] = int(filter_year.strip())
        except ValueError:
            # Just ignore bad year input
            pass
    if filter_report_type.strip():
        payload["report_type"] = filter_report_type.strip()

    # 3) Assistant "typing" + lightweight streaming effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        typing_placeholder = st.empty()

        # Show animated "typing" dots while waiting for backend
        with st.spinner("Assistant is thinking..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/query/",
                    json=payload,
                    timeout=180,
                )
                if res.status_code == 200:
                    full_reply = res.json().get("response", "")
                else:
                    full_reply = f" Backend error: {res.status_code}\n\n{res.text}"
            except Exception as e:
                full_reply = f" Failed to reach backend:\n\n{e}"

        typing_placeholder.empty()  # remove spinner text

        # Streaming-style display: reveal text gradually
        streamed_text = ""
        for chunk in full_reply.split(" "):
            streamed_text += chunk + " "
            message_placeholder.markdown(streamed_text + "▌")
            time.sleep(0.015)  # adjust for faster / slower streaming

        # Final message without cursor
        message_placeholder.markdown(streamed_text)

    # 4) Save assistant message in chat history
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

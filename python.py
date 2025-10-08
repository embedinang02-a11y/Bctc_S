import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ============ CẤU HÌNH GEMINI ============
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ============ CẤU HÌNH GIAO DIỆN APP ============
st.set_page_config(page_title="Phân tích Tài chính & Gemini AI", layout="wide")

# CSS: Tùy chỉnh màu sắc và bong bóng chat
st.markdown("""
<style>
/* Nền tổng thể */
body {
    background-color: #f5f6fa;
    color: #333;
}

/* Bong bóng tin nhắn người dùng */
.chat-bubble-user {
    background-color: #d1f1ff;
    color: #000;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
    width: fit-content;
    float: right;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
}

/* Bong bóng tin nhắn AI */
.chat-bubble-ai {
    background-color: #fff;
    color: #333;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    width: fit-content;
    float: left;
    border: 1px solid #ddd;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
}

/* Avatar */
.avatar {
    border-radius: 50%;
    height: 32px;
    width: 32px;
    object-fit: cover;
    margin: 0 8px;
    vertical-align: middle;
}

/* Khung chat tổng */
.chat-container {
    background-color: #fdfdfd;
    border-radius: 12px;
    padding: 10px 15px;
    height: 480px;
    overflow-y: auto;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ============ GIAO DIỆN CHÍNH ============
st.title("📊 Ứng dụng Phân tích Báo cáo Tài chính + Trợ lý Gemini AI")

col1, col2 = st.columns([2, 1])

# --- PHẦN PHÂN TÍCH DỮ LIỆU ---
with col1:
    st.subheader("📁 Phân tích Báo cáo Tài chính")
    uploaded_file = st.file_uploader("Tải lên file Excel", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        if "Năm" in df.columns and "Doanh thu" in df.columns:
            plt.figure()
            plt.plot(df["Năm"], df["Doanh thu"], label="Doanh thu", marker='o', color="#007bff")
            if "Lợi nhuận" in df.columns:
                plt.plot(df["Năm"], df["Lợi nhuận"], label="Lợi nhuận", marker='s', color="#28a745")
            plt.legend()
            plt.title("📈 Xu hướng Doanh thu và Lợi nhuận")
            plt.grid(True)
            st.pyplot(plt)

        st.markdown("### 📊 Thống kê tổng hợp")
        st.write(df.describe())

# --- PHẦN CHAT GEMINI ---
with col2:
    st.subheader("💬 Trợ lý Gemini AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị khung chat
    st.markdown('<div class="chat-container" id="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div

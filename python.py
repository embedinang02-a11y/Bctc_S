import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

# Cấu hình API Gemini
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Giao diện chính
st.set_page_config(page_title="Phân tích tài chính & Chat AI", layout="wide")

st.title("📊 Ứng dụng Phân tích Báo cáo Tài chính + Trợ lý Gemini AI")

# --- CỘT TRÁI: PHÂN TÍCH DỮ LIỆU ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Phân tích Báo cáo Tài chính")
    uploaded_file = st.file_uploader("📂 Tải lên file Excel báo cáo tài chính", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        if "Năm" in df.columns and "Doanh thu" in df.columns:
            plt.figure()
            plt.plot(df["Năm"], df["Doanh thu"], label="Doanh thu", marker='o')
            if "Lợi nhuận" in df.columns:
                plt.plot(df["Năm"], df["Lợi nhuận"], label="Lợi nhuận", marker='s')
            plt.legend()
            plt.title("Xu hướng Doanh thu và Lợi nhuận")
            st.pyplot(plt)

        # Tóm tắt cơ bản
        st.write("**📈 Thống kê mô tả:**")
        st.write(df.describe())

# --- CỘT PHẢI: KHUNG CHAT GEMINI ---
with col2:
    st.subheader("💬 Chat với Gemini AI")
    st.write("Bạn có thể hỏi về dữ liệu hoặc các chỉ số tài chính.")

    # Giữ lịch sử hội thoại
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Hộp nhập tin nhắn
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    if user_input:
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        try:
            # Gọi API Gemini
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_input)

            ai_reply = response.text
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            st.chat_message("assistant").write(ai_reply)

        except Exception as e:
            st.error(f"Lỗi khi kết nối Gemini: {e}")

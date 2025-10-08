import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính (Có Biểu đồ & Tỷ số Nhanh)")

# Khởi tạo Session State cho lịch sử chat và ngữ cảnh dữ liệu
if "chat_history" not in st.session_state:
    # Khởi tạo tin nhắn chào mừng
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý Phân tích Tài chính Gemini. Hãy tải lên Báo cáo Tài chính ở tab bên cạnh để tôi có thể hỗ trợ phân tích và giải đáp thắc mắc!"}
    ]
if "df_markdown" not in st.session_state:
    st.session_state.df_markdown = "" # Lưu trữ dữ liệu đã xử lý dưới dạng Markdown để làm ngữ cảnh cho AI

# Lấy API Key từ Streamlit Secrets
# Lưu ý: Cần cấu hình key GEMINI_API_KEY trong file secrets.toml của Streamlit
api_key = st.secrets.get("GEMINI_API_KEY")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        # Xử lý các lỗi khi chuyển đổi thành số (như chuỗi trống hoặc dấu phẩy)
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', ''), 
            errors='coerce'
        ).fillna(0)
            
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý chia cho 0 cho Tỷ trọng (Nếu Tổng Tài sản bằng 0, không thể tính tỷ trọng)
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else float('nan')
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else float('nan')

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df.fillna(0) # Thay thế NaN bằng 0 sau khi tính toán (trừ NaN từ division by zero)

# --- Hàm gọi API Gemini cho phân tích tự động ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tự động."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành (Current Ratio) và thanh toán nhanh (Quick Ratio).
                                    
Dữ liệu thô và chỉ số:
{data_for_ai}
"""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM MỚI: Xử lý Chatbot Q&A với Ngữ cảnh ---
def get_chat_response(prompt, processed_data_markdown, chat_history, api_key):
    """Gửi câu hỏi của người dùng cùng với ngữ cảnh dữ liệu và lịch sử chat đến Gemini API."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        # 1. System Instruction: Define persona and rules
        system_instruction = (
            "Bạn là Trợ lý Tài chính AI (FA-Gemini) chuyên nghiệp. Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng "
            "dựa trên Dữ liệu Báo cáo Tài chính đã được xử lý và cung cấp dưới dạng Markdown. "
            "Hãy giữ giọng điệu chuyên nghiệp, chính xác và chỉ trả lời dựa trên dữ liệu hiện có trong bảng Markdown. "
            "KHÔNG TỰ Ý SÁNG TẠO DỮ LIỆU HOẶC CHỈ SỐ. Nếu thông tin không có, hãy nói rõ rằng bạn chỉ có thể phân tích dữ liệu đã cung cấp."
        )
        
        # 2. Chuẩn bị nội dung gửi đi (Lịch sử Chat + Prompt hiện tại với Context)
        full_contents = []
        
        # Chuyển đổi lịch sử chat của Streamlit sang định dạng API
        for message in chat_history:
            if "content" in message and message["role"] in ["user", "assistant"]:
                role = "user" if message["role"] == "user" else "model"
                full_contents.append({"role": role, "parts": [{"text": message["content"]}]})

        # 3. Gắn ngữ cảnh dữ liệu vào tin nhắn cuối cùng của người dùng (tin nhắn hiện tại)
        user_prompt_with_context = f"Đây là Bảng Dữ liệu Tài chính đã được phân tích:\n{processed_data_markdown}\n\nCâu hỏi của tôi: {prompt}"
        
        # Tin nhắn cuối cùng trong full_contents là tin nhắn người dùng hiện tại (với nội dung thô là 'prompt')
        if full_contents and full_contents[-1]["role"] == "user":
             # Thay thế nội dung thô bằng nội dung đã gắn context
             full_contents[-1]["parts"][0]["text"] = user_prompt_with_context
        else:
             # Trường hợp lỗi: Thêm tin nhắn người dùng mới nếu không tìm thấy (nên được xử lý ở đây)
             full_contents.append({"role": "user", "parts": [{"text": user_prompt_with_context}]})

        # 4. Gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=full_contents,
            system_instruction=system_instruction
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Bắt đầu luồng ứng dụng chính: Tách thành 2 Tabs ---
tab1, tab2 = st.tabs([" PHÂN TÍCH TỰ ĐỘNG & CHỈ SỐ", " TRỢ LÝ TÀI CHÍNH AI (Q&A)"])

# --- Tab 1: Phân tích Tự động & Nhận xét AI ---
with tab1:
    # Chức năng 1: Tải File
    uploaded_file = st.file_uploader(
        "1. Tải file Excel Báo cáo Tài chính (Cột: Chỉ tiêu | Năm trước | Năm sau)",
        type=['xlsx', 'xls']
    )

    df_processed = None
    
    # Xử lý khi không có file được tải lên
    if uploaded_file is None:
        st.session_state.df_markdown = ""
        st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            
            # Xử lý dữ liệu
            df_processed = process_financial_data(df_raw.copy())

            # Cập nhật dữ liệu đã xử lý vào session state cho Chatbot sử dụng
            st.session_state.df_markdown = df_processed.to_markdown(index=False)
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            if df_processed is not None:
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Khả năng Thanh toán Cơ bản")
                
                # Biến lưu trữ kết quả tính toán
                thanh_toan_hien_hanh_N = None
                thanh_toan_hien_hanh_N_1 = None
                quick_ratio_N = None
                quick_ratio_N_1 = None
                
                # Tên cột để hiển thị delta
                delta_cr = None
                delta_qr = None

                try:
                    # Lọc giá trị cho Tài sản ngắn hạn (TSNH) và Nợ ngắn hạn (NNH)
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                    
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0] 
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # --- 4a. TÍNH CURRENT RATIO ---
                    if no_ngan_han_N != 0:
                        thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    if no_ngan_han_N_1 != 0:
                        thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                    
                    # Tính Delta Current Ratio
                    if thanh_toan_hien_hanh_N is not None and thanh_toan_hien_hanh_N_1 is not None:
                        delta_cr = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                        
                    # --- 4b. TÍNH QUICK RATIO (Tỷ số Thanh toán Nhanh) ---
                    # Lọc giá trị cho Hàng tồn kho (HTK)
                    htk_n = df_processed[df_processed['Chỉ tiêu'].str.contains('HÀNG TỒN KHO', case=False, na=False)]['Năm sau'].iloc[0]
                    htk_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('HÀNG TỒN KHO', case=False, na=False)]['Năm trước'].iloc[0]

                    # Công thức: (TSNH - HTK) / NNH
                    if no_ngan_han_N != 0:
                        quick_ratio_N = (tsnh_n - htk_n) / no_ngan_han_N
                    if no_ngan_han_N_1 != 0:
                        quick_ratio_N_1 = (tsnh_n_1 - htk_n_1) / no_ngan_han_N_1

                    # Tính Delta Quick Ratio
                    if quick_ratio_N is not None and quick_ratio_N_1 is not None:
                        delta_qr = f"{quick_ratio_N - quick_ratio_N_1:.2f}"

                    # --- HIỂN THỊ METRICS ---
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Current Ratio N-1
                    with col1:
                        st.metric(
                            label="Thanh toán Hiện hành (Năm trước)",
                            value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 is not None else "N/A"
                        )
                    # Current Ratio N
                    with col2:
                        st.metric(
                            label="Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N is not None else "N/A",
                            delta=delta_cr
                        )
                    # Quick Ratio N-1
                    with col3:
                        st.metric(
                            label="Thanh toán Nhanh (Năm trước)",
                            value=f"{quick_ratio_N_1:.2f} lần" if quick_ratio_N_1 is not None else "N/A"
                        )
                    # Quick Ratio N
                    with col4:
                        st.metric(
                            label="Thanh toán Nhanh (Năm sau)",
                            value=f"{quick_ratio_N:.2f} lần" if quick_ratio_N is not None else "N/A",
                            delta=delta_qr
                        )

                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN', 'NỢ NGẮN HẠN' hoặc 'HÀNG TỒN KHO' để tính chỉ số.")
                except Exception as e:
                    st.error(f"Lỗi tính toán chỉ số: {e}")

                
                # --- Chức năng MỚI: Biểu đồ Trực quan hóa ---
                st.subheader("5. Biểu đồ Trực quan hóa Tăng trưởng")
                
                # Lọc các chỉ tiêu quan trọng để vẽ biểu đồ 
                # Loại bỏ các chỉ tiêu TỔNG CỘNG và các dòng có tăng trưởng gần bằng 0 để biểu đồ dễ nhìn
                df_chart = df_processed[
                    (df_processed['Tốc độ tăng trưởng (%)'].abs() > 0.5) & 
                    (~df_processed['Chỉ tiêu'].str.contains('TỔNG CỘNG', case=False, na=False))
                ].sort_values(by='Tốc độ tăng trưởng (%)', ascending=False).head(10)

                if not df_chart.empty:
                    st.markdown("##### 📈 Top 10 Chỉ tiêu có Tốc độ Tăng trưởng thay đổi lớn nhất (%)")
                    # Tùy chỉnh cột cho biểu đồ
                    df_chart_styled = df_chart[['Chỉ tiêu', 'Tốc độ tăng trưởng (%)']].set_index('Chỉ tiêu')
                    st.bar_chart(df_chart_styled, color='#307FFF')
                else:
                    st.info("Không đủ dữ liệu hoặc không có sự thay đổi đáng kể (>0.5%) để vẽ biểu đồ Tăng trưởng.")


                # --- Chức năng 6: Nhận xét AI ---
                st.subheader("6. Nhận xét Tình hình Tài chính (AI)")
                if st.button("Yêu cầu AI Phân tích"):
                    if api_key:
                        # Chuẩn bị dữ liệu để gửi cho AI 
                        data_for_ai = pd.DataFrame({
                            'Chỉ tiêu': [
                                'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                                'Thanh toán hiện hành (N-1)', 
                                'Thanh toán hiện hành (N)',
                                'Thanh toán nhanh (N-1)', 
                                'Thanh toán nhanh (N)'
                            ],
                            'Giá trị': [
                                st.session_state.df_markdown,
                                f"{thanh_toan_hien_hanh_N_1}" if thanh_toan_hien_hanh_N_1 is not None else "N/A", 
                                f"{thanh_toan_hien_hanh_N}" if thanh_toan_hien_hanh_N is not None else "N/A",
                                f"{quick_ratio_N_1}" if quick_ratio_N_1 is not None else "N/A", 
                                f"{quick_ratio_N}" if quick_ratio_N is not None else "N/A"
                            ]
                        }).to_markdown(index=False)

                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
            st.session_state.df_markdown = "" # Reset context
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
            st.session_state.df_markdown = "" # Reset context


# --- Tab 2: Trợ lý Tài chính AI (Q&A) ---
with tab2:
    st.subheader("Trò chuyện với Trợ lý Tài chính AI (FA-Gemini)")
    
    # Hiển thị tất cả tin nhắn trong lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý đầu vào từ người dùng
    if prompt := st.chat_input("Hỏi về Tốc độ tăng trưởng, Tỷ trọng tài sản, hoặc bất kỳ chỉ tiêu nào trong bảng...") :
        # 1. Kiểm tra điều kiện cần thiết trước khi xử lý
        if not api_key:
            st.warning("Không có Khóa API Gemini, không thể trò chuyện. Vui lòng kiểm tra cấu hình Secrets.")
        elif st.session_state.df_markdown == "":
            st.warning("Vui lòng tải lên và xử lý báo cáo tài chính ở tab 'PHÂN TÍCH TỰ ĐỘNG' trước khi hỏi đáp.")
        else:
            # 2. Thêm tin nhắn người dùng vào lịch sử và hiển thị
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 3. Lấy phản hồi từ Gemini và hiển thị
            with st.chat_message("assistant"):
                # Sử dụng khối try/except để bắt và hiển thị lỗi API/mạng nếu có
                try:
                    with st.spinner("Đang chờ FA-Gemini trả lời..."):
                        # Gọi hàm chat với ngữ cảnh dữ liệu và lịch sử chat
                        response = get_chat_response(
                            prompt, 
                            st.session_state.df_markdown, 
                            st.session_state.chat_history,
                            api_key
                        )
                    st.markdown(response)
                    # 4. Thêm phản hồi của AI vào lịch sử
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                     error_message = f"Đã xảy ra lỗi trong quá trình gọi API/xử lý: {e}. Vui lòng thử lại."
                     st.error(error_message)
                     # Lưu ý: Khi lỗi xảy ra, tin nhắn người dùng vẫn nằm trong lịch sử, 
                     # nhưng tin nhắn trả lời lỗi này sẽ không được thêm vào lịch sử để tránh làm phức tạp API.
                     
    if st.session_state.df_markdown != "":
        st.caption("Dữ liệu báo cáo đã xử lý đang được cung cấp cho AI để trả lời các câu hỏi chuyên sâu của bạn.")
    else:
        st.caption("Chưa có dữ liệu nào được tải lên. Chatbot sẽ chỉ có thể trả lời các câu hỏi chung chung.")

if not api_key:
    st.sidebar.error("Cảnh báo: Không tìm thấy Khóa 'GEMINI_API_KEY'. Vui lòng cấu hình trên Streamlit Secrets.")

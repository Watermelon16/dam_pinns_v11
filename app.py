import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
os.makedirs("data", exist_ok=True)  # tạo thư mục data nếu chưa tồn tại
import base64
from io import BytesIO
import sqlite3
import threading
from contextlib import contextmanager

# Import các module tùy chỉnh
from pinns_optimizer import optimize_dam_section, create_force_diagram_plotly, plot_loss_curve
from database import DamDatabase

# Khởi tạo cơ sở dữ liệu
def get_database():
    db = DamDatabase("data/dam_results.db")
    db.create_tables()  # Thêm dòng này để tạo bảng nếu chưa có
    return db

# Hàm tạo báo cáo Excel
def create_excel_report(result):
    # Tạo DataFrame cho báo cáo
    data = {
        'Thông số': [
            'Chiều cao đập (H)',
            'Trọng lượng riêng bê tông (γ_bt)',
            'Trọng lượng riêng nước (γ_n)',
            'Hệ số ma sát (f)',
            'Cường độ kháng cắt (C)',
            'Hệ số ổn định yêu cầu (Kc)',
            'Hệ số áp lực thấm (a1)',
            'Hệ số mái thượng lưu (n)',
            'Hệ số mái hạ lưu (m)',
            'Tham số ξ',
            'Diện tích mặt cắt (A)',
            'Hệ số ổn định (K)',
            'Ứng suất mép thượng lưu (σ)',
            'Chiều rộng đáy đập (B)',
            'Độ lệch tâm (e)',
            'Lực đẩy nổi (W1)',
            'Lực đẩy ngang (W2)',
            'Áp lực thấm (Wt)',
            'Trọng lượng đập (G)',
            'Lực chống trượt (Fct)',
            'Lực gây trượt (Fgt)',
            'Số vòng lặp thực tế',
            'Số vòng lặp tối đa',
            'Thời gian tính toán (giây)'
        ],
        'Giá trị': [
            result['H'],
            result['gamma_bt'],
            result['gamma_n'],
            result['f'],
            result['C'],
            result['Kc'],
            result['a1'],
            result['n'],
            result['m'],
            result['xi'],
            result['A'],
            result['K'],
            result['sigma'],
            result['B'],
            result['e'],
            result['W1'],
            result['W2'],
            result['Wt'],
            result['G'],
            result['Fct'],
            result['Fgt'],
            result['iterations'],
            result.get('max_iterations', 5000),
            result.get('computation_time', 0)
        ],
        'Đơn vị': [
            'm',
            'T/m³',
            'T/m³',
            '',
            'T/m²',
            '',
            '',
            '',
            '',
            '',
            'm²',
            '',
            'T/m²',
            'm',
            'm',
            'T/m',
            'T/m',
            'T/m',
            'T/m',
            'T/m',
            'T/m',
            'vòng',
            'vòng',
            's'
        ]
    }
    
    return pd.DataFrame(data)

# Hàm tạo link tải xuống Excel
def get_excel_download_link(df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Báo cáo')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Tải xuống báo cáo Excel</a>'
    return href

# Thiết lập trang Streamlit
st.set_page_config(
    page_title="Tối ưu mặt cắt đập bê tông trọng lực",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề ứng dụng
st.title("🏞️ Tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs")
st.markdown("""
Ứng dụng này sử dụng mô hình Physics-Informed Neural Networks (PINNs) để tính toán mặt cắt kinh tế đập bê tông trọng lực thỏa mãn các điều kiện:
- Diện tích mặt cắt A là nhỏ nhất (hàm mục tiêu cần tối ưu)
- Thỏa mãn điều kiện ổn định trượt: K=Kc
- Thỏa mãn điều kiện ứng suất mép thượng lưu: σ≈0 (không có ứng suất kéo)
""")

# Tạo tabs
tabs = st.tabs(["Tính toán mới", "Lịch sử tính toán"])

# Tab Tính toán mới
with tabs[0]:
    # Khởi tạo giá trị mặc định
    default_values = {
        'H': 30.0,
        'gamma_bt': 2.4,
        'gamma_n': 1.0,
        'f': 0.7,
        'C': 0.0,
        'Kc': 1.2,
        'a1': 0.5,
        'max_iterations': 5000,
        'convergence_threshold': 1e-6,
        'patience': 50
    }
    
    # Khởi tạo session state nếu chưa có
    for k, v in default_values.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # Chia layout thành 2 cột
    col1, col2 = st.columns([1, 2])
    
    # Cột 1: Form nhập liệu
    with col1:
        st.markdown("### Thông số tính toán")
        
        # Form nhập liệu
        with st.form("input_form"):
            # Thông số đập
            H = st.number_input(
                "Chiều cao đập (m)", 
                min_value=10.0, 
                max_value=300.0, 
                value=st.session_state.H, 
                step=10.0
            )
            gamma_bt = st.number_input(
                "Trọng lượng riêng bê tông (T/m³)", 
                min_value=2.0, 
                max_value=3.0, 
                value=st.session_state.gamma_bt, 
                step=0.1
            )
            gamma_n = st.number_input(
                "Trọng lượng riêng nước (T/m³)", 
                min_value=0.9, 
                max_value=1.1, 
                value=st.session_state.gamma_n, 
                step=0.1
            )
            
            # Thông số vật liệu
            f = st.number_input(
                "Hệ số ma sát", 
                min_value=0.3, 
                max_value=2.0, 
                value=st.session_state.f, 
                step=0.05
            )
            C = st.number_input(
                "Cường độ kháng cắt (T/m²)", 
                min_value=0.0, 
                max_value=50.0, 
                value=st.session_state.C, 
                step=5.0
            )
            Kc = st.number_input(
                "Hệ số ổn định yêu cầu", 
                min_value=1.1, 
                max_value=4.0, 
                value=st.session_state.Kc, 
                step=0.1
            )
            a1 = st.number_input(
                "Hệ số áp lực thấm", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.a1, 
                step=0.01
            )
            
            # Thông số tính toán
            with st.expander("Thông số tính toán"):
                max_iterations = st.slider(
                    "Số vòng lặp tối đa", 
                    min_value=1000, 
                    max_value=10000, 
                    value=st.session_state.max_iterations, 
                    step=1000
                )
                convergence_threshold = st.number_input(
                    "Ngưỡng hội tụ", 
                    min_value=1e-8, 
                    max_value=1e-4, 
                    value=st.session_state.convergence_threshold, 
                    format="%.1e"
                )
                patience = st.slider(
                    "Số vòng lặp kiên nhẫn", 
                    min_value=10, 
                    max_value=200, 
                    value=st.session_state.patience, 
                    step=10,
                    help="Số vòng lặp chờ đợi khi không có cải thiện trước khi dừng sớm"
                )
            
                convergence_threshold = st.session_state.convergence_threshold
                patience = st.session_state.patience
            
            # Nút tính toán
            submitted = st.form_submit_button("Tính toán tối ưu")
        
        # Nút đặt lại
        reset_clicked = st.button("🔄 Đặt lại")
        if reset_clicked:
            for k, v in default_values.items():
                st.session_state[k] = v
            st.success("Đã reset lại các giá trị!")  # Thông báo trực quan khi đặt lại
    
        # Xử lý khi form được gửi
        if submitted:
            # Cập nhật session state
            st.session_state.H = H
            st.session_state.gamma_bt = gamma_bt
            st.session_state.gamma_n = gamma_n
            st.session_state.f = f
            st.session_state.C = C
            st.session_state.Kc = Kc
            st.session_state.a1 = a1
            st.session_state.max_iterations = max_iterations
            st.session_state.convergence_threshold = convergence_threshold
            st.session_state.patience = patience
            
            with st.spinner("Đang tính toán tối ưu mặt cắt đập..."):
                # Ghi lại thời gian bắt đầu
                start_time = datetime.now()
                
                # Thực hiện tính toán
                try:
                    result = optimize_dam_section(
                        H=H,
                        gamma_bt=gamma_bt,
                        gamma_n=gamma_n,
                        f=f,
                        C=C,
                        Kc=Kc,
                        a1=a1,
                        max_iterations=max_iterations,
                        convergence_threshold=convergence_threshold,
                        patience=patience
                    )
                    
                    # Tính thời gian tính toán
                    computation_time = (datetime.now() - start_time).total_seconds()
                    result['computation_time'] = computation_time
                    
                    # Lưu kết quả vào session state
                    st.session_state['result'] = result
                    
                    try:
                        # Lưu kết quả vào cơ sở dữ liệu
                        db = get_database()
                        result_id = db.save_result(result)
                        st.session_state['last_result_id'] = result_id
                        st.success(f"Đã lưu kết quả tính toán vào cơ sở dữ liệu (ID: {result_id})")
                    except Exception as e:
                        st.warning(f"Không thể lưu kết quả vào cơ sở dữ liệu: {str(e)}")
                except Exception as e:
                    st.error(f"Lỗi trong quá trình tính toán: {str(e)}")
                    st.exception(e)
    
    # Hiển thị kết quả nếu có
    with col2:
        if 'result' in st.session_state and st.session_state['result'] is not None:
            result = st.session_state['result']
            
            st.markdown("### Kết quả tính toán")
            
            # Hiển thị các tham số tối ưu
            col_params1, col_params2 = st.columns(2)
            
            with col_params1:
                st.metric("Hệ số mái thượng lưu (n)", f"{result['n']:.3f}")
                st.metric("Hệ số mái hạ lưu (m)", f"{result['m']:.3f}")
                st.metric("Tham số ξ", f"{result['xi']:.3f}")
            
            with col_params2:
                st.metric("Diện tích mặt cắt (A)", f"{result['A']:.2f} m²")
                st.metric("Hệ số ổn định (K)", f"{result['K']:.2f}")
                st.metric("Ứng suất mép thượng lưu (σ)", f"{result['sigma']:.2f} T/m²")
            
            # Hiển thị trạng thái
            if abs(result['K'] - result['Kc']) < 0.05:  # Sai số cho phép 5%
                st.success(f"Mặt cắt đập thỏa mãn điều kiện ổn định (K = {result['K']:.2f} ≈ Kc = {result['Kc']:.2f})")
            elif result['K'] > result['Kc']:
                st.info(f"Mặt cắt đập thỏa mãn điều kiện ổn định (K = {result['K']:.2f} > Kc = {result['Kc']:.2f})")
            else:
                st.error(f"Mặt cắt đập KHÔNG thỏa mãn điều kiện ổn định (K = {result['K']:.2f} < Kc = {result['Kc']:.2f})")
            
            if result['sigma'] <= 0:
                st.success(f"Mặt cắt đập thỏa mãn điều kiện không kéo (σ = {result['sigma']:.2f} T/m² ≤ 0)")
            else:
                st.warning(f"Mặt cắt đập có ứng suất kéo ở mép thượng lưu (σ = {result['sigma']:.2f} T/m² > 0)")
            
            # Hiển thị thông tin về số vòng lặp
            st.info(f"Số vòng lặp thực tế: {result['iterations']} / {result.get('max_iterations', max_iterations)} (tối đa)")
            
            # Hiển thị thời gian tính toán
            st.info(f"Thời gian tính toán: {result['computation_time']:.2f} giây")
            
            # Tạo tabs cho các biểu đồ
            result_tabs = st.tabs(["Mặt cắt đập", "Biểu đồ hàm mất mát", "Xuất báo cáo"])
            
            # Tab mặt cắt đập
            with result_tabs[0]:
                # Tạo biểu đồ Plotly tương tác
                try:
                    fig = create_force_diagram_plotly(
                        H=result['H'],
                        n=result['n'],
                        m=result['m'],
                        xi=result['xi']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Không thể tạo biểu đồ mặt cắt đập: {str(e)}")
                    st.exception(e)
            
            # Tab biểu đồ hàm mất mát
            with result_tabs[1]:
                # Tạo biểu đồ Plotly tương tác
                try:
                    if 'loss_history' in result and len(result['loss_history']) > 0:
                        loss_fig = plot_loss_curve(result['loss_history'])
                        st.plotly_chart(loss_fig, use_container_width=True)
                    else:
                        st.warning("Không có dữ liệu lịch sử hàm mất mát để hiển thị")
                except Exception as e:
                    st.error(f"Không thể tạo biểu đồ hàm mất mát: {str(e)}")
                    st.exception(e)
            
            # Tab xuất báo cáo
            with result_tabs[2]:
                st.markdown("### Xuất báo cáo")
                
                # Tạo báo cáo Excel
                excel_df = create_excel_report(result)
                
                # Hiển thị báo cáo
                st.dataframe(excel_df, use_container_width=True)
                
                # Tạo link tải xuống
                st.markdown(
                    get_excel_download_link(excel_df, f"bao_cao_dam_H{int(result['H'])}.xlsx"),
                    unsafe_allow_html=True
                )

# Tab Lịch sử tính toán
with tabs[1]:
    st.markdown("### Lịch sử tính toán")
    
    try:
        # Lấy dữ liệu từ cơ sở dữ liệu
        db = get_database()
        history_df = db.get_all_results()
        
        if len(history_df) > 0:
            # Hiển thị bảng lịch sử
            st.dataframe(
                history_df[['id', 'timestamp', 'H', 'gamma_bt', 'gamma_n', 'f', 'C', 'Kc', 'a1', 'n', 'm', 'xi', 'A', 'K', 'sigma']],
                use_container_width=True
            )
            
            # Chọn kết quả để xem chi tiết
            selected_id = st.selectbox("Chọn ID để xem chi tiết:", history_df['id'].tolist())
            
            if st.button("Xem chi tiết"):
                # Lấy kết quả từ cơ sở dữ liệu
                selected_result = db.get_result_by_id(selected_id)
                
                if selected_result:
                    # Hiển thị thông tin chi tiết
                    st.markdown("#### Thông tin chi tiết")
                    
                    # Tạo DataFrame từ kết quả
                    detail_df = pd.DataFrame({
                        'Thông số': [
                            'Chiều cao đập (H)',
                            'Trọng lượng riêng bê tông (γ_bt)',
                            'Trọng lượng riêng nước (γ_n)',
                            'Hệ số ma sát (f)',
                            'Cường độ kháng cắt (C)',
                            'Hệ số ổn định yêu cầu (Kc)',
                            'Hệ số áp lực thấm (a1)',
                            'Hệ số mái thượng lưu (n)',
                            'Hệ số mái hạ lưu (m)',
                            'Tham số ξ',
                            'Diện tích mặt cắt (A)',
                            'Hệ số ổn định (K)',
                            'Ứng suất mép thượng lưu (σ)',
                            'Số vòng lặp thực tế',
                            'Số vòng lặp tối đa',
                        ],
                        'Giá trị': [
                            selected_result['H'],
                            selected_result['gamma_bt'],
                            selected_result['gamma_n'],
                            selected_result['f'],
                            selected_result['C'],
                            selected_result['Kc'],
                            selected_result['a1'],
                            selected_result['n'],
                            selected_result['m'],
                            selected_result['xi'],
                            selected_result['A'],
                            selected_result['K'],
                            selected_result['sigma'],
                            selected_result['iterations'],
                            selected_result.get('max_iterations', 5000),
                        ]
                    })
                    
                    # Hiển thị bảng thông tin chi tiết
                    st.dataframe(detail_df, use_container_width=True)
                    
                    # Tạo biểu đồ mặt cắt đập
                    try:
                        fig = create_force_diagram(selected_result)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Không thể tạo biểu đồ mặt cắt đập: {str(e)}")
                        st.exception(e)
                else:
                    st.error(f"Không tìm thấy kết quả với ID = {selected_id}")
        else:
            st.info("Chưa có kết quả tính toán nào được lưu trong cơ sở dữ liệu")
    except Exception as e:
        st.error(f"Lỗi khi truy cập cơ sở dữ liệu: {str(e)}")
        st.exception(e)

# Thông tin về ứng dụng
with st.sidebar:
    st.markdown("### Thông tin")
    st.markdown("""
    **Ứng dụng tối ưu mặt cắt đập bê tông trọng lực sử dụng PINNs**
    
    Ứng dụng này sử dụng mô hình Physics-Informed Neural Networks (PINNs) để tính toán mặt cắt kinh tế đập bê tông trọng lực thỏa mãn các điều kiện ổn định.
    
    **Các tính năng chính:**
    - Tính toán tối ưu mặt cắt đập bê tông trọng lực
    - Hiển thị biểu đồ mặt cắt đập và sơ đồ lực
    - Hiển thị biểu đồ hàm mất mát
    - Xuất báo cáo Excel
    - Lưu trữ và xem lại lịch sử tính toán
    
    **Phương pháp PINNs:**
    - Sử dụng mạng nơ-ron để tìm bộ tham số tối ưu (n, m, ξ)
    - Kết hợp các ràng buộc vật lý vào hàm mất mát
    - Tối ưu hóa đồng thời cả hàm mục tiêu và các ràng buộc
    """)
    
    st.markdown("---")
    st.markdown("© 2025 - Ứng dụng tính toán mặt cắt kinh tế đập bê tông trọng lực")

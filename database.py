import sqlite3
import pandas as pd
import os
from datetime import datetime
import threading
from contextlib import contextmanager

# Thread-local storage để lưu trữ kết nối cơ sở dữ liệu cho mỗi thread
thread_local = threading.local()

class DamDatabase:
    def __init__(self, db_path):
        """
        Khởi tạo kết nối đến cơ sở dữ liệu SQLite
        
        Parameters:
        -----------
        db_path : str
            Đường dẫn đến file cơ sở dữ liệu
        """
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.lock = threading.RLock()
    
    @contextmanager
    def get_connection(self):
        """
        Tạo và quản lý kết nối cơ sở dữ liệu an toàn cho thread
        
        Yields:
        -------
        sqlite3.Connection
            Kết nối đến cơ sở dữ liệu
        """
        # Kiểm tra xem thread hiện tại đã có kết nối chưa
        if not hasattr(thread_local, 'connection'):
            # Tạo kết nối mới cho thread hiện tại
            thread_local.connection = sqlite3.connect(self.db_path)
        
        try:
            # Trả về kết nối cho context
            yield thread_local.connection
        finally:
            # Không đóng kết nối ở đây để tái sử dụng trong cùng một thread
            pass
    
    def close_all_connections(self):
        """Đóng tất cả các kết nối cơ sở dữ liệu"""
        if hasattr(thread_local, 'connection'):
            thread_local.connection.close()
            del thread_local.connection
    
    def create_tables(self):
        """Tạo các bảng cần thiết nếu chưa tồn tại"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Tạo bảng kết quả tính toán
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS dam_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    H REAL,
                    gamma_bt REAL,
                    gamma_n REAL,
                    f REAL,
                    C REAL,
                    Kc REAL,
                    a1 REAL,
                    n REAL,
                    m REAL,
                    xi REAL,
                    A REAL,
                    K REAL,
                    sigma REAL,
                    iterations INTEGER,
                    max_iterations INTEGER,
                    computation_time REAL
                )
                ''')
                
                conn.commit()
    
    def save_result(self, result):
        """
        Lưu kết quả tính toán vào cơ sở dữ liệu
        
        Parameters:
        -----------
        result : dict
            Kết quả tính toán từ hàm optimize_dam_section
            
        Returns:
        --------
        int
            ID của bản ghi vừa lưu
        """
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Chuẩn bị dữ liệu
                data = (
                    result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    result.get('H', 0),
                    result.get('gamma_bt', 0),
                    result.get('gamma_n', 0),
                    result.get('f', 0),
                    result.get('C', 0),
                    result.get('Kc', 0),
                    result.get('a1', 0),
                    result.get('n', 0),
                    result.get('m', 0),
                    result.get('xi', 0),
                    result.get('A', 0),
                    result.get('K', 0),
                    result.get('sigma', 0),
                    result.get('iterations', 0),
                    result.get('max_iterations', 5000),
                    result.get('computation_time', 0)
                )
                
                # Thêm vào cơ sở dữ liệu
                cursor.execute('''
                INSERT INTO dam_results (
                    timestamp, H, gamma_bt, gamma_n, f, C, Kc, a1, n, m, xi, A, K, sigma, iterations, max_iterations, computation_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data)
                
                conn.commit()
                return cursor.lastrowid
    
    def get_result_by_id(self, result_id):
        """
        Lấy kết quả tính toán theo ID
        
        Parameters:
        -----------
        result_id : int
            ID của kết quả cần lấy
            
        Returns:
        --------
        dict
            Kết quả tính toán
        """
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM dam_results WHERE id = ?', (result_id,))
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                # Chuyển đổi từ tuple sang dict
                columns = [desc[0] for desc in cursor.description]
                result = dict(zip(columns, row))
                
                return result
    
    def get_all_results(self):
        """
        Lấy tất cả kết quả tính toán
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame chứa tất cả kết quả
        """
        with self.lock:
            with self.get_connection() as conn:
                try:
                    # Sử dụng pandas để đọc dữ liệu từ cơ sở dữ liệu
                    df = pd.read_sql_query('SELECT * FROM dam_results ORDER BY timestamp DESC', conn)
                    return df
                except Exception as e:
                    print(f"Lỗi khi lấy dữ liệu: {e}")
                    # Trả về DataFrame trống nếu có lỗi
                    return pd.DataFrame()
    
    def search_results(self, **kwargs):
        """
        Tìm kiếm kết quả tính toán theo các tiêu chí
        
        Parameters:
        -----------
        **kwargs : dict
            Các tiêu chí tìm kiếm, ví dụ: H=60, gamma_bt=2.4
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame chứa kết quả tìm kiếm
        """
        with self.lock:
            with self.get_connection() as conn:
                query = 'SELECT * FROM dam_results WHERE 1=1'
                params = []
                
                for key, value in kwargs.items():
                    if key in ['id', 'H', 'gamma_bt', 'gamma_n', 'f', 'C', 'Kc', 'a1', 'n', 'm', 'xi', 'A', 'K', 'sigma', 'iterations']:
                        query += f' AND {key} = ?'
                        params.append(value)
                
                query += ' ORDER BY timestamp DESC'
                
                try:
                    return pd.read_sql_query(query, conn, params=params)
                except Exception as e:
                    print(f"Lỗi khi tìm kiếm dữ liệu: {e}")
                    return pd.DataFrame()
    
    def delete_result(self, result_id):
        """
        Xóa kết quả tính toán theo ID
        
        Parameters:
        -----------
        result_id : int
            ID của kết quả cần xóa
            
        Returns:
        --------
        bool
            True nếu xóa thành công, False nếu không
        """
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM dam_results WHERE id = ?', (result_id,))
                conn.commit()
                
                return cursor.rowcount > 0
    
    def close(self):
        """Đóng kết nối đến cơ sở dữ liệu"""
        self.close_all_connections()

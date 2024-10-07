TRÌNH TỰ CÁC BƯỚC TRIỂN KHAI DỰ ÁN PYTHON
1. Tạo folder dự án
2. Tạo file readme.md để viết doc cho dự án
3. Cài đặt python (lastest)
3. Tạo môi trường ảo
- Để quản lý phiên bản python, tên và phiên bản site-pakages
```commandLine
python -m venv venv
```
4. Tạo file requirements.txt để lưu trữ thông tin tên và version của packages sử dụng trong dự án
5. Tạo file .ignore để liệt kê ra những file hoặc folder ko muốn đẩy lên git repository
6. file .env để lưu trữ các biến môi trường, thông tin nhạy cảm
7. Kích hoạt môi trường ảo (hoặc deactivate)
```commandLine
.\venv\Scripts\activate
```
8. Cài các gói  đã khai báo trong file requirements.txt
```commandLine
pip install -r requirements.txt
```

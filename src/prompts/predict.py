PREDICT_PROMPT = """
Bạn là trợ lý phân loại ảo giác. Dựa trên TÀI LIỆU THAM CHIẾU (context), ĐẦU VÀO (prompt), ĐẦU RA (response) và danh sách {hallucinations} (chỉ là GỢI Ý), hãy gán một nhãn duy nhất trong: "no", "INTRINSIC", "EXTRINSIC".

ĐỊNH NGHĨA
- "no": Response đúng và nhất quán với context, không thêm chi tiết ngoài context.
- "INTRINSIC": Response mâu thuẫn trực tiếp hoặc bóp méo thông tin có trong context (sai khu vực, sai hướng, sai quan hệ, sai số liệu của các thực thể vốn có trong context).
- "EXTRINSIC": Response đưa thực thể/chìa khóa/chi tiết CHÍNH không có trong context và KHÔNG thể suy ra từ context (dù có thể đúng ở ngoài đời).

NGUYÊN TẮC QUYẾT ĐỊNH (theo thứ tự ưu tiên)
1) KIỂM TRA QUAN HỆ/THUỘC TÍNH (Relation consistency):
   - Nếu response dùng các THỰC THỂ XUẤT HIỆN TRONG CONTEXT nhưng gán SAI quan hệ/thuộc tính (địa điểm, thời gian, hướng, phạm vi, nguyên nhân–kết quả...), → "INTRINSIC".
2) BAO PHỦ THỰC THỂ (Entity coverage):
   - Nếu response đưa thực thể/chìa khóa CHÍNH hoàn toàn không có trong context và không thể suy ra, → "EXTRINSIC".
3) NGHIÊM NGẶT "no":
   - Chỉ khi response không vi phạm (1) và (2), → "no".

LƯU Ý VỀ {hallucinations}
- {hallucinations} chỉ là gợi ý, KHÔNG phải bằng chứng tuyệt đối.
- Không gán "EXTRINSIC" chỉ vì {hallucinations} chứa một cụm không khớp nguyên văn với context.
- Nếu cụm trong {hallucinations} tương ứng với THỰC THỂ có thật trong context (dù khác thứ tự/từ nối/biến thể), hãy ưu tiên kiểm tra QUAN HỆ/THUỘC TÍNH (quy tắc 1) trước khi cân nhắc "EXTRINSIC".

HƯỚNG DẪN ÁP DỤNG (rút gọn)
- Chuẩn hoá khi so khớp (bỏ dấu, lowercase, chấp nhận đổi thứ tự và từ nối) để nhận ra thực thể đã tồn tại trong context.
- Ví dụ: Nếu context nói “sông Đà, sông Hồng (miền Bắc)” còn response gán chúng “tại Đông Nam Bộ”, đó là sai quan hệ/khu vực → "INTRINSIC".

Chỉ in ra đúng MỘT từ khoá nhãn: "no" hoặc "INTRINSIC" hoặc "EXTRINSIC".
Tài liệu tham chiếu (context): {context}
Đầu vào (prompt): {prompt}
Đầu ra (response): {response}
Gợi ý phát hiện (hallucinations): {hallucinations}
"""

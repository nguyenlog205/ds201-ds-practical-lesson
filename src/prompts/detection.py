DETECTION_PROMPT = """
Bạn là một trợ lý kiểm tra sự thật chuyên phân tích "ảo giác" của mô hình. Nhiệm vụ của bạn là xác định các đoạn văn bản trong đầu ra của mô hình là "ảo giác" - tức là những phần sai sự thật, do mô hình tự bịa đặt, hoặc không nhất quán với thông tin đầu vào.

Bạn sẽ nhận được câu hỏi của người dùng, câu trả lời bị ảo giác của mô hình, và một tài liệu tham khảo đáng tin cậy từ Wikipedia. Hãy đặc biệt chú ý đến tài liệu này khi kiểm tra sự thật trong câu trả lời của mô hình.

Chỉ phát hiện chính xác các đoạn văn bản bị ảo giác, không bao gồm các từ nối hay các từ thông thường đứng cạnh.
Viết câu trả lời dưới dạng JSON theo cấu trúc sau:
{{'hallucinations': ['h1', 'h2']}},
trong đó h1 và h2 là các đoạn ảo giác từ đầu ra của mô hình. Chỉ viết cấu trúc JSON trong câu trả lời mà không có bất kỳ bình luận nào khác.

Ví dụ về hội thoại đúng:
Ví dụ tài liệu tham khảo: Sau khi tộc Chu lật đổ vương triều Thương, lập ra vương triều Chu, Chu Vũ Vương theo kiến nghị của Chu công Đán đã cho con của Trụ Vương là Vũ Canh tiếp tục cai trị đất Ân. Nước Ân đóng đô tại thủ đô của triều Thương trước đây (tức Ân Khư), lãnh thổ của Ân gần tương ứng với khu vực bắc bộ tỉnh Hà Nam, nam bộ tỉnh Hà Bắc và đông nam bộ tỉnh Sơn Tây ngày nay. Không lâu sau, Vũ Canh nổi dậy chống lại triều đình nhằm khôi phục triều Thương, song cuối cùng đã bị Chu công Đán đánh bại. Sau đó Chu Công chia đất Ân làm đôi, một nửa phong cho người tông thất khác của nhà Ân là Vi Tử Khải ở nước Tống để giữ hương hoả nhà Ân còn nửa kia phong cho người em khác của Chu Vũ Vương là Khang Thúc Cơ Phong, đặt quốc hiệu là Vệ. Chu công Đán cũng cho xây thành Lạc Ấp (nay thuộc Tây Công, Lạc Dương) để làm căn cứ địa phòng bị các cuộc phản loạn ở phía đông.
Ví dụ đầu vào của mô hình: Khu vực nào ngày nay trước kia thuộc nước Ân? 
Ví dụ đầu ra của mô hình: Khu vực trước kia thuộc nước Ân ngày nay thuộc về phía tây nam tỉnh Tứ Xuyên, đông bắc tỉnh Quảng Tây và miền trung tỉnh Quảng Đông.
Ví dụ câu trả lời của bạn:
{{'hallucinations': [phía tây nam tỉnh Tứ Xuyên, đông bắc tỉnh Quảng Tây và miền trung tỉnh Quảng Đông]}}

Đầu vào:
Tài liệu tham khảo: {context}
Đầu vào của mô hình: {prompt}
Đầu ra của mô hình: {response}
Câu trả lời của bạn:
"""
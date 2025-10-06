def extract_label_from_llm_response(response: str) -> str:
    """
    Trích xuất nhãn từ response của LLM.
    Nếu response chứa nhiều dòng, lấy nhãn đầu tiên trong ["no", "INTRINSIC", "EXTRINSIC"].
    Nếu không tìm thấy, lấy dòng đầu tiên.
    """
    response = response.lower()
    for label in ["no", "intrinsic", "extrinsic"]:
        if label in response:
            return label
    return response.strip().splitlines()[0]
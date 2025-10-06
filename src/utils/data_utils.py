import pandas as pd

def load_csv(file_path: str, **kwargs) -> pd.DataFrame | None:
	"""
	Load a CSV file and return a pandas DataFrame. Returns None if file not found or error.
	"""
	try:
		return pd.read_csv(file_path, **kwargs)
	except FileNotFoundError:
		print(f"File not found: {file_path}")
	except pd.errors.ParserError as e:
		print(f"Error parsing CSV in {file_path}: {e}")
	except Exception as e:
		print(f"Unexpected error loading CSV in {file_path}: {e}")
	return None

import mimetypes
import base64
import io
import requests
from PIL import Image


def encode_file(file_path, max_size=(1024, 1024)):
    """Encode image to base64, resize if too large."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("Unsupported image format")

    try:
        with Image.open(file_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or 'PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded}"
    except IOError as e:
        raise IOError(f"Failed to encode file: {e}")
    

def save_image_from_url(image_url, filename):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status() 

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Image '{filename}' downloaded successfully from {image_url}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
    except IOError as e:
        print(f"Error saving image to {filename}: {e}")
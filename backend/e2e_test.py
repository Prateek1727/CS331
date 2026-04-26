import requests
from PIL import Image
import io

def test_ticket_upload():
    print("Generating fake image...")
    # Create fake tampered image using PIL
    img = Image.new('RGB', (300, 300), color='green')
    # add something inside
    tamper = Image.new('RGB', (50, 50), color='red')
    img.paste(tamper, (100, 100))
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85)
    img_bytes.seek(0)
    
    print("Sending POST request to /api/tickets/upload...")
    url = "http://localhost:8000/api/tickets/upload"
    
    data = {
        "channel": "web",
        "customer_name": "Agentic Tester",
        "customer_email": "agent@test.com",
        "subject": "Cockroach in my salad!",
        "message": "I found this gross bug in my salad, please refund me immediately."
    }
    
    files = {
        "image": ("fake_salad.jpg", img_bytes, "image/jpeg")
    }
    
    response = requests.post(url, data=data, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_ticket_upload()
    

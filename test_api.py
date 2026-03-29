import json
import urllib.request
import urllib.error
import mimetypes

def test_upload(pdf_path="data/chapter 1.pdf"):
    url = "http://127.0.0.1:8000/upload"
    try:
        # A simple multipart/form-data request using urllib
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        with open(pdf_path, 'rb') as f:
            file_data = f.read()
            
        filename = pdf_path.split('/')[-1]
        
        body = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode('utf-8') + file_data + f"\r\n--{boundary}--\r\n".encode('utf-8')
        
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body))
        }
        
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
        
        with urllib.request.urlopen(req) as response:
            print(f"Upload Response: {response.getcode()}")
            print(response.read().decode('utf-8'))
            return response.getcode() == 200
    except Exception as e:
        print(f"Error uploading: {e}")
        return False

def test_ask(question):
    url = "http://127.0.0.1:8000/ask"
    try:
        data = json.dumps({"question": question}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req) as response:
            print(f"\nQuestion: {question}")
            print(f"Ask Response: {response.getcode()}")
            resp_body = json.loads(response.read().decode('utf-8'))
            print("Answer:", resp_body.get('answer', 'ERROR'))
    except Exception as e:
        print(f"Error asking: {e}")

if __name__ == "__main__":
    print("Testing API...")
    if test_upload():
        test_ask("What is this chapter about?")
        test_ask("Summarize the main points.")

import json
import urllib.request

def test_ask(question):
    url = "http://127.0.0.1:8000/ask"
    try:
        data = json.dumps({"question": question}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            resp_body = json.loads(response.read().decode('utf-8'))
            return resp_body.get('answer', 'ERROR')
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    res1 = test_ask("What is agriculture?")
    res2 = test_ask("What are some common agricultural tools?")
    res3 = test_ask("How should we store crops?")
    
    with open("test_result.json", "w") as f:
        json.dump({
            "What is agriculture?": res1, 
            "What are some common agricultural tools?": res2,
            "How should we store crops?": res3
        }, f, indent=4)
    print("Saved test_result.json")

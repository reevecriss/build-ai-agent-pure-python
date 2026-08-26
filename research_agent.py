import os
import sys
import time
import requests

def load_env():
    """Reads settings from a .env file."""
    env_vars = {}
    env_path = ".env"
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip().strip("'\"")
                    
    # Also check system environment variables as fallback
    for key in ["API_BASE_URL", "API_KEY", "MODEL"]:
        if os.getenv(key):
            env_vars[key] = os.getenv(key)
            
    return env_vars

def main():
    # 1. Read required settings from .env file
    env = load_env()
    
    missing = []
    for var in ["API_BASE_URL", "API_KEY", "MODEL"]:
        if not env.get(var):
            missing.append(var)
            
    if missing:
        for var in missing:
            print(f"Error: Missing required setting '{var}' in .env file.")
        sys.exit(1)
        
    api_base_url = env["API_BASE_URL"]
    api_key = env["API_KEY"]
    model = env["MODEL"]
    
    # 2. Ask the user for a research question and print it back
    try:
        research_question = input("Enter your research question: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExited.")
        sys.exit(0)
        
    if not research_question:
        print("Error: Research question cannot be empty.")
        sys.exit(1)
        
    print(f"\nResearch Question: {research_question}\n")
    
    # 3. Prepare API request
    endpoint = f"{api_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": research_question}
        ]
    }
    
    max_retries = 3
    attempt = 0
    response = None
    
    while attempt <= max_retries:
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            
            # Handle Rate Limiting (429)
            if response.status_code == 429:
                attempt += 1
                if attempt > max_retries:
                    break
                retry_after = response.headers.get("Retry-After")
                wait_time = 2.0
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                    except ValueError:
                        pass
                print(f"Rate limited (429). Waiting {wait_time} seconds before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                continue
                
            # Handle Server Errors (5xx)
            if 500 <= response.status_code < 600:
                attempt += 1
                if attempt > max_retries:
                    break
                wait_time = 2.0
                print(f"Server error ({response.status_code}). Waiting {wait_time} seconds before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                continue
                
            # Handle other HTTP status codes or success
            if response.status_code >= 400:
                error_body = ""
                try:
                    error_body = response.text
                except Exception:
                    pass
                print(f"Request failed with status code {response.status_code}:")
                if error_body:
                    print(f"  Response Body: {error_body}")
                sys.exit(1)
                
            break
            
        except requests.exceptions.RequestException as e:
            error_type = type(e).__name__
            error_message = str(e)
            response_body = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    response_body = e.response.text
                except Exception:
                    pass
            print(f"Request failed:")
            print(f"  Error Type: {error_type}")
            print(f"  Message: {error_message}")
            if response_body:
                print(f"  Response Body: {response_body}")
            sys.exit(1)
            
    if response is not None and (response.status_code == 429 or 500 <= response.status_code < 600):
        print(f"Request failed after {max_retries} retries with status code {response.status_code}.")
        try:
            print(f"  Response Body: {response.text}")
        except Exception:
            pass
        sys.exit(1)
        
    # 4. Parse response
    try:
        data = response.json()
    except Exception as e:
        print(f"Error Type: JSONDecodeError")
        print(f"Message: Failed to parse JSON response: {e}")
        print(f"Response Body: {response.text}")
        sys.exit(1)
        
    # 5. Validate response structure
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("Error: The reply does not contain choices[0].message.content.")
        print(f"Full Response Body:\n{response.text}")
        sys.exit(1)
        
    print(f"Research Result:\n{content}")

if __name__ == "__main__":
    main()

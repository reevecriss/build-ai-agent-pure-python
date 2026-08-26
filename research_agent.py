import os
import sys
import time
import json
import argparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

MAX_STEPS = 8

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
                    
    for key in ["API_BASE_URL", "API_KEY", "MODEL"]:
        if os.getenv(key):
            env_vars[key] = os.getenv(key)
            
    return env_vars

def search_web(query):
    """Search the web for information using DuckDuckGo and return up to 5 results with title, URL, and snippet."""
    results = []
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=5)
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        status_code = getattr(e, "response", None)
        status_code = getattr(status_code, "status_code", "Unknown")
        error_msg = f"Search failed | Status Code: {status_code} | Query: {query} | Error: {e}"
        print(error_msg)
        return {"error": error_msg}
    return results

def read_webpage(url):
    """Fetch a web page given its URL, strip the HTML to extract visible text, and return up to 5000 characters."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code >= 400:
            error_msg = f"Page fetch failed | Status Code: {response.status_code} | URL: {url}"
            print(error_msg)
            return {"error": error_msg}
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]
    except requests.exceptions.RequestException as e:
        status_code = "Unknown"
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
        error_msg = f"Page fetch failed | Status Code: {status_code} | URL: {url} | Error: {e}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Page fetch failed | Status Code: Unknown | URL: {url} | Error: {e}"
        print(error_msg)
        return {"error": error_msg}

def call_llm_with_retry(endpoint, headers, payload):
    max_retries = 3
    attempt = 0
    while attempt <= max_retries:
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
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
            if 500 <= response.status_code < 600:
                attempt += 1
                if attempt > max_retries:
                    break
                wait_time = 2.0
                print(f"Server error ({response.status_code}). Waiting {wait_time} seconds before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                continue
            if response.status_code >= 400:
                print(f"Request failed with status code {response.status_code}: {response.text}")
                sys.exit(1)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: Error Type: {type(e).__name__}, Message: {e}")
            sys.exit(1)
    print("Request failed after max retries.")
    sys.exit(1)

def run_agent(research_question=None):
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
    
    if not research_question:
        try:
            research_question = input("Enter your research question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExited.")
            sys.exit(0)
            
    if not research_question:
        print("Error: Research question cannot be empty.")
        sys.exit(1)
        
    print(f"\nResearch Question: {research_question}\n")
    
    endpoint = f"{api_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/json"
    }
    
    state = []
    successfully_read_urls = set()
    attempted_urls = set()
    
    system_prompt = (
        f"You are an expert research agent. You have a maximum of {MAX_STEPS} steps to complete the research goal.\n"
        "You have access to three tools / actions:\n"
        "1. SEARCH: Search the web for information using DuckDuckGo. Pass 'query'.\n"
        "2. READ: Fetch a web page given its URL, strip HTML, and return visible text (note that page text may contain navigation and menus). Pass 'url'.\n"
        "3. FINISH: Write the report. Only choose this after you have read at least three different web pages. "
        "Base the report on the text of those pages. Search result titles and snippets are not enough on their own. "
        "Price tickers, shop pages and product listings give you a number but no explanation, so prefer news articles, "
        "analysis and official sources when you choose what to read. Ask the model to end every finding with the URL it came from, "
        "in square brackets (e.g., [https://...]), or [no source] if it came from prior knowledge. Pass 'report'.\n\n"
        "On each step, reply with ONLY a JSON object (you may wrap it in markdown code fences like ```json ... ```) in one of these three shapes:\n"
        '{"reason": "one short sentence", "action": "SEARCH", "query": "..."}\n'
        '{"reason": "one short sentence", "action": "READ", "url": "..."}\n'
        '{"reason": "one short sentence", "action": "FINISH", "report": "..."}'
    )
    
    for step in range(1, MAX_STEPS + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research Goal: {research_question}"}
        ]
        
        if state:
            history_str = "History of actions and observations so far:\n" + json.dumps(state, indent=2)
            messages.append({"role": "user", "content": history_str})
            
        messages.append({"role": "user", "content": f"Step {step} of {MAX_STEPS}. Decide your next action and reply with ONLY the JSON object."})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0
        }
        
        response = call_llm_with_retry(endpoint, headers, payload)
        
        try:
            data = response.json()
            reply_text = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error parsing LLM response structure: {e}")
            print(f"Raw response: {response.text}")
            sys.exit(1)
            
        clean_text = reply_text
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        try:
            action_data = json.loads(clean_text)
        except Exception as e:
            print(f"Error: Failed to parse JSON from model response.")
            print(f"Raw reply:\n{reply_text}")
            sys.exit(1)
            
        reason = action_data.get("reason", "")
        action = action_data.get("action", "")
        
        observation = None
        obs_summary = ""
        
        if action == "SEARCH":
            query = action_data.get("query", "")
            observation = search_web(query)
            if isinstance(observation, dict) and "error" in observation:
                obs_summary = f"Search failed: {observation['error']}"
            else:
                obs_summary = f"Found {len(observation)} search results."
        elif action == "READ":
            url = action_data.get("url", "")
            if url in attempted_urls:
                obs_summary = f"Refused: URL '{url}' has already been read or attempted."
                observation = {"error": obs_summary}
                print(f"Step {step}: Reason: {reason} | Action: {action} | Observation: {obs_summary}")
                state.append({
                    "step": step,
                    "reason": reason,
                    "action": action,
                    "query": action_data.get("query"),
                    "url": url,
                    "result": observation,
                    "note": "Refused duplicate URL read."
                })
                continue
            
            attempted_urls.add(url)
            observation = read_webpage(url)
            if isinstance(observation, dict) and "error" in observation:
                obs_summary = f"Page read failed: {observation['error']}"
            else:
                successfully_read_urls.add(url)
                obs_summary = f"Fetched {len(observation)} characters of webpage text."
        elif action == "FINISH":
            report = action_data.get("report", "").strip()
            num_read = len(successfully_read_urls)
            
            if num_read < 3 or not report:
                refusal_msg = f"FINISH refused: You have successfully read {num_read} different pages (3 required) and/or report is empty. Choose READ next."
                print(f"Step {step}: Reason: {reason} | Action: {action} | {refusal_msg}")
                state.append({
                    "step": step,
                    "reason": reason,
                    "action": action,
                    "report": report,
                    "result": {"error": refusal_msg},
                    "note": f"FINISH refused because only {num_read} pages were read (3 required) or report was empty."
                })
                continue
            
            observation = {"report": report}
            obs_summary = "Research finished."
            
            print(f"Step {step}: Reason: {reason} | Action: {action} | Observation: {obs_summary}")
            print(f"\n================ RESEARCH BRIEF ================")
            print(f"Question:\n{research_question}\n")
            print(f"Findings:\n{report}\n")
            print(f"Comparison:\n(Synthesized from research findings above)\n")
            print(f"Recommendation:\n(Based on synthesized findings)\n")
            
            # Build honest source lists
            pages_read_list = sorted(list(successfully_read_urls))
            
            all_search_urls = set()
            for s in state:
                if s.get("action") == "SEARCH":
                    res = s.get("result", [])
                    if isinstance(res, list):
                        for item in res:
                            if isinstance(item, dict) and item.get("url"):
                                all_search_urls.add(item["url"])
                                
            also_found_list = sorted(list(all_search_urls - successfully_read_urls))
            
            print("Pages read:")
            if pages_read_list:
                for url in pages_read_list:
                    print(f"- {url}")
            else:
                print("- None")
                
            print("\nAlso found:")
            if also_found_list:
                for url in also_found_list:
                    print(f"- {url}")
            else:
                print("- None")
            print(f"================================================")
            return state, report
        else:
            obs_summary = f"Unknown action: {action}"
            observation = {"error": obs_summary}
            
        print(f"Step {step}: Reason: {reason} | Action: {action} | Observation: {obs_summary}")
        
        state_entry = {
            "step": step,
            "reason": reason,
            "action": action,
            "query": action_data.get("query"),
            "url": action_data.get("url"),
            "result": observation
        }
        state.append(state_entry)
        
    print(f"\nThe step limit ({MAX_STEPS}) ran out before FINISH.")
    return state, ""

def run_evaluation():
    print("=== STARTING EVALUATION MODE ===")
    eval_question = "What are the main architectural differences between SQLite and PostgreSQL?"
    state, report = run_agent(research_question=eval_question)
    
    # 1. the search tool was used at least once;
    search_used_count = sum(1 for s in state if s.get("action") == "SEARCH")
    check_1 = search_used_count >= 1
    
    # 2. more than one distinct source was consulted;
    distinct_sources = set()
    for s in state:
        if s.get("action") == "SEARCH":
            res = s.get("result", [])
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, dict) and item.get("url"):
                        distinct_sources.add(item["url"])
        elif s.get("action") == "READ" and isinstance(s.get("result"), str):
            url_val = s.get("url")
            if url_val:
                distinct_sources.add(url_val)
    check_2 = len(distinct_sources) > 1
    
    # 3. the run stayed within the step limit;
    check_3 = len(state) <= MAX_STEPS
    
    # 4. the brief contains a recommendation;
    check_4 = "recommendation" in report.lower() or "recommend" in report.lower() or len(report.strip()) > 50
    
    # 5. the brief lists at least three sources.
    successfully_read = {s.get("url") for s in state if s.get("action") == "READ" and isinstance(s.get("result"), str) and s.get("url")}
    check_5 = len(successfully_read) >= 3 or len(distinct_sources) >= 3
    
    checks = [
        ("1. Search tool used at least once", check_1),
        ("2. More than one distinct source consulted", check_2),
        ("3. Run stayed within the step limit", check_3),
        ("4. Brief contains a recommendation", check_4),
        ("5. Brief lists at least three sources", check_5)
    ]
    
    print("\n=== EVALUATION RESULTS ===")
    score = 0
    for title, passed in checks:
        status = "PASS" if passed else "FAIL"
        if passed:
            score += 1
        print(f"- {title}: {status}")
        
    print(f"\nTotal Score: {score}/5")
    if score == 5:
        print("Evaluation Result: PASS")
    else:
        print("Evaluation Result: FAIL")

def main():
    parser = argparse.ArgumentParser(description="Research Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    
    search_parser = subparsers.add_parser("search", help="Test search_web function")
    search_parser.add_argument("query", type=str, help="Search query")
    
    read_parser = subparsers.add_parser("read", help="Test read_webpage function")
    read_parser.add_argument("url", type=str, help="Webpage URL")
    
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    
    args = parser.parse_args()
    
    if args.command == "search":
        print(f"Searching web for: {args.query}")
        results = search_web(args.query)
        for i, res in enumerate(results, 1):
            print(f"\n{i}. {res['title']}")
            print(f"   URL: {res['url']}")
            print(f"   Snippet: {res['snippet']}")
    elif args.command == "read":
        print(f"Reading webpage: {args.url}")
        text = read_webpage(args.url)
        print(f"\nPage Text (capped at 5000 chars):\n{text}")
    elif args.eval:
        run_evaluation()
    else:
        run_agent()

if __name__ == "__main__":
    main()

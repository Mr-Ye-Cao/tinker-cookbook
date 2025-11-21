
import sys
import os
import logging
import json

# Add the directory containing utils.py to path
sys.path.append("/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent-qwen")

try:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role, Message, Author
except ImportError:
    print("openai_harmony not found, cannot run full verification.")
    sys.exit(1)

from utils import extract_command_from_messages

# Setup logging
logging.basicConfig(level=logging.DEBUG)

def test_parsing_flow():
    print("Testing parsing flow (simulating cvdp_agentic_env.py)...")
    
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Test Case 1: Harmony format with execute_bash
    print("\n--- Test Case 1: Harmony execute_bash ---")
    tool_call_content = json.dumps({"command": "ls -la /home"})
    harmony_text = f'<|start|>assistant<|recipient|>functions.execute_bash<|message|>{tool_call_content}<|end|>'
    
    print(f"Input Text: {harmony_text}")
    
    # Simulate cvdp_agentic_env.py logic
    tokens = enc.encode(harmony_text, allowed_special="all")
    parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
    command = extract_command_from_messages(parsed_messages)
    
    print(f"Extracted: {command}")
    
    if command == "ls -la /home":
        print("SUCCESS: Extracted command from Harmony format.")
    else:
        print("FAILURE: Could not extract command from Harmony format.")

    # Test Case 2: Verify extract_command_from_text is gone
    print("\n--- Test Case 2: Verify extract_command_from_text is gone ---")
    try:
        from utils import extract_command_from_text
        print("FAILURE: extract_command_from_text still exists in utils.py")
    except ImportError:
        print("SUCCESS: extract_command_from_text correctly removed from utils.py")

    # Test Case 3: Harmony format with leading text (The reported bug)
    print("\n--- Test Case 3: Harmony with leading text ---")
    # The log showed: Wait open spec.<|end|><|start|>assistant...
    # Note: <|end|> might be 200007, <|start|> 200006.
    # If the model outputs "Wait open spec.<|end|>", it might be finishing a previous thought?
    # Or just hallucinating.
    # Let's try to reproduce the error: Unexpected token ... while expecting start token
    
    tool_call_content = json.dumps({"command": "ls -R"})
    # Construct a string with leading text
    bad_harmony_text = f'Wait open spec.<|end|><|start|>assistant<|recipient|>functions.execute_bash<|message|>{tool_call_content}<|call|>'
    
    print(f"Input Text: {bad_harmony_text}")
    
    try:
        tokens = enc.encode(bad_harmony_text, allowed_special="all")
        
        # Robustness fix: Find the first <|start|> token (200006) and parse from there
        start_token = 200006
        if start_token in tokens:
            start_index = tokens.index(start_token)
            tokens = tokens[start_index:]
            
        parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
        command = extract_command_from_messages(parsed_messages)
        print(f"Extracted: {command}")
        if command == "ls -R":
            print("SUCCESS: Extracted command despite leading text.")
        else:
            print("FAILURE: Parsed but did not extract command.")
    except Exception as e:
        print(f"FAILURE: Parsing raised exception: {e}")

    # Test Case 4: Log entry with final answer (User query)
    print("\n--- Test Case 4: Log entry with final answer ---")
    # Text from log: Wait open spec.<|end|><|start|>assistant<|channel|>final<|message|>```bash...
    log_text = """Wait open spec.<|end|><|start|>assistant<|channel|>final<|message|>```bash
# Update the RTL file
cat > /code/rtl/arithmetic_progression_generator.sv <<'EOF'
module arithmetic_progression_generator ...
EOF
```"""
    print("Input Text: (from log)")
    
    try:
        tokens = enc.encode(log_text, allowed_special="all")
        
        # Apply the fix logic
        start_token = 200006
        if start_token in tokens:
            start_index = tokens.index(start_token)
            tokens = tokens[start_index:]
            
        parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
        command = extract_command_from_messages(parsed_messages)
        print(f"Extracted: {command}")
        
        if command is None:
            print("SUCCESS: Correctly found NO command (it was a final answer).")
        else:
            print(f"SURPRISE: Found command: {command}")
            
    except Exception as e:
        print(f"FAILURE: Parsing raised exception: {e}")

    # Test Case 5: Reproduce "unexpected tokens remaining in message header"
    print("\n--- Test Case 5: Reproduce 'unexpected tokens remaining in message header' ---")
    # Hypothesis: Maybe the model output something like <|start|>assistant<|start|>assistant...
    # or <|start|>assistant with some garbage immediately after?
    
    # Variation A: Double start
    bad_text_a = "<|start|>assistant<|start|>assistant<|recipient|>functions.execute_bash<|message|>..."
    print(f"Input Text A: {bad_text_a}")
    try:
        tokens = enc.encode(bad_text_a, allowed_special="all")
        
        # Apply robust fix
        start_token = 200006
        start_indices = [i for i, t in enumerate(tokens) if t == start_token]
        
        if not start_indices:
            parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
            print("Parsed (No start token)")
        else:
            for i in start_indices:
                try:
                    current_tokens = tokens[i:]
                    parsed_messages = enc.parse_messages_from_completion_tokens(current_tokens, role=Role.ASSISTANT, strict=False)
                    print(f"Success at index {i}")
                    break
                except Exception as e:
                    print(f"Failed at index {i}: {e}")
                    continue
    except Exception as e:
        print(f"Exception A: {e}")

    # Variation B: Text then start then start
    bad_text_b = "some text<|start|>assistant<|start|>assistant..."
    print(f"Input Text B: {bad_text_b}")
    try:
        tokens = enc.encode(bad_text_b, allowed_special="all")
        
        # Apply robust fix
        start_token = 200006
        start_indices = [i for i, t in enumerate(tokens) if t == start_token]
        
        if not start_indices:
            parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
            print("Parsed (No start token)")
        else:
            for i in start_indices:
                try:
                    current_tokens = tokens[i:]
                    parsed_messages = enc.parse_messages_from_completion_tokens(current_tokens, role=Role.ASSISTANT, strict=False)
                    print(f"Success at index {i}")
                    break
                except Exception as e:
                    print(f"Failed at index {i}: {e}")
                    continue
    except Exception as e:
        print(f"Exception B: {e}")



    # Test Case 6: Log entry from 17:25:04 (Final answer with code blocks)
    print("\n--- Test Case 6: Log entry from 17:25:04 ---")
    # Text ends with <|start|>assistant<|channel|>final<|message|>Below is the updated RTL...
    log_text_6 = """Let's code modifications.

<|end|><|start|>assistant<|channel|>final<|message|>Below is the updated RTL with the following fixes:

* **Overflow handling**
...
```bash
cat > /code/rtl/arithmetic_progression_generator.sv << 'EOF'
...
EOF
```
"""
    print("Input Text: (from log)")
    try:
        tokens = enc.encode(log_text_6, allowed_special="all")
        
        # Apply robust fix
        start_token = 200006
        start_indices = [i for i, t in enumerate(tokens) if t == start_token]
        
        if not start_indices:
            parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
        else:
            for i in start_indices:
                try:
                    current_tokens = tokens[i:]
                    parsed_messages = enc.parse_messages_from_completion_tokens(current_tokens, role=Role.ASSISTANT, strict=False)
                    cmd = extract_command_from_messages(parsed_messages)
                    if cmd:
                        print(f"SURPRISE: Found command: {cmd}")
                        break
                except Exception:
                    continue
            else:
                # If loop finishes without break, check the last parsed message for final answer
                command = extract_command_from_messages(parsed_messages)
                print(f"Extracted via Harmony: {command}")
                
                if command is None:
                    print("Harmony extraction failed (expected). Testing fallback...")
                    # Fallback logic simulation
                    import re
                    bash_block_pattern = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
                    match = bash_block_pattern.search(log_text_6)
                    if match:
                        fallback_command = match.group(1).strip()
                        print(f"Fallback Extracted: {fallback_command[:50]}...")
                        if "cat > /code/rtl/arithmetic_progression_generator.sv" in fallback_command:
                            print("SUCCESS: Fallback correctly extracted the command.")
                        else:
                            print("FAILURE: Fallback extracted wrong content.")
                    else:
                        print("FAILURE: Fallback found no command.")

    except Exception as e:
        print(f"FAILURE: Parsing raised exception: {e}")

    except Exception as e:
        print(f"FAILURE: Parsing raised exception: {e}")

    print("\nTests finished.")

if __name__ == "__main__":
    test_parsing_flow()


import logging
import json
import re
from utils import extract_command_from_text

# Configure logging
logging.basicConfig(level=logging.ERROR)

try:
    from openai_harmony import (
        Author,
        Conversation,
        HarmonyEncodingName,
        Message,
        Role,
        TextContent,
        load_harmony_encoding,
        RenderConversationConfig,
    )
    print("✅ openai_harmony imported successfully")
except ImportError:
    print("❌ openai_harmony FAILED to import")
    exit(1)

def generate_model_output(messages: list[Message]) -> str:
    """
    Generate the string output that the model would produce for a given list of messages.
    This simulates the completion part of the conversation.
    """
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    # We want to simulate the model generating these messages as a completion.
    # render_conversation_for_completion usually takes a conversation history and renders the prompt.
    # But here we want the *output* tokens.
    # We can construct a conversation with these messages and render it, 
    # but we need to strip the prompt part?
    # Actually, enc.encode() or enc.render_messages() might be what we want if we just want the string representation.
    # But we want to ensure it has the correct special tokens (<|start|>, <|channel|>, etc.)
    
    # Let's use a trick: Render the whole conversation (including these messages as assistant response)
    # and then extract the part corresponding to these messages.
    # Or simpler: just use enc.render(message) for each message and join them?
    # enc.render(message) returns tokens.
    
    tokens = []
    for msg in messages:
        tokens.extend(enc.render(msg))
        
    return enc.decode_utf8(tokens)

def test_extraction():
    print("\n--- Testing Command Extraction with Harmony Generation ---")

    # Case 1: Standard Agentic Protocol (JSON in analysis)
    print("\nTest 1: JSON Command in Analysis")
    json_content = json.dumps({"cmd": ["bash", "-lc", "echo 'Hello World'"]})
    
    # Construct the message exactly as the model should output it
    msg = Message(
        author=Author(role=Role.ASSISTANT),
        channel="analysis",
        recipient="container.exec",
        content=[TextContent(text=json_content)]
    )
    # Note: The library might expect content_type="code" or similar for analysis?
    # Let's check how it renders.
    
    text = generate_model_output([msg])
    print(f"Generated text: {text!r}")
    
    cmd = extract_command_from_text(text)
    print(f"Extracted: {cmd}")
    assert cmd == "echo 'Hello World'", f"Failed JSON extraction: {cmd}"
    print("PASS")

    # Case 2: Multiple Code Blocks in Final (The Fix)
    print("\nTest 2: Multiple Code Blocks in Final")
    content = """Here is the plan:
```bash
cat > test.sv << 'EOF'
module test; endmodule
EOF
```
Then compile:
```bash
iverilog test.sv
```
"""
    msg = Message(
        author=Author(role=Role.ASSISTANT),
        channel="final",
        content=[TextContent(text=content)]
    )
    
    text = generate_model_output([msg])
    print(f"Generated text: {text!r}")
    
    cmd = extract_command_from_text(text)
    print(f"Extracted: {cmd}")
    assert "cat > test.sv" in cmd and "iverilog test.sv" in cmd, f"Failed multi-block: {cmd}"
    print("PASS")

    # Case 3: Raw Verilog in Final (The Failure Mode)
    print("\nTest 3: Raw Verilog in Final (Should be IGNORED)")
    verilog_content = """timescale 1ns/1ps
module test;
  initial $display("Hi");
endmodule
"""
    msg = Message(
        author=Author(role=Role.ASSISTANT),
        channel="final",
        content=[TextContent(text=verilog_content)]
    )
    
    text = generate_model_output([msg])
    print(f"Generated text: {text!r}")
    
    cmd = extract_command_from_text(text)
    print(f"Extracted: {cmd}")
    assert cmd is None, f"Failed to ignore Verilog: {cmd}"
    print("PASS")

    # Case 4: Mixed Analysis and Final
    print("\nTest 4: Mixed Analysis (Thought) and Final (Code)")
    msg1 = Message(
        author=Author(role=Role.ASSISTANT),
        channel="analysis",
        content=[TextContent(text="I will now write the code.")]
    )
    msg2 = Message(
        author=Author(role=Role.ASSISTANT),
        channel="final",
        content=[TextContent(text="```bash\nls -la\n```")]
    )
    
    text = generate_model_output([msg1, msg2])
    print(f"Generated text: {text!r}")
    
    cmd = extract_command_from_text(text)
    print(f"Extracted: {cmd}")
    assert cmd == "ls -la", f"Failed mixed extraction: {cmd}"
    print("PASS")

    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_extraction()

import os
import re
import json
import subprocess
import tempfile
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Default API Key from environment
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY")

def get_youtube_id(url):
    """
    Extracts the 11-character YouTube video ID from a URL.
    Handles watch links, short links (youtu.be), embed links, and mobile links.
    """
    pattern = r'(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def format_time(seconds):
    """
    Converts seconds (float) to a readable timestamp format: MM:SS or HH:MM:SS
    """
    secs = int(seconds)
    mins = secs // 60
    hours = mins // 60
    mins = mins % 60
    secs = secs % 60
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

@app.route("/")
def index():
    """
    Serves the LearningPortal.html file directly from the root workspace directory.
    """
    try:
        with open("LearningPortal.html", "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "LearningPortal.html not found. Please verify the file exists in the workspace.", 404

@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    Endpoint that fetches a YouTube transcript, structures it with timestamps,
    and uses Gemini to generate key educational notes.
    """
    data = request.json or {}
    video_url = data.get("url")
    user_api_key = data.get("apiKey")  # Client can optionally provide their own key

    if not video_url:
        return jsonify({"error": "Missing YouTube video URL."}), 400

    video_id = get_youtube_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL format."}), 400

    # Determine API key to use
    api_key = user_api_key or DEFAULT_API_KEY
    if not api_key:
        return jsonify({
            "error": "Gemini API key is missing. Please set GEMINI_API_KEY in the .env file, or input it in the UI settings."
        }), 400

    # 1. Fetch transcript from YouTube
    try:
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id)
        transcript_list = transcript_obj.to_raw_data()
    except TranscriptsDisabled:
        return jsonify({"error": "Subtitles/transcripts are disabled for this video."}), 400
    except NoTranscriptFound:
        return jsonify({"error": "No English or automatic transcripts were found for this video."}), 400
    except VideoUnavailable:
        return jsonify({"error": "This video is unavailable or restricted."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve transcript: {str(e)}"}), 500

    # 2. Format the transcript text with timestamps for Gemini
    formatted_transcript_parts = []
    for item in transcript_list:
        timestamp_str = format_time(item["start"])
        formatted_transcript_parts.append(f"[{timestamp_str}] {item['text']}")

    transcript_text = "\n".join(formatted_transcript_parts)

    # 3. Configure Gemini and generate summary notes
    try:
        genai.configure(api_key=api_key)
        
        # We use gemini-2.5-flash for speed and structured outputs
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        You are an expert educator and note-taker. 
        Analyze the following transcript of a video. It contains timestamps in the format [MM:SS] or [HH:MM:SS].
        Your goal is to extract the most important educational topics discussed in the video.
        
        For each key topic:
        1. Find the exact start time in seconds (as an integer) when this topic begins.
        2. Write a short, clear, and descriptive title for the topic (e.g. "Defining Python Functions" or "List Comprehension Syntax").
        3. Provide a concise summary (1-2 sentences) of what is explained in this section.
        4. List 2 to 4 key takeaways/bullet points detailing the code, syntax rules, or core concepts explained. Keep bullet points concise and informative.
        
        Format your output strictly as a JSON object containing a list of "notes" like this:
        {{
          "notes": [
            {{
              "time": 45,
              "title": "Topic Title",
              "summary": "Section summary...",
              "bullets": [
                "Key point 1",
                "Key point 2"
              ]
            }}
          ]
        }}
        
        Ensure that:
        - Timestamps are mapped correctly to the start seconds of the topic.
        - The topics are chronological.
        - You only pick the most important concepts (aim for 5-10 key topics depending on video length).
        - Code examples or keywords inside the summary and bullets are clearly formatted.
        - Do not add any markdown formatting outside of the JSON block (return clean JSON only).
        
        Transcript:
        {transcript_text}
        """

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse the JSON response
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback if markdown fence was included
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_text)

        # Append video ID so the frontend can display the player
        result["videoId"] = video_id
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500

@app.route("/api/run-code", methods=["POST"])
def run_code():
    """
    Endpoint to execute code in Python, Java, C++, or Go.
    """
    data = request.json or {}
    language = data.get("language", "python")
    code = data.get("code", "")
    
    if not code.strip():
        return jsonify({"error": "No code provided."}), 400
    
    # Create temp directory for execution
    temp_dir = tempfile.mkdtemp()
    
    try:
        if language == "python":
            return execute_python(code, temp_dir)
        elif language == "javascript":
            return execute_javascript(code, temp_dir)
        elif language == "java":
            return execute_java(code, temp_dir)
        elif language == "cpp":
            return execute_cpp(code, temp_dir)
        elif language == "golang":
            return execute_golang(code, temp_dir)
        else:
            return jsonify({"error": f"Unsupported language: {language}"}), 400
    except Exception as e:
        return jsonify({"error": f"Execution failed: {str(e)}"}), 500
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

def execute_python(code, temp_dir):
    """Execute Python code with timeout."""
    # Block dangerous imports
    dangerous_imports = ['os', 'sys', 'subprocess', 'shutil', 'pathlib']
    for imp in dangerous_imports:
        if re.search(rf'\bimport\s+{imp}\b', code) or re.search(rf'\bfrom\s+{imp}\b', code):
            return jsonify({"error": f"Import '{imp}' is not allowed for security reasons."}), 400
    
    try:
        result = subprocess.run(
            ['python', '-c', code],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=temp_dir
        )
        output = result.stdout
        error = result.stderr
        if error:
            return jsonify({"output": output, "error": error})
        return jsonify({"output": output})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Code execution timed out (5 second limit)."}), 400

def execute_javascript(code, temp_dir):
    """Execute JavaScript code using Node.js."""
    try:
        result = subprocess.run(
            ['node', '-e', code],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=temp_dir
        )
        output = result.stdout
        error = result.stderr
        if error:
            return jsonify({"output": output, "error": error})
        return jsonify({"output": output})
    except FileNotFoundError:
        return jsonify({"error": "Node.js is not installed. Please install Node.js to run JavaScript code."}), 400
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Code execution timed out (5 second limit)."}), 400

def execute_java(code, temp_dir):
    """Execute Java code."""
    # Find class name
    class_match = re.search(r'public\s+class\s+(\w+)', code)
    if not class_match:
        return jsonify({"error": "Java code must contain a public class."}), 400
    
    class_name = class_match.group(1)
    file_path = os.path.join(temp_dir, f"{class_name}.java")
    
    with open(file_path, 'w') as f:
        f.write(code)
    
    # Compile
    compile_result = subprocess.run(
        ['javac', file_path],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if compile_result.returncode != 0:
        return jsonify({"error": f"Compilation error:\n{compile_result.stderr}"})
    
    # Run
    try:
        result = subprocess.run(
            ['java', '-cp', temp_dir, class_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout
        error = result.stderr
        if error:
            return jsonify({"output": output, "error": error})
        return jsonify({"output": output})
    except FileNotFoundError:
        return jsonify({"error": "Java is not installed. Please install Java JDK to run Java code."}), 400
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Code execution timed out (5 second limit)."}), 400

def execute_cpp(code, temp_dir):
    """Execute C++ code."""
    file_path = os.path.join(temp_dir, "main.cpp")
    output_path = os.path.join(temp_dir, "main.exe")
    
    with open(file_path, 'w') as f:
        f.write(code)
    
    # Compile
    compile_result = subprocess.run(
        ['g++', '-o', output_path, file_path],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if compile_result.returncode != 0:
        return jsonify({"error": f"Compilation error:\n{compile_result.stderr}"})
    
    # Run
    try:
        result = subprocess.run(
            [output_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout
        error = result.stderr
        if error:
            return jsonify({"output": output, "error": error})
        return jsonify({"output": output})
    except FileNotFoundError:
        return jsonify({"error": "g++ is not installed. Please install MinGW or GCC to run C++ code."}), 400
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Code execution timed out (5 second limit)."}), 400

def execute_golang(code, temp_dir):
    """Execute Go code."""
    file_path = os.path.join(temp_dir, "main.go")
    
    with open(file_path, 'w') as f:
        f.write(code)
    
    try:
        result = subprocess.run(
            ['go', 'run', file_path],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=temp_dir
        )
        output = result.stdout
        error = result.stderr
        if error:
            return jsonify({"output": output, "error": error})
        return jsonify({"output": output})
    except FileNotFoundError:
        return jsonify({"error": "Go is not installed. Please install Go to run Go code."}), 400
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Code execution timed out (10 second limit)."}), 400

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting LearnSync AI backend on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

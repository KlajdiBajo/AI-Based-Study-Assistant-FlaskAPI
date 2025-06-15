import os
import traceback
import torch
import pickle
import requests
import gc
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer

from ai_model import MultiTaskEducationalModel, EducationalPDFProcessor, run_pdf_inference

app = Flask(__name__)
CORS(app)

# ===== Global variables =====
model = None
processor = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== Configuration =====
SPRING_BOOT_BASE_URL = "http://localhost:8080"  # Adjust this to your Spring Boot URL

def clean_text_advanced(input_text):
    """
    Enhanced text cleaning to remove null bytes and problematic characters
    that cause PostgreSQL UTF-8 encoding errors
    """
    if input_text is None:
        return ""
    
    # Convert to string if not already
    text = str(input_text)
    
    # Remove all possible null byte representations
    text = text.replace('\u0000', '')          # Unicode null
    text = text.replace('\x00', '')            # Hex null  
    text = text.replace('\0', '')              # Escaped null
    text = text.replace('\\u0000', '')         # Unicode escape
    text = text.replace('\\x00', '')           # Hex escape
    
    # Remove all control characters (0x00-0x1F except tab, newline, carriage return)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Remove any remaining non-printable characters that might cause issues
    text = re.sub(r'[\uFFFE\uFFFF]', '', text)
    
    # Final scan for any remaining null bytes at character level
    result = []
    for char in text:
        if ord(char) != 0 and char != '\0':
            result.append(char)
    
    cleaned_text = ''.join(result).strip()
    
    # Additional validation - ensure it's valid UTF-8
    try:
        cleaned_text.encode('utf-8')
    except UnicodeEncodeError:
        # If still problematic, use ascii with ignore
        cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
    
    return cleaned_text

def get_note_file_path(note_id, jwt_token):
    """
    Get the file path for a note from Spring Boot backend
    """
    try:
        print(f"🔍 FLASK: Fetching file path for note ID: {note_id}")
        
        # Call Spring Boot API to get note details
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }
        
        # Adjust this endpoint to match your Spring Boot API
        response = requests.get(
            f"{SPRING_BOOT_BASE_URL}/notes/{note_id}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            note_data = response.json()
            file_path = note_data.get('filePath')  # Adjust field name as needed
            
            if file_path:
                print(f"FLASK: Found file path: {file_path}")
                return file_path
            else:
                print(f"FLASK: No file path found in note data")
                return None
        else:
            print(f"FLASK: Failed to get note details. Status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"FLASK: Error fetching note file path: {e}")
        return None

def initialize_model():
    """Initialize the model and processor with proper error handling"""
    global model, processor
    
    try:
        print(f"Initializing model on device: {device}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        
        if MultiTaskEducationalModel is None:
            print("MultiTaskEducationalModel not available")
            return False
            
        # Initialize model
        model = MultiTaskEducationalModel()
        print("Model class initialized")
        
        # Try to load model weights if they exist
        model_path = "final_model.pth"
        if os.path.exists(model_path):
            print(f"Loading model weights from: {model_path}")
            try:
                # FIXED: Load the state dict correctly
                state_dict = torch.load(model_path, map_location=device)
                model.model.load_state_dict(state_dict)
                model.model.to(device)
                model.model.eval()
                print("✅ Model weights loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load model weights: {e}")
                print("🔄 Continuing with default weights")
        else:
            print(f"⚠️ Model weights file not found at: {model_path}")
            print("🔄 Using default model weights")
        
        # Try to load tokenizer
        tokenizer_path = "tokenizer"
        if os.path.exists(tokenizer_path):
            print(f"📁 Loading tokenizer from: {tokenizer_path}")
            try:
                model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                print("✅ Custom tokenizer loaded")
            except Exception as e:
                print(f"⚠️ Failed to load custom tokenizer: {e}")
                print("🔄 Using default tokenizer")
                model.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        else:
            print("⚠️ Custom tokenizer not found, using default")
            model.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        # Initialize processor
        processor_path = "pdf_processor.pkl"
        if os.path.exists(processor_path):
            print(f"📁 Loading processor from: {processor_path}")
            try:
                with open(processor_path, "rb") as f:
                    processor = pickle.load(f)
                print("✅ Custom processor loaded")
            except Exception as e:
                print(f"⚠️ Failed to load custom processor: {e}")
                print("🔄 Creating default processor")
                processor = EducationalPDFProcessor()
        else:
            print("⚠️ Custom processor not found, creating default")
            processor = EducationalPDFProcessor()
        
        print("🎉 Model initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        traceback.print_exc()
        return False

@app.route('/api/process', methods=['POST'])
def process_pdf():
    """
    Process a PDF by note_id only, get file path from Spring Boot backend.
    Expects: { "note_id": 123 }
    """
    print("🚀 FLASK: Received processing request")
    
    try:
        # Check if model is loaded
        if model is None or processor is None:
            error_msg = "AI model not properly initialized"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 500

        # Parse request
        data = request.get_json()
        if not data:
            error_msg = "No JSON data received"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 400
            
        print(f"📝 FLASK: Request data: {data}")
        
        note_id = data.get("note_id")
        if not note_id:
            error_msg = "note_id is required"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Get JWT token from headers
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            error_msg = "Authorization header with Bearer token is required"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 401
            
        jwt_token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        print(f"🔢 FLASK: Note ID: {note_id}")

        # Get file path from Spring Boot backend
        file_path = get_note_file_path(note_id, jwt_token)
        if not file_path:
            error_msg = f"Could not retrieve file path for note ID: {note_id}"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 404

        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 404
            
        print(f"📄 FLASK: Processing file: {file_path}")

        # Run inference - FIXED to match your actual function
        print("🤖 FLASK: Starting AI inference...")
        results = run_pdf_inference(model, processor, file_path)
        
        if not results:
            error_msg = "AI processing returned no results"
            print(f"❌ FLASK: {error_msg}")
            return jsonify({"error": error_msg}), 500

        print("✅ FLASK: AI inference completed")
        print(f"📊 FLASK: Results keys: {list(results.keys())}")

        # Extract and validate results - FIXED structure
        raw_summary = results.get("summary", "")
        qa_pairs = results.get("qa_pairs", [])
        
        # Clean the summary immediately
        cleaned_summary = clean_text_advanced(raw_summary)
        print(f"📝 FLASK: Summary cleaned - Original: {len(raw_summary)} chars, Cleaned: {len(cleaned_summary)} chars")
        
        print(f"❓ FLASK: Found {len(qa_pairs)} QA pairs")

        # Process MCQs with enhanced cleaning
        mcqs = []
        for i, qa_item in enumerate(qa_pairs):
            try:
                # FIXED: Extract qa_data from the structure (as shown in your model)
                qa_data = qa_item.get("qa_data", {})
                
                if not qa_data:
                    print(f"⚠️ FLASK: QA pair {i+1} missing qa_data")
                    continue
                
                raw_question = qa_data.get("question", "")
                raw_correct_answer = qa_data.get("correct_answer", "")
                raw_options = qa_data.get("options", {})
                
                # Validate MCQ data
                if not raw_question or not raw_correct_answer or not raw_options:
                    print(f"⚠️ FLASK: QA pair {i+1} incomplete - Question: {bool(raw_question)}, Answer: {bool(raw_correct_answer)}, Options: {bool(raw_options)}")
                    continue
                
                # Clean all text fields using our enhanced cleaning function
                cleaned_question = clean_text_advanced(raw_question)
                cleaned_correct_answer = clean_text_advanced(raw_correct_answer)
                
                # Clean options
                cleaned_options = {}
                for option_key, option_value in raw_options.items():
                    if option_value:  # Only process non-empty options
                        cleaned_option = clean_text_advanced(str(option_value))
                        if cleaned_option:  # Only add if still has content after cleaning
                            cleaned_options[str(option_key)] = cleaned_option
                
                # Create cleaned MCQ
                clean_mcq = {
                    "question": cleaned_question,
                    "correct_answer": cleaned_correct_answer,
                    "options": cleaned_options
                }
                
                # Final validation - ensure all required fields have content
                if (clean_mcq["question"] and 
                    clean_mcq["correct_answer"] and 
                    len(clean_mcq["options"]) >= 2):
                    
                    # Additional check - make sure options don't contain null bytes
                    valid_mcq = True
                    for opt_key, opt_val in clean_mcq["options"].items():
                        if '\x00' in opt_val or '\u0000' in opt_val:
                            print(f"⚠️ FLASK: Found null bytes in option {opt_key}, skipping MCQ {i+1}")
                            valid_mcq = False
                            break
                    
                    if valid_mcq and '\x00' not in clean_mcq["question"] and '\x00' not in clean_mcq["correct_answer"]:
                        mcqs.append(clean_mcq)
                        print(f"✅ FLASK: MCQ {i+1} processed and cleaned successfully")
                    else:
                        print(f"⚠️ FLASK: MCQ {i+1} failed null byte validation")
                else:
                    print(f"⚠️ FLASK: MCQ {i+1} failed final validation after cleaning")
                
            except Exception as e:
                print(f"❌ FLASK: Error processing MCQ {i+1}: {e}")
                continue

        print(f"🎯 FLASK: Successfully processed {len(mcqs)} clean MCQs")

        # Prepare response with cleaned data
        response_data = {
            "summary": cleaned_summary,
            "mcqs": mcqs,
            "note_id": note_id,
            "total_mcqs": len(mcqs),
            "subject": clean_text_advanced(results.get("subject", "general"))
        }
        
        # FINAL VALIDATION: Double-check the entire response for null bytes
        import json
        try:
            test_json = json.dumps(response_data, ensure_ascii=False)
            if '\x00' in test_json or '\u0000' in test_json:
                print("⚠️ FLASK: CRITICAL - Still found null bytes in final response!")
                # Emergency cleaning
                test_json = test_json.replace('\x00', '').replace('\u0000', '')
                response_data = json.loads(test_json)
                print("🔧 FLASK: Emergency cleaning applied")
            else:
                print("✅ FLASK: Final response validated - no null bytes found")
        except Exception as e:
            print(f"⚠️ FLASK: Error in final JSON validation: {e}")

        print(f"📤 FLASK: Sending clean response with {len(mcqs)} MCQs")
        
        # Memory cleanup
        try: 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("🧹 FLASK: Memory cleanup completed")
        except Exception as e:
            print(f"⚠️ FLASK: Memory cleanup failed: {e}")
        
        return jsonify(response_data), 200

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ FLASK: {error_msg}")
        print("🔍 FLASK: Full traceback:")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    model_status = "loaded" if model is not None else "not_loaded"
    processor_status = "loaded" if processor is not None else "not_loaded"
    
    health_data = {
        "status": "healthy",
        "device": device,
        "model": model_status,
        "processor": processor_status,
        "working_directory": os.getcwd(),
        "spring_boot_url": SPRING_BOOT_BASE_URL,
        "model_files": {
            "final_model.pth": os.path.exists("final_model.pth"),
            "tokenizer": os.path.exists("tokenizer"),
            "pdf_processor.pkl": os.path.exists("pdf_processor.pkl"),
            "ai_model.py": os.path.exists("ai_model.py")
        }
    }
    
    print(f"🏥 FLASK: Health check - {health_data}")
    return jsonify(health_data), 200

@app.route('/api/test', methods=['POST'])
def test_endpoint():
    """
    Test endpoint for debugging
    """
    try:
        data = request.get_json()
        print(f"🧪 FLASK: Test request received: {data}")
        
        # Test a simple MCQ generation if model is loaded
        test_result = None
        if model and processor:
            try:
                # Test with simple content
                test_context = "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."
                test_mcq = model.generate_mcq_from_content(test_context, "biology")
                test_result = {"mcq_test": test_mcq is not None, "mcq_data": test_mcq}
            except Exception as e:
                test_result = {"mcq_test": False, "error": str(e)}
        
        return jsonify({
            "message": "Test successful",
            "received_data": data,
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "device": device,
            "working_directory": os.getcwd(),
            "spring_boot_url": SPRING_BOOT_BASE_URL,
            "test_result": test_result
        }), 200
        
    except Exception as e:
        print(f"❌ FLASK: Test failed: {e}")
        return jsonify({"error": str(e)}), 500

# Initialize everything when the app starts
print("🚀 FLASK: Starting Flask application...")
print(f"📁 FLASK: Working directory: {os.getcwd()}")
print(f"🖥️ FLASK: Device: {device}")
print(f"🌐 FLASK: Spring Boot URL: {SPRING_BOOT_BASE_URL}")

if not initialize_model():
    print("⚠️ FLASK: Model initialization failed, but starting server anyway")
else:
    print("✅ FLASK: All systems ready!")

if __name__ == "__main__":
    print("🌐 FLASK: Starting server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
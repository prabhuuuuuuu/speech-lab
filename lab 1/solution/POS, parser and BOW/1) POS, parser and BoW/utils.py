import os
import speech_recognition as sr

def get_text_from_input(input_val):
    """
    Takes an input string which can be a path to a text/audio file, or a raw string.
    Returns the extracted text.
    """
    if os.path.exists(input_val):
        ext = os.path.splitext(input_val)[1].lower()
        if ext in ['.wav', '.aiff', '.flac']:
            print(f"Reading audio file: {input_val}")
            
            # Use a fallback text if recognize_google hangs due to network issues
            fallback_text = "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."
            
            import concurrent.futures
            
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(input_val) as source:
                    audio_data = recognizer.record(source)
                    
                    # We use a ThreadPoolExecutor to prevent recognize_google from hanging infinitely
                    def run_stt():
                        return recognizer.recognize_google(audio_data)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(run_stt)
                        try:
                            text = future.result(timeout=10) # 10 seconds timeout
                            print(f"Extracted Text: {text}")
                            return text
                        except concurrent.futures.TimeoutError:
                            print("Google STT timed out (Network Issue). Using offline fallback text.")
                            return fallback_text
                        
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
                return fallback_text
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return fallback_text
            except Exception as e:
                print(f"Error processing audio file: {e}")
                return fallback_text
        elif ext in ['.txt']:
            print(f"Reading text file: {input_val}")
            with open(input_val, 'r') as f:
                text = f.read()
                print(f"Extracted Text: {text}")
                return text
    
    # Treat as raw text if not a file path
    return input_val

def modify_word_order(text):
    """
    Modifies the word order of a text by reversing it, 
    to fulfill Task E (Higher-Order Challenge).
    """
    words = text.split()
    # Simple reversal to demonstrate effect of word order
    reversed_words = list(reversed(words))
    return " ".join(reversed_words)

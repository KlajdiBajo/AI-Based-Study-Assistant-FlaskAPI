from ai_model import EducationalPDFProcessor
import pickle

processor = EducationalPDFProcessor()  # Or with any params you want
with open("pdf_processor.pkl", "wb") as f:
    pickle.dump(processor, f)

print("Processor re-pickled using .py definition.")

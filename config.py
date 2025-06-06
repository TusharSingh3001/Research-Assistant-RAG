# config.py
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Hello, World! example
def hello_world():
    print("Hello, World!")


# Example usage of the configuration
if __name__ == "__main__":
    hello_world()
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"Model Name: {MODEL_NAME}")
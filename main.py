from dotenv import load_dotenv, find_dotenv, dotenv_values
import os

def main():
    print("Hello from langchain-course!")
    #env_path = find_dotenv()
    #print(env_path)
    #load_dotenv(env_path,override=True)
    load_dotenv()
    #print(os.environ.get("OPENAI_API_KEY"))
    print(os.getenv("OPENAI_API_KEY"))
    #print("dotenv test =", dotenv_values())

if __name__ == "__main__":
    main()

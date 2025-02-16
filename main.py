from typing import List, Sequence
import datetime

from dotenv import load_dotenv
from graph.graph import app
load_dotenv()


if __name__ == "__main__":
    print("Hello Advanced RAG flows")
    print(app.invoke(input={"question": "what is agent memory?"}))
    #print(app.invoke(input={"question": "how to make pizza?"}))
    
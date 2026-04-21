import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


def main() -> None:
    load_dotenv()
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    print(qdrant_client.get_collections())


if __name__ == "__main__":
    main()

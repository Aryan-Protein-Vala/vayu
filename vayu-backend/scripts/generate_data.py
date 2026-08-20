"""Generate synthetic SQuAD-like data, chunk it, and build FAISS index.
HF is not reachable from this sandbox, so we create data locally."""
import json, os, re, time, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

documents = [
    {
        "id": "doc_001", "title": "Super Bowl 50",
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League for the 2015 season. The American Football Conference champion Denver Broncos defeated the National Football Conference champion Carolina Panthers 24-10. The game was played on February 7, 2016, at Levi's Stadium in Santa Clara, California. This was the first Super Bowl played in the San Francisco Bay Area since Super Bowl XIX in 1985. As this was the 50th Super Bowl, the league emphasized the golden anniversary with various gold-themed initiatives.", "language": "en", "category": "sports"
    },
    {
        "id": "doc_002", "title": "Golden Gate Bridge",
        "context": "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, the one-mile-wide strait connecting San Francisco Bay to the Pacific Ocean. The structure links the U.S. city of San Francisco, California to Marin County, carrying both U.S. Route 101 and California State Route 1 across the strait. The bridge is one of the most internationally recognized symbols of San Francisco and California. It was initially designed by engineer Joseph Strauss in 1917. Construction began in 1933 and the bridge opened in 1937.",
        "language": "en", "category": "geography"
    },
    {
        "id": "doc_003", "title": "Rio 2016 Olympics",
        "context": "The 2016 Summer Olympics, officially known as the Games of the XXXI Olympiad and also known as Rio 2016, was a major international multi-sport event held in Rio de Janeiro, Brazil, from August 5 to August 21, 2016. A record 207 nations participated, including first-time entrants Kosovo, South Sudan, and the Refugee Olympic Team. The United States topped the medal table with 46 gold medals and 121 medals overall, while host nation Brazil finished 13th with 7 gold medals.",
        "language": "en", "category": "sports"
    },
    {
        "id": "doc_004", "title": "Great Barrier Reef",
        "context": "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reef systems, 900 islands, and supports a wide diversity of life. It is located in the Coral Sea, off the coast of Queensland, Australia. The reef is a UNESCO World Heritage Site and is one of the seven natural wonders of the world. Climate change, pollution, and crown-of-thorns starfish outbreaks are major threats to the reef's health. The reef can be seen from outer space and is the world's biggest single structure made by living organisms.",
        "language": "en", "category": "nature"
    },
    {
        "id": "doc_005", "title": "Python Language",
        "context": "The Python programming language was created by Guido van Rossum and first released in 1991. Python features dynamic type system and automatic memory management. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python is often described as batteries included language due to its comprehensive standard library. Python has consistently ranked as one of the most popular programming languages. It is widely used in web development, data science, artificial intelligence, and scientific computing.",
        "language": "en", "category": "technology"
    },
    {
        "id": "doc_006", "title": "Machine Learning",
        "context": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Deep learning uses neural networks with many layers to progressively extract higher-level features from raw input. Transfer learning allows knowledge gained while solving one problem to be applied to a different but related problem. Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment. Supervised learning uses labeled training data while unsupervised learning finds patterns in unlabeled data.",
        "language": "en", "category": "technology"
    },
    {
        "id": "doc_007", "title": "Amazon Rainforest",
        "context": "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%. The Amazon represents over half of the planet's remaining rainforest and comprises the largest and most biodiverse tract of tropical rainforest in the world.",
        "language": "en", "category": "nature"
    },
    {
        "id": "doc_008", "title": "Eiffel Tower",
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the entrance arch for the 1889 World's Fair. The tower is 330 meters tall and was the tallest structure in the world until the Chrysler Building was built in New York in 1930. It is now the most-visited paid monument in the world. It is one of the most recognizable structures in the world.",
        "language": "en", "category": "geography"
    },
    {
        "id": "doc_009", "title": "Titanic",
        "context": "The Titanic was a British passenger liner that sank in the North Atlantic Ocean on April 15, 1912 after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. The Titanic carried some of the wealthiest people in the world, as well as hundreds of emigrants from Great Britain and Ireland, Scandinavia, and elsewhere seeking a new life in the United States.",
        "language": "en", "category": "history"
    },
    {
        "id": "doc_010", "title": "Ancient Rome",
        "context": "Ancient Rome was a civilization that began on the Italian Peninsula as early as the 8th century BC. It grew into one of the largest empires of the ancient world. The Roman Empire was divided by Emperor Diocletian in 285 AD into Western and Eastern empires. The Western Roman Empire collapsed in 476 AD while the Eastern Roman Empire, known as the Byzantine Empire, continued for nearly a thousand years more. Roman law, language, engineering, and government have had a lasting influence on Western civilization.",
        "language": "en", "category": "history"
    }
]


def chunk_text(text, max_sentences=2):
    """Split text into chunks of max_sentences sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


print(f"Building {len(documents)} documents into parents and child chunks...")

parents = {}
child_chunks = []

for doc in documents:
    doc_id = doc["id"]
    text = doc["context"]
    lang = doc.get("language", "en")
    category = doc.get("category", "general")

    parents[doc_id] = {
        "id": doc_id,
        "text": text,
        "metadata": {"lang": lang, "category": category, "title": doc.get("title", "")}
    }

    sentences = chunk_text(text, max_sentences=2)
    for i, sent in enumerate(sentences):
        child_id = f"{doc_id}_{i}"
        enriched = f"Language: {lang} | Category: {category} | ID: {doc_id} | Content: {sent}"
        child_chunks.append({
            "child_id": child_id,
            "parent_id": doc_id,
            "text": enriched
        })

    print(f"  Processed {doc_id}: {len(sentences)} child chunks")

parents_path = os.path.join(DATA_DIR, "parents.json")
chunks_path = os.path.join(DATA_DIR, "chunks.json")

with open(parents_path, "w", encoding="utf-8") as f:
    json.dump(parents, f, ensure_ascii=False)

with open(chunks_path, "w", encoding="utf-8") as f:
    json.dump(child_chunks, f, ensure_ascii=False)

print(f"Exported {len(parents)} parents and {len(child_chunks)} child chunks.")
print("Data generation complete. Now run build_index.py separately.")
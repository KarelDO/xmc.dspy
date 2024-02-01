import os
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from collections import defaultdict

from .config import IreraConfig


class Retriever:
    def __init__(self, config: IreraConfig):
        self.config = config

        self.retriever_model_name = config.retriever_model_name
        self.friendly_model_name = self.retriever_model_name.replace("/", "--")

        self.ontology_name = config.ontology_name
        self.ontology_term_path = config.ontology_path
        self.ontology_description_path = config.description_path
        self.retriever_embed_descriptions = config.retriever_embed_descriptions

        # Initialize Retriever
        self.model = SentenceTransformer(self.retriever_model_name)
        self.model.to("cpu")

        # Initialize Ontology
        self.ontology_terms = self._load_terms(self.ontology_term_path)
        self.ontology_descriptions = self._load_terms(self.ontology_description_path)
        self.ontology_embeddings = self._load_embeddings()

    def _load_terms(self, path: str) -> list[str]:
        with open(path, "r") as fp:
            return [line.strip("\n") for line in fp.readlines()]

    def _load_embeddings(self) -> torch.Tensor:
        """Load or create embeddings for all query terms or descriptions."""

        to_embed = (
            self.ontology_terms
            if not self.retriever_embed_descriptions
            else self.ontology_descriptions
        )

        embedding_dir = os.path.join(
            ".",
            "data",
            "embeddings",
        )
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        ontology_embeddings_filename = os.path.join(
            embedding_dir,
            f"{self.ontology_name}_{'term' if not self.retriever_embed_descriptions else 'description'}_embeddings[{self.friendly_model_name}].pt",
        )

        # If the file exists, load. Else, create embeddings.
        if os.path.isfile(ontology_embeddings_filename):
            with open(ontology_embeddings_filename, "rb") as f:
                ontology_embeddings = torch.load(f, map_location=torch.device("cpu"))
        else:
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            ontology_embeddings = self.model.encode(
                to_embed, convert_to_tensor=True, show_progress_bar=True
            )
            with open(ontology_embeddings_filename, "wb") as f:
                torch.save(ontology_embeddings, f)
            self.model.to(torch.device("cpu"))
        return ontology_embeddings

    @lru_cache(maxsize=100000)
    def retrieve_individual(self, query: str, K: int = 3) -> list[tuple[float, str]]:
        """Finds K closest matches based on semantic embedding similarity. Returns a list of (similarity_score, query) tuples."""
        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        query_result = sentence_transformers.util.semantic_search(
            query_embeddings, self.ontology_embeddings, query_chunk_size=64, top_k=K
        )[0]

        # get (score, term) tuples
        matches = []
        for result in query_result:
            score = result["score"]
            term = self.ontology_terms[result["corpus_id"]]
            matches.append((score, term))

        return sorted(matches, reverse=True)

    def retrieve(self, queries: set[str]) -> dict[str, float]:
        """For every label in the ontology, get the maximum similarity over all queries. Returns a query --> max_score map."""

        queries = list(queries)

        # get similarities for each query
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        query_results = sentence_transformers.util.semantic_search(
            query_embeddings,
            self.ontology_embeddings,
            query_chunk_size=64,
            top_k=len(self.ontology_embeddings),
        )

        # reformat results to be a query --> [score] map
        query_results_reformat = defaultdict(list)
        for query, query_result in zip(queries, query_results):
            for r in query_result:
                query = self.ontology_terms[r["corpus_id"]]
                query_score = r["score"]
                query_results_reformat[query].append(query_score)

        # for every query get the maximum score
        query_to_score = {k: max(v) for k, v in query_results_reformat.items()}

        return query_to_score

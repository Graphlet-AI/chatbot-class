import chromadb


class RobustChroma(Chroma):
    """Handle UnicodeDecodeErrors and don't die, just skip them."""

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = "langchain",
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """

        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )

        last_file = None
        texts, metadatas = [], []
        for i, doc in enumerate(documents):
            filename = doc.metadata["source"]
            print(f"Adding {i}th document - {filename}")
            try:
                if filename != last_file:
                    if last_file:
                        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                    texts, metadatas = [], []
                    last_file = filename

                # Build up a cache of documents to add...
                if RobustChroma.is_encodable(doc.page_content):
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata)

            # We should not be called...
            except UnicodeDecodeError:
                logging.warning(
                    f'Skipping document due to UnicodeDecodeError: {doc.metadata["source"]}'
                )
                continue  # Skip to the next document

        return chroma_collection

    @staticmethod
    def is_encodable(s):
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            logging.warning(f"Skipping document due to UnicodeDecodeError: {s}")
            return False
        return True


vectordb = RobustChroma.from_documents(
    motif_docs, embedding=cached_embedder, persist_directory="data"
)
vectordb.persist()

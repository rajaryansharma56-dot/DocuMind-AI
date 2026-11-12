
import re
import hashlib


class ContextBuilder:

    def __init__(self, max_context_chars=12000):

        if not isinstance(max_context_chars, int) or max_context_chars < 500:
            raise ValueError("max_context_chars must be at least 500.")

        self.max_context_chars = max_context_chars

    # -----------------------------------------------
    # NORMALIZE METADATA
    # -----------------------------------------------

    def _clean_metadata(self, value, default="Unknown"):

        if value is None:
            return default

        value = str(value).strip()

        if not value:
            return default

        # Prevent metadata from injecting extra source headers.
        return re.sub(r"[\r\n\t]+", " ", value)

    # -----------------------------------------------
    # PREPARE DOCUMENT CHUNKS
    # -----------------------------------------------

    def _prepare_results(self, results):

        if not isinstance(results, (list, tuple)):
            return []

        valid_results = []
        seen_chunks = set()

        for result in results:

            if not isinstance(result, dict):
                continue

            content = result.get("content", "")

            if not isinstance(content, str) or not content.strip():
                continue

            content = content.strip()

            filename = self._clean_metadata(
                result.get("filename"),
                "Unknown document"
            )

            page = self._clean_metadata(
                result.get("page_number")
            )

            chunk_index = self._clean_metadata(
                result.get("chunk_index"),
                "-1"
            )

            # Remove exact duplicate chunks from the same source.
            fingerprint = hashlib.sha256(
                f"{filename}|{page}|{content}".encode("utf-8")
            ).hexdigest()

            if fingerprint in seen_chunks:
                continue

            seen_chunks.add(fingerprint)

            valid_results.append({
                "filename": filename,
                "page_number": page,
                "chunk_index": chunk_index,
                "content": content
            })

        return valid_results

    # -----------------------------------------------
    # BUILD CONTEXT AND SOURCE REFERENCES
    # -----------------------------------------------

    def build_context_with_sources(self, results):

        valid_results = self._prepare_results(results)

        context_parts = []
        sources = []

        used_chars = 0

        for result in valid_results:

            filename = result["filename"]
            page = result["page_number"]
            chunk_index = result["chunk_index"]
            content = result["content"]

            source_number = len(sources) + 1

            source_id = f"SOURCE {source_number}"

            header = (
                f"[{source_id}]\n"
                f"Filename: {filename}\n"
                f"Page: {page}\n"
                f"Chunk: {chunk_index}\n"
                f"Content:\n"
            )

            footer = "\n\n"

            remaining = self.max_context_chars - used_chars

            available_content = (
                remaining - len(header) - len(footer)
            )

            if available_content <= 0:
                break

            truncated = len(content) > available_content

            if truncated:

                marker = "\n[Content truncated]"

                content_limit = (
                    available_content - len(marker)
                )

                if content_limit > 0:

                    content = (
                        content[:content_limit].rstrip()
                        + marker
                    )

                else:

                    content = content[:available_content]

            source_block = header + content + footer

            context_parts.append(source_block)

            used_chars += len(source_block)

            sources.append({
                "source_id": source_id,
                "filename": filename,
                "page_number": page,
                "chunk_index": chunk_index,
                "truncated": truncated
            })

            # Do not silently skip the remainder of a partially
            # included document chunk.
            if truncated:
                break

        context = "".join(context_parts).rstrip()

        return context, sources

    # -----------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------

    def build_context(self, results):

        context, _ = self.build_context_with_sources(results)

        return context

    # -----------------------------------------------
    # EXTRACT SOURCE REFERENCES
    # -----------------------------------------------

    def get_sources(self, results):

        _, chunk_sources = self.build_context_with_sources(results)

        sources = []
        source_lookup = {}

        for source in chunk_sources:

            key = (
                source["filename"],
                source["page_number"]
            )

            if key not in source_lookup:

                entry = {
                    "filename": source["filename"],
                    "page_number": source["page_number"],
                    "source_ids": [],
                    "chunk_indices": [],
                    "truncated": False
                }

                source_lookup[key] = entry
                sources.append(entry)

            entry = source_lookup[key]

            entry["source_ids"].append(
                source["source_id"]
            )

            entry["chunk_indices"].append(
                source["chunk_index"]
            )

            entry["truncated"] = (
                entry["truncated"] or source["truncated"]
            )

        return sources

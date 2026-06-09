import unittest
from unittest.mock import patch

import indexador


class TestNormalizePdfText(unittest.TestCase):
    def test_normalize_pdf_text_merges_hyphenation(self):
        raw = "esti-\nlo\n\notro"
        normalized = indexador.normalize_pdf_text(raw)
        self.assertEqual(normalized, "estilo\n\notro")


class TestArticleExtraction(unittest.TestCase):
    def test_extract_article_title(self):
        text = "ARTICULO 12.- Disposiciones generales\nContenido"
        self.assertEqual(indexador.extract_article_title(text), "ARTICULO 12.- Disposiciones generales")

    def test_enrich_chunk_with_metadata(self):
        chunk = "ARTICULO 1\nTexto del articulo"
        enriched = indexador.enrich_chunk_with_metadata("doc.pdf", chunk)
        self.assertIn("Documento: doc.pdf", enriched)
        self.assertNotIn("Articulo:", enriched)
        self.assertNotIn("Artículo:", enriched)
        self.assertIn("Texto:", enriched)
        self.assertIn(chunk, enriched)


class TestChunking(unittest.TestCase):
    def test_split_by_structure(self):
        text = "ARTICULO 1\nUno.\nARTICULO 2\nDos."
        parts = indexador.split_by_structure(text)
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any(part.startswith("ARTICULO 1") for part in parts))
        self.assertTrue(any(part.startswith("ARTICULO 2") for part in parts))

    def test_strip_front_matter_pages_discards_cover_and_index(self):
        pages = [
            "REGLAMENTO DE REGIMEN INTERNO\nPortada",
            "INDICE\nCAPITULO I .... 3\nARTICULO 1 .... 4",
            "CAPITULO I. DE LAS PERSONAS SOCIAS\nARTICULO 1. DE LAS PERSONAS SOCIAS\nContenido real",
        ]
        stripped = indexador.strip_front_matter_pages(pages)
        self.assertEqual(len(stripped), 1)
        self.assertTrue(stripped[0].startswith("CAPITULO I"))

    def test_split_by_sentences_max_len(self):
        text = "Uno. Dos. Tres."
        chunks = indexador.split_by_sentences(text, max_len=10)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 10)

    def test_chunk_text_structured_fallback(self):
        text = "ARTICULO 1 Uno. Dos. Tres."
        chunks = indexador.chunk_text(text, chunk_size=15, overlap=2)
        self.assertGreaterEqual(len(chunks), 2)


class TestLargeChunkSplit(unittest.TestCase):
    def test_split_large_chunk_respects_token_limit(self):
        sample = "Uno dos tres cuatro cinco.\n\nSeis siete ocho nueve diez."

        def fake_count_tokens(value):
            return len(value.split())

        with patch.object(indexador, "count_tokens", side_effect=fake_count_tokens):
            chunks = indexador.split_large_chunk(sample, max_tokens=5)

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(fake_count_tokens(chunk), 5)


class TestNoiseDetection(unittest.TestCase):
    def test_is_faq_markdown_file(self):
        self.assertTrue(indexador.is_faq_markdown_file("FAQ.md"))
        self.assertTrue(indexador.is_faq_markdown_file("/tmp/docs/FAQ.md"))
        self.assertFalse(indexador.is_faq_markdown_file("guide.md"))

    def test_is_page_index_chunk(self):
        self.assertTrue(indexador.is_page_index_chunk("12"))
        self.assertTrue(indexador.is_page_index_chunk("12/34"))
        self.assertTrue(indexador.is_page_index_chunk("pag. 3"))
        self.assertFalse(indexador.is_page_index_chunk("Articulo 3"))

    def test_is_title_like(self):
        self.assertTrue(indexador.is_title_like("TITULO I"))
        self.assertFalse(indexador.is_title_like("Titulo con demasiadas palabras y texto"))
        self.assertFalse(indexador.is_title_like("Titulo con punto."))

    def test_is_non_relevant_title_chunk(self):
        self.assertTrue(indexador.is_non_relevant_title_chunk("INDICE"))
        self.assertTrue(indexador.is_non_relevant_title_chunk("CAPITULO I"))
        self.assertFalse(indexador.is_non_relevant_title_chunk("ARTICULO 1"))

    def test_is_noise_chunk(self):
        self.assertTrue(indexador.is_noise_chunk("INDICE"))
        self.assertFalse(indexador.is_noise_chunk("ARTICULO 1 Texto"))

    def test_is_toc_like_chunk_by_dotted_leaders(self):
        text = "CAPITULO I .... 3\nCAPITULO II .... 7"
        self.assertTrue(indexador.is_toc_like_chunk(text))
        self.assertTrue(indexador.is_noise_chunk(text))


if __name__ == "__main__":
    unittest.main()

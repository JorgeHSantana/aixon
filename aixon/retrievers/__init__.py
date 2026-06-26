"""aixon vendor retrievers — bases genéricas de retrieval (Weaviate/Ragie/Tavily).

Cada módulo importa o SDK do vendor lazy (dentro dos métodos), atrás de um extra
opcional (aixon[weaviate]/[ragie]/[tavily]). Importar este pacote nunca exige um
SDK de vendor instalado. As classes são Retriever neutros (as_tool dual)."""

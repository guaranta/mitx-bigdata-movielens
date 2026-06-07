# mitx-bigdata-movielens

**MITx xPro — Data Science and Big Data Analytics**

Big Data na prática: filtragem colaborativa (MovieLens 100k) e redução de dimensionalidade com PCA/k-means no caso genômico.

---

## Objetivos de estudo

O xPro Big Data combina **escala** (100k ratings) com **redução de dimensionalidade** (PCA em dados genômicos). Este repositório implementa: **(1)** user-based collaborative filtering com avaliação hold-out (RMSE/MAE); **(2)** port Python do pipeline MATLAB de k-mers → PCA → k-means; **(3)** métricas de avaliação rigorosas. O objetivo é entender recomendação como problema de **vizinhança em matriz esparsa** e PCA como ferramenta de **compressão com perda controlada**.

---

## Figuras e interpretação

### PCA + k-means em k-mers (genomics port)

![Clusters em projeção PC1-PC2](docs/figures/kmer_pca_clusters.png)

Cada ponto é um fragmento de DNA representado por frequências de k-mers (dimensão ~256). Após PCA para 2D, k-means separa 3 clusters coloridos. Fragmentos próximos compartilham composição de k-mers — análogo a **segmentar logs** ou **tráfego de rede** por assinatura estatística. O genoma ensina que alta dimensão exige redução antes de clustering — mesma lógica em SIEM com milhares de features por evento.

### Collaborative filtering — erros de predição

![Histograma de erros de predição](docs/figures/cf_errors.png)

A distribuição de erros (predito − real) centra-se em zero com desvio ~0.9 estrelas. RMSE penaliza erros grandes (filmes que o usuário ama/odeia e o modelo erra). Em produção: recomendação de conteúdo, **sugestão de playbooks SOAR**, ou **ranking de alertas** — o mesmo framework de "prever preferência/resposta a partir de vizinhos".

---

## Módulos

| Módulo | Técnica | Comando |
|--------|---------|---------|
| `movielens/` | User-based CF | `python movielens/run.py` |
| `pca_lda/` | k-mer PCA + k-means | `python pca_lda/run.py` |
| `evaluation/` | RMSE, silhouette | import |

> MovieLens: download em `data/README.md` (não redistribuível).

## Setup

```bash
pip install -r requirements.txt
python docs/generate_figures.py
```

---

## Aprendizados e aplicação no mercado

Big Data não é Hadoop por Hadoop — é **decisão sobre vizinhança, esparsidade e dimensionalidade**. CF alimenta recomendação e personalização; PCA alimenta compressão e visualização de dados de alta dimensão. Para CTO em plataforma de IA, este repo demonstra avaliação rigorosa (hold-out, RMSE) e port de pipelines legados (MATLAB → Python) — competências de **modernização de stack analítica** sem perder rigor estatístico.

---

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)

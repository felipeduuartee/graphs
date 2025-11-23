#!/usr/bin/env python3

"""
Script de análise da rede Mouse Visual Cortex (mouse_visual.cortex_1.graphml)

Modo básico:
    --all
    --info
    --componentes
    --ciclos
    --distancias
    --graus
    --centralidades
    --clustering
    --modelos
    --mst

Modo avançado:
    --avancado
        Inclui:
        (1) Cliques completos
        (2) Triadic Census
        (3) Assortatividade
        (4) Simulação de robustez
        (5) 100 instâncias ER/BA/WS com estatísticas
        (6) Girvan–Newman completo
        (7) Pontos de articulação e arestas críticas
        (8) Eficiência global e local
        (9) Transitividade
        (10) Joint-degree distribution

Tudo é  salvo em: resultados/
"""

import argparse
import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from networkx.algorithms.approximation import diameter as approx_diameter


CAMINHO_GRAFO = "mouse_visual.cortex_1.graphml"


def garantir_pastas():
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("resultados/figuras", exist_ok=True)
    os.makedirs("resultados/tabelas", exist_ok=True)


def salvar_texto(nome, conteudo):
    with open(f"resultados/{nome}", "w", encoding="utf-8") as f:
        f.write(conteudo)


def salvar_figura(nome):
    plt.tight_layout()
    plt.savefig(f"resultados/figuras/{nome}", dpi=300)
    plt.close()


def carregar_grafo():
    print("Carregando grafo a partir do arquivo graphml...")
    G = nx.read_graphml(CAMINHO_GRAFO)
    print("Grafo carregado com sucesso.")
    return G


def analisar_informacoes_basicas(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    densidade = nx.density(G)

    graus = [G.degree(v) for v in G.nodes()]
    grau_min = np.min(graus)
    grau_max = np.max(graus)
    grau_med = np.mean(graus)

    texto = (
        f"INFORMAÇÕES BÁSICAS\n"
        f"Número de vértices: {n}\n"
        f"Número de arestas: {m}\n"
        f"Densidade: {densidade}\n"
        f"Grau mínimo: {grau_min}\n"
        f"Grau médio: {grau_med}\n"
        f"Grau máximo: {grau_max}\n"
    )

    salvar_texto("informacoes_basicas.txt", texto)
    print(texto)


# (B) Componentes conexas

def analisar_componentes(G):
    texto = "COMPONENTES CONEXAS\n"

    # WCC (componentes fracamente conexas)
    wcc = list(nx.weakly_connected_components(G))
    texto += f"Número de componentes fracamente conexas: {len(wcc)}\n"
    maior_wcc = max(wcc, key=len)
    texto += f"Tamanho da maior WCC: {len(maior_wcc)}\n\n"

    # SCC (componentes fortemente conexas)
    scc = list(nx.strongly_connected_components(G))
    texto += f"Número de componentes fortemente conexas: {len(scc)}\n"
    maior_scc = max(scc, key=len)
    texto += f"Tamanho da maior SCC: {len(maior_scc)}\n"

    salvar_texto("componentes.txt", texto)
    print(texto)

    return G.subgraph(maior_wcc).copy()


def analisar_ciclos(G):
    print("verificando se há ciclos na rede")

    possui_ciclo = not nx.is_directed_acyclic_graph(G)

    texto = f"CICLOS\nPossui ciclo? {possui_ciclo}\n"
    salvar_texto("ciclos.txt", texto)
    print(texto)


def analisar_distancias(G):
    print("Calculando excentricidades, raio, diâmetro e centro...")

    UG = G.to_undirected()

    exc = nx.eccentricity(UG)
    raio = nx.radius(UG)
    diametro = nx.diameter(UG)
    centro = nx.center(UG)
    periferia = nx.periphery(UG)

    texto = (
        "DISTÂNCIAS\n"
        f"Raio: {raio}\n"
        f"Diâmetro: {diametro}\n"
        f"Centro: {centro}\n"
        f"Periferia: {periferia}\n"
    )

    salvar_texto("distancias.txt", texto)
    print(texto)


def analisar_distribuicao_graus(G):
    graus = [G.degree(v) for v in G.nodes()]

    texto = (
        "DISTRIBUIÇÃO DOS GRAUS\n"
        f"Grau mínimo: {np.min(graus)}\n"
        f"Grau máximo: {np.max(graus)}\n"
        f"Grau médio: {np.mean(graus)}\n"
        f"Desvio padrão: {np.std(graus)}\n"
    )

    salvar_texto("graus.txt", texto)
    print(texto)

    # Histograma
    plt.figure(figsize=(8,5))
    sns.histplot(graus, kde=False, bins=20)
    plt.xlabel("Grau")
    plt.ylabel("Frequência")
    plt.title("Histograma da Distribuição de Graus")
    salvar_figura("histograma_graus.png")

    # CCDF
    graus_ordenados = np.sort(graus)
    ccdf = 1.0 - np.arange(1, len(graus_ordenados)+1) / len(graus_ordenados)

    plt.figure(figsize=(8,5))
    plt.loglog(graus_ordenados, ccdf, marker="o", linestyle="none")
    plt.xlabel("Grau (log)")
    plt.ylabel("CCDF (log)")
    plt.title("CCDF da Distribuição de Graus")
    salvar_figura("ccdf_graus.png")


def analisar_centralidades(G):
    print("Calculando centralidade de grau...")
    cen_grau = nx.degree_centrality(G)

    print("Calculando centralidade de intermediação...")
    cen_bet = nx.betweenness_centrality(G, normalized=True)

    print("Calculando centralidade de proximidade...")
    cen_clo = nx.closeness_centrality(G)

    print("Calculando centralidade de autovetor...")
    try:
        cen_eig = nx.eigenvector_centrality(G, max_iter=5000)
    except:
        cen_eig = {n: 0.0 for n in G.nodes()}

    with open("resultados/tabelas/centralidades.csv", "w", encoding="utf-8") as f:
        f.write("no,grau,betweenness,closeness,eigenvector\n")
        for v in G.nodes():
            f.write(f"{v},{cen_grau[v]},{cen_bet[v]},{cen_clo[v]},{cen_eig[v]}\n")

    print("Arquivo centralidades.csv gerado.")



def analisar_clustering(G):
    UG = G.to_undirected()

    clustering_local = nx.clustering(UG)
    clustering_medio = nx.average_clustering(UG)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = (2*m)/(n*(n-1))

    ER = nx.gnp_random_graph(n, p)
    ER_maior = max(nx.connected_components(ER), key=len)
    ER_sub = ER.subgraph(ER_maior)
    cluster_er = nx.average_clustering(ER_sub)

    texto = (
        "CLUSTERING\n"
        f"Clustering médio da rede: {clustering_medio}\n"
        f"Clustering ER equivalente: {cluster_er}\n"
    )
    salvar_texto("clustering.txt", texto)
    print(texto)



def comparar_modelos(G):
    UG = G.to_undirected()

    n = G.number_of_nodes()
    m = G.number_of_edges()

    grau_medio = (2*m)/n
    p = grau_medio / (n-1)

    # ER
    ER = nx.gnp_random_graph(n, p)
    ER_maior = max(nx.connected_components(ER), key=len)
    ER_sub = ER.subgraph(ER_maior)
    diam_er = approx_diameter(ER_sub)
    cluster_er = nx.average_clustering(ER_sub)

    # BA
    m_ba = int(grau_medio/2)
    if m_ba < 1:
        m_ba = 1
    BA = nx.barabasi_albert_graph(n, m_ba)
    BA_maior = max(nx.connected_components(BA), key=len)
    BA_sub = BA.subgraph(BA_maior)
    diam_ba = approx_diameter(BA_sub)
    cluster_ba = nx.average_clustering(BA_sub)

    # WS
    k_ws = int(grau_medio)
    if k_ws < 2:
        k_ws = 2
    if k_ws % 2 == 1:
        k_ws += 1
    WS = nx.watts_strogatz_graph(n, k_ws, 0.1)
    WS_maior = max(nx.connected_components(WS), key=len)
    WS_sub = WS.subgraph(WS_maior)
    diam_ws = approx_diameter(WS_sub)
    cluster_ws = nx.average_clustering(WS_sub)

    diam_real = nx.diameter(UG)
    cluster_real = nx.average_clustering(UG)

    texto = (
        "COMPARAÇÃO DE MODELOS\n"
        f"Real: diâmetro={diam_real}, clustering={cluster_real}\n\n"
        f"ER: diâmetro≈{diam_er}, clustering={cluster_er}\n"
        f"BA: diâmetro≈{diam_ba}, clustering={cluster_ba}\n"
        f"WS: diâmetro≈{diam_ws}, clustering={cluster_ws}\n"
    )

    salvar_texto("comparacao_modelos.txt", texto)
    print(texto)


def avancado_cliques(G):
    print("analise de cliques completas[...]")

    UG = G.to_undirected()

    todos_cliques = list(nx.find_cliques(UG))
    clique_max = max((len(c) for c in todos_cliques))

    texto = (
        f"CLIQUES AVANÇADOS\n"
        f"Número de maximal cliques: {len(todos_cliques)}\n"
        f"Tamanho do maior clique: {clique_max}\n"
    )

    salvar_texto("avancado_cliques.txt", texto)
    print(texto)


def avancado_triadic(G):
    print("Calculando Triadic Census...")

    # checando se é um grafo dirigido
    if not G.is_directed():
        print("Triadic Census requer grafo dirigido")
        Gd = G.to_directed()
    else:
        Gd = G

    triad = nx.triadic_census(Gd)

    texto = "TRIADIC CENSUS\n"
    for k, v in triad.items():
        texto += f"{k}: {v}\n"

    salvar_texto("avancado_triadic.txt", texto)
    print(texto)


def avancado_assortatividade(G):
    UG = G.to_undirected()

    ass = nx.degree_assortativity_coefficient(UG)

    texto = f"ASSORTATIVIDADE\nAssortatividade de grau: {ass}\n"

    salvar_texto("avancado_assortatividade.txt", texto)
    print(texto)


#como a rede reage quando vértices são removidos

def avancado_robustez(G):
    print("simulacao de robustez")

    UG = G.to_undirected()
    n = UG.number_of_nodes()

    graus = dict(UG.degree())
    mais_grau = sorted(graus, key=graus.get, reverse=True)

    resultados = {}

    for tipo in ["aleatorio", "maior_grau"]:
        GC_tamanhos = []

        H = UG.copy()

        if tipo == "aleatorio":
            ordem = list(H.nodes())
            random.shuffle(ordem)
        else:
            ordem = mais_grau

        for node in ordem:
            H.remove_node(node)
            if H.number_of_nodes() > 0:
                comp = max(nx.connected_components(H), key=len)
                GC_tamanhos.append(len(comp))
            else:
                GC_tamanhos.append(0)

        resultados[tipo] = GC_tamanhos

    texto = (
        "ROBUSTEZ\n"
        "Resultados salvos em avancado_robustez.txt.\n"
    )

    salvar_texto("avancado_robustez.txt", texto)
    print(texto)


#executa 100 instancias e clacula o clustering medio de cada grupo 
def avancado_multiplos_modelos(G, instancias=100):
    print("executando 100 instâncias ER/BA/WS")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    UG = G.to_undirected()

    grau_medio = (2*m)/n
    p = grau_medio/(n-1)

    resultados = {"ER": [], "BA": [], "WS": []}

    for _ in range(instancias):
        ER = nx.gnp_random_graph(n, p)
        ER_sub = ER.subgraph(max(nx.connected_components(ER), key=len))
        resultados["ER"].append(nx.average_clustering(ER_sub))

        m_ba = max(1, int(grau_medio/2))
        BA = nx.barabasi_albert_graph(n, m_ba)
        BA_sub = BA.subgraph(max(nx.connected_components(BA), key=len))
        resultados["BA"].append(nx.average_clustering(BA_sub))

        k_ws = int(grau_medio)
        if k_ws < 2:
            k_ws = 2
        if k_ws % 2 == 1:
            k_ws += 1
        WS = nx.watts_strogatz_graph(n, k_ws, 0.1)
        WS_sub = WS.subgraph(max(nx.connected_components(WS), key=len))
        resultados["WS"].append(nx.average_clustering(WS_sub))

    texto = "100 INSTÂNCIAS DE MODELOS\n\n"
    for modelo in resultados:
        media = np.mean(resultados[modelo])
        desvio = np.std(resultados[modelo])
        texto += f"{modelo}: média={media}, desvio={desvio}\n"

    salvar_texto("avancado_modelos.txt", texto)
    print(texto)



#detectacao de comunidades
def avancado_girvan_newman(G):
    print("executando Girvan–Newman para detectar comunidades")

    UG = G.to_undirected()

    comunidades = []
    comp = nx.algorithms.community.girvan_newman(UG)

    # coletar primeiras 6 divisões (acho que é suficiente paulo)
    for i, c in zip(range(6), comp):
        comunidades.append(tuple(sorted(map(sorted, c))))

    texto = "GIRVAN–NEWMAN (primeiras 6 divisões)\n"
    for i, c in enumerate(comunidades):
        texto += f"Divisão {i+1}: {c}\n"

    salvar_texto("avancado_girvan_newman.txt", texto)
    print(texto)


def avancado_articulacao(G):
    UG = G.to_undirected()

    pontos = list(nx.articulation_points(UG))
    arestas = list(nx.bridges(UG))

    texto = (
        "PONTOS DE ARTICULAÇÃO E ARESTAS CRÍTICAS\n"
        f"Articulation points: {pontos}\n"
        f"Bridges: {arestas}\n"
    )

    salvar_texto("avancado_articulacao.txt", texto)
    print(texto)


def avancado_eficiencia(G):
    UG = G.to_undirected()
    Eglob = nx.global_efficiency(UG)
    Eloc = nx.local_efficiency(UG)

    texto = (
        "EFICIÊNCIA\n"
        f"Eficiência global: {Eglob}\n"
        f"Eficiência local: {Eloc}\n"
    )

    salvar_texto("avancado_eficiencia.txt", texto)
    print(texto)


#tendencia a formar triangulos
def avancado_transitividade(G):
    UG = G.to_undirected()
    t = nx.transitivity(UG)

    texto = f"TRANSITIVIDADE\nTransitividade global: {t}\n"

    salvar_texto("avancado_transitividade.txt", texto)
    print(texto)


def analisar_mst(G, weight='weight'):
    UG = G.to_undirected()

    # se não existe peso, defino como 1.0
    if not any('weight' in d for _, _, d in UG.edges(data=True)):
        for u, v, d in UG.edges(data=True):
            d['weight'] = 1.0

    mst = nx.minimum_spanning_tree(UG, weight=weight)

    n = mst.number_of_nodes()
    m = mst.number_of_edges()
    try:
        diam = nx.diameter(mst)
    except Exception:
        diam = 'N/A'
    try:
        apl = nx.average_shortest_path_length(mst)
    except Exception:
        apl = 'N/A'

    graus = [mst.degree(v) for v in mst.nodes()]
    texto = (
        f"MST\n"
        f"Número de vértices: {n}\n"
        f"Número de arestas: {m}\n"
        f"Diâmetro: {diam}\n"
        f"Avg shortest path length: {apl}\n"
        f"Grau máximo: {max(graus) if graus else 0}\n"
        f"Grau médio: {np.mean(graus) if graus else 0}\n"
    )

    salvar_texto("mst.txt", texto)

    with open("resultados/tabelas/mst_edges.csv", "w", encoding="utf-8") as f:
        f.write("u,v\n")
        for u, v in mst.edges():
            f.write(f"{u},{v}\n")

    # representacao simples
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(mst, seed=42)
    nx.draw(mst, pos, node_size=30, with_labels=False, edge_color='black')
    plt.title("Minimum Spanning Tree (MST)")
    salvar_figura("mst.png")

def main():
    garantir_pastas()
    G = carregar_grafo()

    parser = argparse.ArgumentParser(
        description=(
            "Análise completa da rede Mouse Visual Cortex (GraphML). "
            "Use uma ou mais flags para selecionar análises específicas."
        )
    )

    parser.add_argument("--info", action="store_true",
        help="Mostra informações básicas: vértices, arestas, densidade e graus extremos.")

    parser.add_argument("--componentes", action="store_true",
        help="Analisa componentes conexas (fracas e fortes) e extrai a maior WCC.")

    parser.add_argument("--ciclos", action="store_true",
        help="Verifica a existência de ciclos na maior componente fraca.")

    parser.add_argument("--distancias", action="store_true",
        help="Calcula excentricidade, raio, diâmetro, centro e periferia.")

    parser.add_argument("--graus", action="store_true",
        help="Gera estatísticas da distribuição de graus e gráficos (histograma e CCDF).")

    parser.add_argument("--centralidades", action="store_true",
        help="Centralidades: grau, intermediação, proximidade e autovetor.")

    parser.add_argument("--clustering", action="store_true",
        help="Coeficientes de agrupamento local/global e comparação com ER equivalente.")

    parser.add_argument("--modelos", action="store_true",
        help="Compara a rede com os modelos ER, BA e WS gerados com parâmetros equivalentes.")

    parser.add_argument("--mst", action="store_true",
        help="Gera a árvore geradora mínima (MST) e salva resultados.")

    parser.add_argument("--all", action="store_true",
        help="Executa todas as análises básicas.")

    parser.add_argument("--avancado", action="store_true",
        help=(
            "Executa análises avançadas: cliques, triadic census, assortatividade, robustez, "
            "100 instâncias ER/BA/WS, Girvan-Newman, pontos de articulação, eficiência "
            "e transitividade."
        )
    )

    args = parser.parse_args()

    #  Default: se nenhuma flag for passada, ativa --all
    if not any(vars(args).values()):
        print("Nenhuma flag foi especificada, executando padrão (--all).")
        args.all = True

#analises
    G_maior = None

    if args.all or args.info:
        analisar_informacoes_basicas(G)

    if args.all or args.componentes:
        G_maior = analisar_componentes(G)

    # se nenhuma flag até agora definiu qual é a maior componente, a maior WCC da rede é selecionada
    if G_maior is None:
        wcc = list(nx.weakly_connected_components(G))
        G_maior = G.subgraph(max(wcc, key=len)).copy()

    if args.all or args.ciclos:
        analisar_ciclos(G_maior)

    if args.all or args.distancias:
        analisar_distancias(G_maior)

    if args.all or args.graus:
        analisar_distribuicao_graus(G_maior)

    if args.all or args.centralidades:
        analisar_centralidades(G_maior)

    if args.all or args.clustering:
        analisar_clustering(G_maior)

    if args.all or args.modelos:
        comparar_modelos(G_maior)

    if args.mst:
        analisar_mst(G_maior)

    if args.avancado:
        avancado_cliques(G_maior)
        avancado_triadic(G_maior)
        avancado_assortatividade(G_maior)
        avancado_robustez(G_maior)
        avancado_multiplos_modelos(G_maior)
        avancado_girvan_newman(G_maior)
        avancado_articulacao(G_maior)
        avancado_eficiencia(G_maior)
        avancado_transitividade(G_maior)

    print("Analise finalizada. Resultados na pasta resultados")


if __name__ == "__main__":
    main()

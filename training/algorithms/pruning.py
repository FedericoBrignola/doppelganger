import copy
from math import inf

from training.decisionTree3 import DecisionTree


def pruning(decisionTree: DecisionTree, nodeId: int) -> None:
    """
    Dato un albero decisionale e un nodo, viene effettuato il pruning del sottoalbero che ha il nodo come radice.
    Una spiegazione grafica del funzionamento di questa funzione e' presente nella relazione del progetto all'interno
    della sezione del primo esperimento
    """

    """        
    Un dominio e' rappresentato come una terna T tale che 
    - T[0] e' il minimo valore del dominio (dominio maggiore o uguale di T[0]), 
    - T[1] e' il solo valore del dominio (dominio = T[1])
    - T[2] e' il massimo valore del dominio (dominio minore di T[2])

    dominio = [t[0],t[2]) or t[1]

    None rappresenta un "non limite"
    """

    domain = dict()
    # inizializzazione del dominio di ogni features
    for feature in decisionTree.features:
        domain[feature] = [-inf, None, inf]

    # Calcolo dei domini da nodeId risalendo fino alla radice
    currentNode = decisionTree.nodes[nodeId]
    while not currentNode.isRoot():
        currentLabel = currentNode.label
        currentNode = decisionTree.nodes[currentNode.parent]

        # Limite superiore feature
        if currentLabel is True:
            # in questo caso, ci troviamo in un nodo dove, il parent e' della forma 'feature < threshold' e il ramo che unisce il nodo al padre ha label True
            if domain[currentNode.feature][2] == inf:
                domain[currentNode.feature][2] = currentNode.threshold
            elif currentNode.threshold < domain[currentNode.feature][2]: #TODO: dovrebe essere <= no <
                # se il dominio non e' inf, vuol dire che abbiamo gia' trovato un nodo che faceva un taglio su quella feature, tuttavia
                # se il nodo attuale dovesse avere un valore di threshold minore di quello precedentemente incontrato, allora abbiamo un inconsistenza nell'albero
                # questo elif non dovrebbe mai essere vero. In caso contrario, ci sono degli errori nel codice
                decisionTree.printToFile("local/broken")
                print("[ Error ] Trovata inconsistenza nel partizionamento della feature ", currentNode.feature)

        # limite inferiore feature
        elif currentLabel is False:
            # analogo all'if sopra
            if domain[currentNode.feature][0] == -inf:
                domain[currentNode.feature][0] = currentNode.threshold
            elif currentNode.threshold >= domain[currentNode.feature][0]:
                print("[ Error ] Trovata inconsistenza nel partizionamento della feature ", currentNode.feature)
                decisionTree.printToFile("local/broken")

        # Uguaglianza
        else:
            if domain[currentNode.feature][1] is not None:
                # anche questo if e' per trovare inconsistenze nell'albero, pertanto non dovrebbe mai essere true
                print("[ Error ] Trovata inconsistenza nel partizionamento della feature ", currentNode.feature)
                decisionTree.printToFile("local/broken")
            domain[currentNode.feature][1] = currentLabel

    # ottenuti i domini, passiamo alla rimozione dei nodi inconsistenti nel nuovo sottoalbero inserito
    stack = [(nodeId, copy.deepcopy(domain))]
    while len(stack) > 0:
        nid, cdomain = stack.pop()
        node = decisionTree.nodes.get(nid)
        if node.isLeaf():
            continue
        nodeFeatureDomain = cdomain[node.feature]
        replacedBy = None
        if not decisionTree.features[node.feature].isNumerical():
            # Se la feature non e' ordinata e il dominio[1] ha un qualche valore, allora il nodo attuale e' inconsistente
            if nodeFeatureDomain[1] is not None:
                # rimuove il nodo e lo rimpiazza con il figlio sul ramo con label corrispondente al dominio individuato
                replacedBy = decisionTree.removeNode(nid, nodeFeatureDomain[1])
        elif node.threshold <= nodeFeatureDomain[0]:
            # analogo all'if precedente, rimuove il nodo e lo rimpiazza con il ramo "False"
            replacedBy = decisionTree.removeNode(nid, False)
        elif node.threshold >= nodeFeatureDomain[2]:
            # rimuove il nodo e lo rimpiazza con il ramo "True"
            replacedBy = decisionTree.removeNode(nid, True)

        # se il nodo corrente non e' stato rimpiazzato, allora restringiamo i domini in base al nodo corrente e inseriamo nello stack
        # i nodi figli, altrimenti inseriamo nello stack il nodo che ha rimpiazzato quello corrente senza alterare i domini
        if replacedBy is None:
            for label, cid in node.children.items():
                nextDomain = cdomain
                if nodeId == nid:
                    nextDomain = copy.deepcopy(cdomain)
                    match label:
                        case True:
                            nextDomain[node.feature][2] = node.threshold
                        case False:
                            nextDomain[node.feature][0] = node.threshold
                        case _:
                            nextDomain[node.feature][1] = label
                stack.append((cid, nextDomain))
        else:
            stack.append((replacedBy, cdomain))

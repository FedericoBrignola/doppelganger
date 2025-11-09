from __future__ import annotations
import copy
import random
from datetime import datetime
import pickle
import graphviz


class Domain:
    def __init__(self, minV: int | float = None, maxV: int | float = None, step: int | float = None, values: list[str] = None, tests: list = None):
        self.numerical = values is None
        self.values = values
        self.min = minV
        self.max = maxV
        self.step = step
        # self.tFunctions = tests

    def __iter__(self):
        if self.numerical:
            current = self.min
            while current <= self.max:
                yield current
                current += self.step
        else:
            return self.values.__iter__()

    # def tests(self):
    #     return self.tFunctions.__iter__()

    def getRand(self):
        if self.numerical:
            return random.uniform(self.min, self.max)
        return random.choice(self.values)

    # def getTest(self):
    #     return random.choice(self.tFunctions)

    def isNumerical(self) -> bool:
        return self.numerical


class FeatureSet(dict):
    """
    Rappresenta l'insieme di features

    Methods
    -------
    add(name: str, isOrdered: bool)
        aggiunge una feature all'insieme
    isOrdered(name: str) -> bool:
        restituisce True se una feature ha un ordinamento, False altrimenti
    toCsv(dictionary: dict[str, any] = None, **kwargs) -> str
        Permette la conversione da dizionario a stringa formattata come csv.
    """

    def __init__(self):
        # self._features: dict[str, tuple[list, int, int, int]] = dict()
        super().__init__()
        # self._features: dict[str, bool] = dict()

    def addNew(self, name: str, values: list = None, minValue: int = None, maxValue: int = None, step: int = 1):
        self[name] = Domain(minValue, maxValue, step, values)

    # @DeprecationWarning
    def add(self, name: str, isOrdered: bool, domain: Domain | None = None):
        """
        Aggiunge una feature all'insieme

        Parameters
        ----------
        name: str
            nome della feature da inserire
        isOrdered: bool
            True se la feature ha un ordinamento, False altrimenti
        domain: Domain | None = None
            Dominio della features
        """
        # TODO: Solo per retrocompatibilita', guarda `addNew`
        self[name] = isOrdered

        #self[name] = [isOrdered, domain]  # (None if isOrdered else list(), 0, 100, 1)

    # TODO: Solo per retrocompatibilita', guarda `isNumerical` di Domain
    def isOrdered(self, name: str) -> bool:
        """
        Indica se una feature ha un ordinamento

        Parameters
        ----------
        name: str
            nome della feature

        Return
        ------
        bool: True se la feature ha un ordinamento, False altrimenti
        """
        # return self._features[name]
        return self[name]

    def toCsv(self, dictionary: dict[str, any] = None) -> str:
        """
        Permette la conversione da dizionario a stringa formattata come csv.

        Parameters
        ----------
        dictionary: dict[str, any]
            Un dizionario con key = featureName, value = valoreOsservato

        Return
        ------
        str: valori osservati delle features come riga CSV
        """
        return ','.join([str(dictionary[f]) for f in self.keys()])

    def __str__(self):
        return ','.join(self.keys())


class DecisionNode:
    """
    Struttura dati di un nodo di un albero decisionale

    Methods
    -------
    isRoot() -> bool:
        True se il nodo e' la radice dell'albero, False altrimenti
    isLeaf() -> bool:
        True se il nodo e' una foglia, False altrimenti
    getChildren() -> list[int]
        Restituisce la lista di id dei suoi nodi figli
    """

    def __init__(self, feature: str = None, threshold: any = None, out: any = None) -> None:
        """
        Costruttore DecisionNode

        Parameters
        ----------
        feature: str = None
            nome della feature collegata al nodo
        threshold: any = None
            Valore di threshold per i nodi interni con ordinamento
        out: any = None
            Valore di output del nodo.
            Se si tratta di un nodo interno, conterra' la classificazione di "default" del nodo
            Se si tratta di un nodo folia, conterra' la classificazione del ramo
        """
        self.feature: str = feature
        self.threshold: any = threshold
        self.children: dict[any, int] = dict()  # key = label del ramo, value = id del nodo figlio collegato al ramo
        self.parent: int = None                 # id del nodo genitore
        self.label: any = None                  # label dell'arco che dal parent va al nodo corrente
        self.out: any = out

    def __deepcopy__(self, memo):
        """
        Permette di generare una copia di un'istanza della classe

        return: copia dell'oggetto
        """
        n = DecisionNode(self.feature, self.threshold, self.out)
        n.children = copy.deepcopy(self.children, memo)
        n.setParent(self.parent, self.label)
        return n

    def setParent(self, parentId: int, label: any):
        """
        Setta il parent di un nodo

        Parameters
        ----------
        parentId: int
            id del nodo genitore
        label: any
            label dell'arco che va dal nodo genitore al nodo corrente
        """
        self.parent = parentId
        self.label = label

    def isRoot(self) -> bool:
        """
        True se il nodo e' la radice dell'albero, False altrimenti
        """
        return self.parent is None

    def isLeaf(self) -> bool:
        """
        True se il nodo e' una foglia, False altrimenti
        """
        return self.feature is None

    def getChildren(self) -> list[int]:
        """
        Restituisce la lista di id dei suoi nodi figli
        """
        return list(self.children.values())


class DecisionTree:
    """
    Struttura dati di un albero decisionale
    """

    def __init__(self, features: FeatureSet = None):
        self.features: FeatureSet = features
        self.uuid: int = 0  # Identificativo per il prossimo nodo che viene aggiunto all'albero
        self.nodes: dict[int, DecisionNode] = dict()  # key = identificativo nodo, value = nodo corrispondente

    def __deepcopy__(self, memo):
        """
        Permette di generare una copia di un'istanza della classe
        """
        t = DecisionTree()
        t.features = self.features
        t.nodes = copy.deepcopy(self.nodes, memo)
        t.uuid = self.uuid
        return t

    @classmethod
    def load(cls, filePath: str) -> DecisionTree:
        """
        Carica un albero precedentemente salvato su file con la funzione "save"

        Parameters
        ----------
        filePath: str
            file con albero

        Return
        ------
        DecisionTree: albero decisionale risultante
        """
        with open(filePath, "rb") as file:
            tree = pickle.load(file)
        return tree

    def save(self, fileName: str, persistent: bool = True) -> str:
        """
        Salva un albero decisionale su file

        Parameters
        ----------
        fileName: str
            file su cui salvare l'albero
        persistent: bool
            se True, a ogni chiamata generera' un nuovo file, altrimenti, scrivera' sempre sullo stesso file

        Return
        ------
        str: nome del file (utile quando la persistenza = True in quanto appende il datatime attuale al nome del file)
        """
        if persistent:
            fileName += datetime.now().strftime("_%d-%H-%M-%S.dtree")
        with open(fileName + ".dtree", 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        return fileName

    def add(self, node: DecisionNode, parentId: int = None, label: any = None) -> int:
        """
        Aggiunge un nodo all'albero

        Parameters
        ----------
        node: DecisionNode
            nodo da aggiungere
        parentId: int = None
            id del nodo padre
        label: any = None
            label arco tra il parent e il nodo

        Return
        ------
        int: uuid del nodo aggiunto
        """
        node.setParent(parentId, label)
        self.nodes[self.uuid] = node
        if parentId is not None:
            self.nodes.get(parentId).children[label] = self.uuid  # Aggiunta nodo attuale con la rispettiva label tra i children del parent
        self.uuid += 1          # incremento del uuid per il prossimo nodo
        return self.uuid - 1    # return dell'uuid del nodo appena aggiunto

    def predict(self, featuresValues: dict[str, any]) -> str:
        """
        Dato un esempio come dizionario "feature: value", interroga l'albero per predire l'output

        Parameters
        ----------
        featuresValues: dict[str, any]
            dizionario tale che key=nomeFeature, value:valoreFeature

        Return
        ------
        str: azione come stringa
        """
        n = self.nodes[0]  # nodo radice
        while len(n.children) > 0:  # finche' non arriviamo ad un nodo foglia
            if self.features[n.feature].isNumerical():
                # Prende il figlio che ha come label True se il valore della feature nell'istanza e' < del valore di threshold, False altrimenti
                nodeId = n.children.get(featuresValues.get(n.feature) < n.threshold)
            else:
                # Prende il figlio che ha come label il valore della feature nell'istanza
                nodeId = n.children.get(featuresValues.get(n.feature))
            # se non ha restituito nodi, abbiamo nell'istanza un valore che non avevamo osservato durante il training
            if nodeId is None:
                break
            n = self.nodes.get(nodeId)  # scende al figlio
        return n.out  # valore di output o di default

    def queryNode(self, featuresValues: dict[str, any], nodeId: int = 0) -> int | None:
        """
        Interroga un singolo nodo dell'albero e restituisce il figlio "corrispondente"

        Parameter
        ---------
        featuresValues: dict[str, any]
            dizionario della forma key=nomeFeature value=valoreOsservato.
            La predizione avverra' sui valori di questo parametro
        nodeId: int
            nodo da interrogare
        return: uuid nodo figlio, potenzialmente None
        """
        n = self.nodes.get(nodeId)
        if n.isLeaf():
            return None
        if self.features[n.feature].isNumerical():
            # Prende il figlio che ha come label True se il valore della feature nell'istanza e' < del valore di threshold, False altrimenti
            return n.children.get(featuresValues.get(n.feature) < n.threshold)
        else:
            # Prende il figlio che ha come label il valore della feature nell'istanza
            return n.children.get(featuresValues.get(n.feature))

    def removeSubtree(self, nodeId: int) -> None:
        """
        Rimuove il sottoalbero che ha come radice il nodo dato in input

        Parameters
        ----------
        nodeId: int
            uuid nodo radice del sotto albero da rimuovere
        """
        n = self.nodes[nodeId]
        # Se il nodo e' la radice dell'albero, rimuove direttamente tutto l'albero e riazzera gli uuid
        if n.isRoot():
            self.nodes.clear()
            self.uuid = 0
            return
        stack = [nodeId]                                # Stack con i nodi da eliminare
        self.nodes.get(n.parent).children.pop(n.label)  # Elimina il nodo corrente dai children del proprio parent
        while len(stack) > 0:                           # Finche' ci sono nodi da eliminare
            n = self.nodes.pop(stack.pop())             # Rimuovi il nodo dall'albero
            stack.extend(n.getChildren())               # Inserisci tutti i figli nello stack per farli rimuovere dall'albero

    def removeNode(self, nodeId: int, branch: str) -> int:
        """
        Rimpiazza il nodo `nodeId` con il figlio nel ramo `branch`

        Parameters
        ----------
        nodeId: int
            uuid del nodo da rimuovere
        branch: str
            il ramo del nodeId che porta al nodo che rimpiazzera' il nodeId

        Return
        ------
        int: id del nodo che ha preso il suo posto (id del nodo figlio che lo ha rimpiazzato)
        """
        n = self.nodes[nodeId]
        replacerId = n.children.pop(branch, None)  # Rimuove 'branch' dai figli del nodo. Restituisce None se non esiste
        parent = n.parent
        label = n.label
        out = n.out

        # Rimuoviamo tutto il sotto albero del nodeId.
        # Nota: il ramo che ci interessa e' stato rimosso dalla lista dei figli del nodo corrente, ma non dall'albero;
        # per questo motivo, non verra' toccato da removeSubtree
        self.removeSubtree(nodeId)  # Rimuove tutto il sotto albero con radice nodeId

        # Se 'branch' non era tra i rami del nodo da rimuovere, viene inserito un nodo foglia che eredita il valore di out di default del nodo rimosso
        if not replacerId:
            return self.add(DecisionNode(out=out), parent, label)

        replacer = self.nodes[replacerId]                   # nodo che rimpiazza il nodeId
        self.nodes[parent].children[label] = replacerId     # inserisce il nodo tra i figli del parent del nodo rimosso
        replacer.setParent(parent, label)                   # setta il nuovo parent e il label al nodo
        return replacerId

    def extract(self, nodeId: int) -> DecisionTree:
        """
        Restituisce una copia del sotto albero che ha come radice il nodo dato in input

        Parameters
        ----------
        nodeId: int
            uuid nodo radice sotto albero da estrarre

        Return
        ------
        DecisionTree: sotto albero che ha come radice il nodeId
        """

        tree = DecisionTree()       # nuovo albero, quello che verra' restituito
        tree.features = copy.deepcopy(self.features)
        stack = [(None, nodeId)]    # Frontiera contenente tuple del tipo (parentId, nodeId)

        # copia di tutto il sotto albero come per il nodo radice
        while len(stack) > 0:
            parentid, id = stack.pop()
            n = copy.deepcopy(self.nodes.get(id))
            idn = tree.add(n, parentid, n.label)
            stack.extend((idn, x) for x in n.getChildren())

        return tree

    def substitute(self, nodeId: int, decisionTree: DecisionTree) -> int:
        """
        Rimpiazza il nodeId con il decisionTree in input

        Parameters
        ----------
        nodeId: int
            uuid del nodo da rimpiazzare
        decisionTree: DecisionTree
            albero da inserire al posto del nodeId

        Return
        ------
        int: uuid nodo radice del sotto albero inserito
        """
        n = copy.deepcopy(self.nodes.get(nodeId))
        self.removeSubtree(nodeId)  # Rimuove sottoalbero di nodeId

        sn = copy.deepcopy(decisionTree.nodes.get(0))  # prende la radice dell'albero in input
        idNode = self.add(sn, n.parent, n.label)  # aggiunge sn all'albero (verra' attaccato al parent di n)

        stack = [(idNode, x) for x in sn.getChildren()]  # frontiera di nodi da inserire nell'albero

        # Inserisce tutti i nodi dell'albero in input
        while len(stack) > 0:
            parentId, id = stack.pop()
            node = copy.deepcopy(decisionTree.nodes.get(id))
            newUuid = self.add(node, parentId, node.label)
            stack.extend([(newUuid, x) for x in node.getChildren()])
        return idNode

    def printToFile(self, fileName: str) -> None:
        """
        Costruisce graficamente l'albero usando Graphviz e lo salva su file svg

        Parameters
        ----------
        fileName: str
            nome del file su cui salvare l'albero
        """
        # Definizione grafo di Graphviz
        g = graphviz.Graph(fileName, edge_attr={"labelfloat": 'True', 'fontsize': '10'}, format="svg", graph_attr={"ranksep": '1'})
        queue = [0]  # Frontiera inizializzata con l'id del nodo radice dell'albero
        while len(queue) > 0:
            id = queue.pop()
            n = self.nodes[id]

            # se il nodo e' una foglia, il label dentro il nodo sara' la classificazione della foglia
            if n.isLeaf():
                nodeLabel = str(n.out)
            # se invece e' un nodo interno ed ha un ordinamento, viene mostrata la condizione di split, ovvero "featureName < threshold"
            elif self.features[n.feature].isNumerical():
                nodeLabel = f"{n.feature} < {n.threshold}"
            # altrimenti mostra solo il nome della feature
            else:
                nodeLabel = f"{n.feature}"

            g.node(str(id), nodeLabel)  # inserisce nodo nel grafo
            if not n.isRoot():
                g.edge(str(n.parent), str(id), str(n.label))  # inserisce arco tra il parent e il nodo, sull'arco viene scritto il label del nodo
            queue.extend(n.getChildren())
        g.render(cleanup=True)

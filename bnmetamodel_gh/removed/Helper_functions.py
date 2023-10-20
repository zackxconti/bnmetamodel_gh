import networkx as nx
import io
import os
import re
import csv
import copy
import matplotlib
import matplotlib.pyplot as plt

from typing import Any, List, Tuple, Union, TYPE_CHECKING

matplotlib.use("TkAgg")

if TYPE_CHECKING:
    # TODO: This file still relies on this import so make sure to check this
    from libpgm.graphskeleton import GraphSkeleton


def discrete_estimatebn(
    learner, data, skel, pvalparam: float = 0.05, indegree: float = 0.5
):
    """
    Estimate the structure of a discrete Bayesian network from data.

    Arguments
    ---------
    learner : PGMLearner # TODO: correct?
        An instance of the :doc:`PGMLearner <pgmlearner>` class.
    data : list
        A list of dicts containing samples from the network in {vertex: value}
        format.
    skel : GraphSkeleton
        An instance of the :doc:`GraphSkeleton <graphskeleton>` class
        containing vertex and edge data.
    pvalparam : float
        The p-value threshold for edge inclusion.
    indegree : float
        The maximum indegree of the graph skeleton.
    """
    # TODO: remove pvalparam and indegree, as they are not used.

    if not isinstance(data, list) or not all(
        isinstance(x, dict) for x in data
    ):
        raise SyntaxError("Data must be provided as a list of dictionaries.")

    # learn parameters
    bn = learner.discrete_mle_estimateparams(skel, data)

    return bn


def alphanum_key(s: str) -> list[Union[str, int]]:
    """
    Return a list of string and number chunks from string `s`.

    Arguments
    ---------
    s : str
        The string to be split.

    Returns
    -------
    list
        A list of string and number chunks from string `s`.

    Example
    -------
    >>> alphanum_key("z23a")
    ["z", 23, "a"]
    """
    key = re.split(r"(\d+)", s)
    key[1::2] = map(int, key[1::2])
    return key


def len_csvdata(csv_file_path: str) -> int:
    """
    Return the number of rows in the csv file at `csv_file_path`. The csv file
    must be formatted such that the first row contains the column names.

    Arguments
    ---------
    csv_file_path : str
        The path to the csv file.

    Returns
    -------
    int
        The number of rows in the csv file.
    """
    data = []
    with io.open(csv_file_path, "rb") as f:
        reader = csv.reader(f, dialect=csv.excel)

        for row in reader:
            data.append(row)

    length = len(data)

    return length


def ranges(data: List[dict]) -> dict:
    """
    Return a dict of the ranges of the variables in `data`.

    Arguments
    ---------
    data : list[dict]
        A list of dicts containing samples from the network in the correct
        format: #TODO example

    Returns
    -------
    dict
        A dict of the ranges of the variables in `data`.
    """
    if not isinstance(data, list) or not all(
        isinstance(x, dict) for x in data
    ):
        raise SyntaxError("Data must be provided as a list of dictionaries.")

    cdata = copy.deepcopy(data)

    # establish ranges
    ranges = dict()
    for variable in cdata[0].keys():
        ranges[variable] = [float("infinity"), float("infinity") * -1]

    for sample in cdata:
        for var in sample.keys():
            if sample[var] < ranges[var][0]:
                ranges[var][0] = round(sample[var], 1)
            if sample[var] > ranges[var][1]:
                ranges[var][1] = round(sample[var], 1)

    return ranges


def draw_barchartpd(binranges, probabilities) -> plt.bar:
    """
    Draw a bar chart of the probabilities of the bins in `binranges`.

    Arguments
    ---------
    binranges : list[list[float, float]]
        A list of bin ranges.
    probabilities : list[float]
        A list of probabilities.

    Returns
    -------
    matplotlib.pyplot.bar
        A bar chart of the probabilities of the bins in `binranges`. The
        function also shows the plot.
    """
    xticksv = []
    widths = []
    edge = []
    for index, range in enumerate(binranges):
        print(f"range {range}")
        edge.append(range[0])
        widths.append(range[1] - range[0])
        xticksv.append(((range[1] - range[0]) / 2) + range[0])
        if index == len(binranges) - 1:
            edge.append(range[1])

    print(f"xticks {xticksv}")
    print(f"probabilities {probabilities}")
    print(f"edge {edge}")

    b = plt.bar(
        xticksv,
        probabilities,
        align="center",
        width=widths,
        color="black",
        alpha=0.2,
    )

    # plt.xlim(edge[0], max(edge))
    plt.xticks(edge)
    plt.ylim(0, 1)
    plt.show()

    return b


def draw_histograms(
    df: pd.DataFrame,
    binwidths: Union[int, dict],
    n_rows: int,
    n_cols: int,
    maintitle: str,
    xlabel: str,
    ylabel: str,
    displayplt: bool = False,
    saveplt: bool = False,
    **kwargs,
):
    """
    Draw a histogram of the data in `df`.

    Arguments
    ---------
    df : pandas.DataFrame
        The data to be plotted.
    binwidths : int or dict
        The width of the bins. If `binwidths` is an int, all bins will have the
        same width. If `binwidths` is a dict, each variable will have a bin
        width specified by the dict.
    n_rows : int
        The number of rows of subplots.
    n_cols : int
        The number of columns of subplots.
    maintitle : str
        The title of the plot.
    xlabel : str
        The label of the x-axis.
    ylabel : str
        The label of the y-axis.
    displayplt : bool
        Whether to display the plot.
    saveplt : bool
        Whether to save the plot.
    **kwargs
        If "xlim" is in kwargs, the x-axis limits will be set to the values in
        "xlim". If "path" is in kwargs, the plot will be saved to the path
        specified. If path does not end with a "/", a "/" will be appended to
        the end of path. If path is not in kwargs, the plot will be saved to
        the path specified by the script.
    """
    fig = plt.figure(
        figsize=((750 * n_cols) / 220, (750 * n_rows) / 220), dpi=220
    )
    # t = fig.suptitle(maintitle, fontsize=4) # INFO: not used
    # t.set_poition(0.5, 1.05)

    # TODO #44: Replace df with probabilities / write bar function

    i = 0
    for var_name in list(df):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        if isinstance(binwidths, int):
            print(f"binwidths {binwidths}")
            df[var_name].hist(bins=binwidths, ax=ax, color="black")
        else:
            df[var_name].hist(bins=binwidths[var_name], ax=ax, color="black")

        ax.grid(
            color="0.2", linestyle=":", linewidth=0.1, dash_capstyle="round"
        )
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_title(var_name, fontweight="bold", size=6)
        ax.set_ylabel(ylabel, fontsize=4)  # Y label
        ax.set_xlabel(xlabel, fontsize=4)  # X label
        ax.xaxis.set_tick_params(labelsize=4)
        ax.yaxis.set_tick_params(labelsize=4)

        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"][0], kwargs["xlim"][1])

        i += 1

    fig.tight_layout()  # Improves appearance a bit.
    fig.subplots_adjust(top=0.85)  # white spacing between plots and title
    # If you want to set backgrond of figure to transpearaent do it here,
    # use facecolor='none' as argument in savefig ()
    if displayplt:
        plt.show()

    path = (
        "/Users/zack_sutd/Dropbox/SUTD/PhD/Thesis/Phase 2/Simple_truss/Plots/"
    )
    if "path" in kwargs:
        path = kwargs["path"]

    if not path.endswith("/"):
        path = f"{path}/"

    if saveplt:
        fig.savefig(
            f"{path}{maintitle}.png",
            dpi=400,
        )


def kfoldToList(
    indexList: List[int], csvData: List[List[Any]], header: List[Any]
) -> List[Any]:  # TODO: Fix Any
    """
    Converts the data from a CSV represented as a list, based on indices,
    into a new list.

    Parameters
    ----------
    indexList : list of int
        A list of indices pointing to rows in `csvData` that should be
        extracted.
    csvData : list of list
        A 2D list representing the CSV data where each inner list is a row of
        data.
    header : list
        The header row of the CSV data.

    Returns
    -------
    list
        A list comprising of the header followed by rows from `csvData` as
        indicated by `indexList`.
    """
    list = []
    list.append(header)
    for i in range(0, len(indexList)):
        list.append(csvData[indexList[i]])

    return list


def graph_to_pdf(nodes: List, edges: List[Tuple], name: str) -> None:
    """
    Create a directed graph using the given nodes and edges, and save it as a
    PDF using Graphviz.

    Parameters
    ----------
    nodes : list
        A list of nodes to be added to the graph.
    edges : list of tuple
        A list of edges where each tuple represents an edge. Each tuple
        contains two nodes, indicating the source and target node of the edge.
    name : str
        The name to use for the saved DOT and PDF files. The files will be
        saved as `name.dot` and `name.pdf` respectively.

    Note
    ----
    Requires Graphviz to be installed on the system, as the function uses the
    "dot" command line tool.

    Example
    -------
    >>> nodes = ["A", "B", "C"]
    >>> edges = [("A", "B"), ("B", "C")]
    >>> graph_to_pdf(nodes, edges, "sample_graph")
    # This will create 'sample_graph.dot' and 'sample_graph.pdf' files.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.drawing.nx_pydot.write_dot(G, name + ".dot")
    os.system("dot -Tpdf %s > %s" % (name + ".dot", name + ".pdf"))


def laplacesmooth(bn):  # TODO: add typing
    """
    Apply Laplace smoothing to the conditional probabilities of a given
    Bayesian network.

    This function modifies the Bayesian network's conditional probability
    tables (CPTs) using Laplace smoothing, which is a method to handle zero
    probabilities in a CPT by adding pseudocounts.

    Parameters
    ----------
    bn : _type_
        The Bayesian network object, which should have attributes 'V' for
        vertices and 'Vdata' for associated data like conditional
        probabilities.

    Attributes of `bn` object
    --------------------------
    bn.V : list
        List of vertices in the Bayesian network.
    bn.Vdata : dict
        Dictionary containing the data associated with each vertex, like
        conditional probabilities, number of outcomes, and parents.

    Returns
    -------
    _type_
        The modified Bayesian network object with updated conditional
        probabilities.
    """
    # TODO #46: Update laplacesmooth to align with condprobve/lmeestimateparams
    for vertex in bn.V:
        print(f"vertex {vertex}")
        numBins = bn.Vdata[vertex]["numoutcomes"]

        if not (bn.Vdata[vertex]["parents"]):
            # has no parents
            for i in range(len(bn.Vdata[vertex]["cprob"])):
                # numerator (count)
                bn.Vdata[vertex]["cprob"][i][0] += 1

                # denomenator (total count)
                bn.Vdata[vertex]["cprob"][i][1] += numBins
        else:
            for i in range(numBins):
                binindex = [str(float(i))]
                bincounts = bn.Vdata[vertex]["cprob"][str(binindex)]
                for j in range(len(bincounts)):
                    # numerator (count)
                    bincounts[j][0] += 1

                    # denomenator (total count)
                    bincounts[j][1] += numBins

    return bn


def expectedValue(
    binRanges: List[Union[Tuple[float, float], List[float]]],
    probabilities: List[float],
) -> float:
    """
    Compute the expected value based on the mean value of bins and their
    associated probabilities.

    The expected value is calculated as the sum of products of mean bin values
    and their corresponding probabilities.

    Parameters
    ----------
    binRanges : list of tuple or list of two float values
        A list where each tuple represents the minimum and maximum value for
        each bin.
    probabilities : list of float
        A list of probabilities associated with each bin. The sum of these
        probabilities should ideally be 1.

    Returns
    -------
    float
        The computed expected value.

    Example
    -------
    >>> bins = [(0, 10), (10, 20), (20, 30)]
    >>> probs = [0.5, 0.3, 0.2]
    >>> expectedValue(bins, probs)
    12.0

    Note
    ----
    The order of bin ranges in 'binRanges' should match the order of
    probabilities in 'probabilities'.
    """
    expectedV = 0.0
    for index, binrange in enumerate(binRanges):
        # TODO: is this the correct order? binrange has max, min?
        v_max, v_min = binrange

        meanBinvalue = ((v_max - v_min) / 2) + v_min

        expectedV += meanBinvalue * probabilities[index]

    return expectedV


def BNskelFromCSV(
    csvdata: Union[str, List[str]], targets: List[str]
) -> GraphSkeleton:
    """
    Generate a GraphSkeleton (Bayesian Network structure) from a CSV file or
    data.

    This function creates a Bayesian Network structure based on the columns
    present in a CSV file or data. Columns not specified as targets are
    treated as input vertices.

    Parameters
    ----------
    csvdata : str or list
        If a string, it is assumed to be the filepath to a CSV file. The CSV
        should have its first row as headers which are considered as the names
        of the vertices in the Bayesian Network. If a list, the first element
        should be a list of headers (vertex names).

    targets : list of str
        A list of strings specifying which columns in the CSV data should be
        considered as target vertices in the Bayesian Network. All other
        columns will be treated as input vertices.

    Returns
    -------
    GraphSkeleton
        An object representing the structure of the Bayesian Network. This
        includes vertices (V) and edges (E).

    Notes
    -----
    - Edges are formed between all input vertices and target vertices. The
      direction of edges depends on the number of inputs vs. targets.
    - The function uses the GraphSkeleton class from the `libpgm` library.

    Example
    -------
    >>> BNskelFromCSV("data.csv", ["TargetA", "TargetB"])
    # This will create a Bayesian Network structure using the columns in "data.csv"
    # with "TargetA" and "TargetB" as target vertices.
    """
    # TODO #49: Refactor BNskelFromCSV to include swapping direction of too many inputs into a node

    # EXTRACT HEADER STRINGS FROM CSV FILE
    # TODO: this breaks now as the libpgm dependency has been removed
    skel = GraphSkeleton()
    BNstructure = {}
    inputVerts = []

    # if data is a filepath
    if isinstance(
        csvdata, str
    ):  # previously (csvdata, basestring) python 2.0 compatability
        dataset = []
        with open(csvdata, "rb") as csvfile:
            lines = csv.reader(csvfile)

            for row in lines:
                dataset.append(row)

        allVertices = dataset[0]

    else:
        allVertices = csvdata[0]

    BNstructure["V"] = allVertices
    skel.V = allVertices

    for verts in allVertices:
        if verts not in targets:
            inputVerts.append(verts)

    # target, each input
    edges = []
    # structure = {}
    if len(inputVerts) > len(targets):
        for target in targets:
            for input in inputVerts:
                # structure [input] = []
                edge = [target, input]
                edges.append(edge)

        BNstructure["E"] = edges
        skel.E = edges
    else:
        for input in inputVerts:
            for target in targets:
                edge = [input, target]
                edges.append(edge)
        BNstructure["E"] = edges
        skel.E = edges

    # print ('edges\n ',edges)

    skel.toporder()

    return skel

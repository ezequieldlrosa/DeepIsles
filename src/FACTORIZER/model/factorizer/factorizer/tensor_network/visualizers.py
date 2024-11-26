from networkx import Graph, MultiGraph

from typing import Tuple, Union, Dict, Hashable

Coordinate2D = Tuple[float, float]
Coordinate3D = Tuple[float, float, float]


def vis2d(
    g: Union[Graph, MultiGraph], pos: Dict[Hashable, Coordinate2D]
) -> Dict:
    xn = []  # x-coordinates of nodes
    yn = []  # y-coordinates
    node_size = []
    node_color_top = []
    node_color_bottom = []
    node_line_color = []
    for k, v in pos.items():
        xn.append(v[0])
        yn.append(v[1])
        node_size.append(g.nodes[k]["size"] * 12)
        if "data" in g.nodes[k]["tags"]:
            node_color_top.append("rgb(218, 232, 252)")
            node_color_bottom.append("rgb(126, 166, 224)")
            node_line_color.append("rgb(108, 142, 191)")
        else:
            node_color_top.append("rgb(248,206,204)")
            node_color_bottom.append("rgb(234,107,102)")
            node_line_color.append("rgb(184,84,80)")

    xe = []  # x-coordinates of edge ends
    ye = []  # y-coordinates
    for e in g.edges():
        xe += [pos[e[0]][0], pos[e[1]][0], None]
        ye += [pos[e[0]][1], pos[e[1]][1], None]

    edge_trace = {
        "type": "scatter",
        "x": xe,
        "y": ye,
        "mode": "lines",
        "line": {"color": "rgb(128,128,128)", "width": 2},
        "hoverinfo": "none",
    }

    node_trace = {
        "type": "scatter",
        "x": xn,
        "y": yn,
        "mode": "markers",
        "name": "actors",
        "marker": {
            "symbol": "circle",
            "size": node_size,
            "color": node_color_top,
            "gradient": {"type": "vertical", "color": node_color_bottom},
            "line": {"color": node_line_color, "width": 2},
            "opacity": 1,
        },
        "text": list(pos.keys()),
        "hoverinfo": "text",
    }

    data = [edge_trace, node_trace]

    layout = {
        "width": 600,
        "height": 600,
        "showlegend": False,
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "plot_bgcolor": "rgb(255,255,255)",
        "hovermode": "closest",
    }

    fig = {"data": data, "layout": layout}

    return fig


def vis3d(g: Union[Graph, MultiGraph], pos: Dict[Hashable, Coordinate3D]):
    xn = []  # x-coordinates of nodes
    yn = []  # y-coordinates
    zn = []  # z-coordinates
    node_size = []
    node_color = []
    node_line_color = []
    for k, v in pos.items():
        xn.append(v[0])
        yn.append(v[1])
        zn.append(v[2])
        node_size.append(g.nodes[k]["size"] * 12)
        if "data" in g.nodes[k]["tags"]:
            node_color.append("rgb(126, 166, 224)")
            node_line_color.append("rgb(108, 142, 191)")
        else:
            node_color.append("rgb(234,107,102)")
            node_line_color.append("rgb(184,84,80)")

    xe = []  # x-coordinates of edge ends
    ye = []  # y-coordinates
    ze = []  # z-coordinates
    for e in g.edges():
        xe += [pos[e[0]][0], pos[e[1]][0], None]
        ye += [pos[e[0]][1], pos[e[1]][1], None]
        ze += [pos[e[0]][2], pos[e[1]][2], None]

    edge_trace = {
        "type": "scatter3d",
        "x": xe,
        "y": ye,
        "z": ze,
        "mode": "lines",
        "line": {"color": "rgb(128,128,128)", "width": 2},
        "hoverinfo": "none",
    }

    node_trace = {
        "type": "scatter3d",
        "x": xn,
        "y": yn,
        "z": zn,
        "mode": "markers",
        "name": "actors",
        "marker": {
            "symbol": "circle",
            "size": node_size,
            "color": node_color,
            "line": {"color": node_line_color, "width": 2},
            "opacity": 1,
        },
        "text": list(pos.keys()),
        "hoverinfo": "text",
    }

    data = [edge_trace, node_trace]

    axis = {
        "visible": False,
        "showbackground": False,
        "showline": False,
        "zeroline": False,
        "showgrid": False,
        "showticklabels": False,
        "title": "",
    }

    layout = {
        "width": 600,
        "height": 600,
        "showlegend": False,
        "scene": {"xaxis": axis, "yaxis": axis, "zaxis": axis},
        "hovermode": "closest",
    }

    fig = {"data": data, "layout": layout}

    return fig

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.utils import to_networkx
from pathlib import Path
from typing import Union, Optional
import py3Dmol
import tempfile
import wandb


# =============================================================================
# Private Helper Functions (must be defined first)
# =============================================================================

def _get_ca_coordinates(pdb_path: str) -> np.ndarray:
    """Extract CA atom coordinates from PDB file."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)


def _get_residue_indices(pdb_path: str) -> list[int]:
    """Extract unique residue indices from PDB in order of appearance."""
    residues = []
    seen = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    res_num = int(line[22:26].strip())
                    if res_num not in seen:
                        residues.append(res_num)
                        seen.add(res_num)
                except ValueError:
                    continue
    return residues


def _create_confusion_matrix_image(tp: int, fp: int, fn: int, tn: int) -> wandb.Image:
    """Create a confusion matrix heatmap as wandb.Image."""
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: 0', 'Pred: 1'])
    ax.set_yticklabels(['True: 0', 'True: 1'])

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)

    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _create_icosphere(center: np.ndarray, radius: float, subdivisions: int = 1) -> tuple:
    """Create an icosphere mesh at the given center."""
    phi = (1 + np.sqrt(5)) / 2

    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float32)

    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius + center

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces, center, radius)

    return verts, faces


def _subdivide_icosphere(verts: np.ndarray, faces: np.ndarray, center: np.ndarray, radius: float) -> tuple:
    """Subdivide icosphere faces."""
    edge_midpoints = {}
    new_faces = []

    def get_midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key not in edge_midpoints:
            mid = (verts[i1] + verts[i2]) / 2
            mid = (mid - center) / np.linalg.norm(mid - center) * radius + center
            edge_midpoints[key] = len(verts) + len(edge_midpoints)
        return edge_midpoints[key]

    new_verts = list(verts)
    for f in faces:
        a, b, c = f
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
        ])

    for key in sorted(edge_midpoints.keys(), key=lambda k: edge_midpoints[k]):
        mid = (verts[key[0]] + verts[key[1]]) / 2
        mid = (mid - center) / np.linalg.norm(mid - center) * radius + center
        new_verts.append(mid)

    return np.array(new_verts), np.array(new_faces)


def _write_ply(path: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """Write PLY file with vertices, faces, and colors."""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# =============================================================================
# PDB File Utilities
# =============================================================================

def write_labeled_pdb(
    pdb_path: str,
    labels: Union[np.ndarray, torch.Tensor],
    output_path: str,
    use_bfactor: bool = True,
) -> None:
    """
    Write a PDB file with labels stored in B-factor column.

    This allows visualization in any molecular viewer (PyMOL, ChimeraX, etc.)
    by coloring by B-factor.

    Args:
        pdb_path: Input PDB file path
        labels: Labels per residue (N,)
        output_path: Output PDB file path
        use_bfactor: If True, write to B-factor column; if False, write to occupancy
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = labels.flatten()

    residue_indices = _get_residue_indices(pdb_path)
    label_map = {res_idx: labels[i] for i, res_idx in enumerate(residue_indices) if i < len(labels)}

    with open(pdb_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    res_num = int(line[22:26].strip())
                    label_val = label_map.get(res_num, 0.0)

                    if use_bfactor:
                        new_line = line[:60] + f'{label_val:6.2f}' + line[66:]
                    else:
                        new_line = line[:54] + f'{label_val:6.2f}' + line[60:]
                    f_out.write(new_line)
                except (ValueError, IndexError):
                    f_out.write(line)
            else:
                f_out.write(line)


# =============================================================================
# Wandb Visualization Functions
# =============================================================================

def create_wandb_molecule(
    pdb_path: str,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold: float = 0.5,
) -> tuple:
    """
    Create wandb.Molecule objects with labels in B-factor for coloring.

    Args:
        pdb_path: Path to PDB file
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels/probabilities (N,), optional
        threshold: Threshold for predictions

    Returns:
        If y_pred is None: (truth_molecule,)
        If y_pred provided: (truth_molecule, prediction_molecule)

    Note: In wandb, select "Color by: B-factor" to see the coloring.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix='_truth.pdb', delete=False, mode='w') as f:
        truth_path = f.name
    write_labeled_pdb(pdb_path, y_true, truth_path)
    truth_mol = wandb.Molecule(truth_path)

    if y_pred is None:
        return (truth_mol,)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix='_pred.pdb', delete=False, mode='w') as f:
        pred_path = f.name
    write_labeled_pdb(pdb_path, y_pred, pred_path)
    pred_mol = wandb.Molecule(pred_path)

    return (truth_mol, pred_mol)


def create_wandb_point_cloud(
    pdb_path: str,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold: float = 0.5,
) -> tuple:
    """
    Create wandb.Object3D point clouds colored by binding labels.

    Args:
        pdb_path: Path to PDB file
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels/probabilities (N,), optional
        threshold: Threshold for predictions

    Returns:
        If y_pred is None: (truth_cloud,)
        If y_pred provided: (truth_cloud, prediction_cloud)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    y_true = y_true.flatten()

    coords = _get_ca_coordinates(pdb_path)
    n_residues = len(coords)

    truth_colors = np.array([[255, 0, 0] if y_true[i] == 1 else [180, 180, 180]
                            for i in range(min(n_residues, len(y_true)))])
    truth_cloud = np.hstack([coords[:len(truth_colors)], truth_colors])
    truth_obj = wandb.Object3D(truth_cloud)

    if y_pred is None:
        return (truth_obj,)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_pred_binary = (y_pred.flatten() >= threshold).astype(int)

    pred_colors = []
    for i in range(min(n_residues, len(y_true), len(y_pred_binary))):
        if y_true[i] == 1 and y_pred_binary[i] == 1:
            pred_colors.append([0, 255, 0])  # TP - green
        elif y_true[i] == 0 and y_pred_binary[i] == 1:
            pred_colors.append([255, 255, 0])  # FP - yellow
        elif y_true[i] == 1 and y_pred_binary[i] == 0:
            pred_colors.append([255, 0, 0])  # FN - red
        else:
            pred_colors.append([180, 180, 180])  # TN - gray
    pred_colors = np.array(pred_colors)
    pred_cloud = np.hstack([coords[:len(pred_colors)], pred_colors])
    pred_obj = wandb.Object3D(pred_cloud)

    return (truth_obj, pred_obj)


def create_wandb_comparison_table(
    pdb_path: str,
    epoch: int,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    pdb_name: Optional[str] = None,
    include_molecules: bool = True,
) -> wandb.Table:
    """
    Create a wandb.Table with point clouds, molecules, and metrics for a single protein.

    Args:
        pdb_path: Path to PDB file
        epoch: Current epoch number
        y_true: Ground truth labels (N,)
        y_pred: Predicted probabilities (N,)
        threshold: Threshold for predictions
        pdb_name: Name to display (defaults to filename)
        include_molecules: If True, include wandb.Molecule columns

    Returns:
        wandb.Table with visualization and metrics columns
    """
    if pdb_name is None:
        pdb_name = Path(pdb_path).stem

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten().astype(int)
    y_pred_flat = y_pred.flatten()
    y_pred_binary = (y_pred_flat >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred_binary == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_binary == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_binary == 0)).sum())
    tn = int(((y_true == 0) & (y_pred_binary == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    truth_cloud, pred_cloud = create_wandb_point_cloud(pdb_path, y_true, y_pred_binary, threshold=0.5)
    cm_image = _create_confusion_matrix_image(tp, fp, fn, tn)

    if include_molecules:
        truth_mol, _ = create_wandb_molecule(pdb_path, y_true, y_pred_flat)

        table = wandb.Table(columns=[
            "PDB + Epoch",
            "Molecule (Ground Truth)",
            "Truth (Points)", "Pred (Points)",
            "Confusion Matrix", "Precision", "Recall", "F1"
        ])

        table.add_data(
            pdb_name + f" (Epoch {epoch})",
            truth_mol,
            truth_cloud, pred_cloud,
            cm_image,
            round(precision, 4), round(recall, 4), round(f1, 4)
        )
    else:
        table = wandb.Table(columns=[
            "PDB + Epoch", "Ground Truth", "Prediction",
            "Confusion Matrix", "Precision", "Recall", "F1"
        ])

        table.add_data(
            pdb_name + f" (Epoch {epoch})",
            truth_cloud, pred_cloud,
            cm_image,
            round(precision, 4), round(recall, 4), round(f1, 4)
        )

    return table


def add_to_wandb_comparison_table(
    table: wandb.Table,
    pdb_path: str,
    epoch: int,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    pdb_name: Optional[str] = None,
    include_molecules: bool = True,
) -> wandb.Table:
    """
    Add a row to an existing comparison table.

    Args:
        table: Existing wandb.Table
        pdb_path: Path to PDB file
        epoch: Current epoch number
        y_true: Ground truth labels (N,)
        y_pred: Predicted probabilities (N,)
        threshold: Threshold for predictions
        pdb_name: Name to display
        include_molecules: If True, include wandb.Molecule columns

    Returns:
        The updated table
    """
    if pdb_name is None:
        pdb_name = Path(pdb_path).stem

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten().astype(int)
    y_pred_flat = y_pred.flatten()
    y_pred_binary = (y_pred_flat >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred_binary == 1)).sum())
    fp = int(((y_true == 0) & (y_pred_binary == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_binary == 0)).sum())
    tn = int(((y_true == 0) & (y_pred_binary == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    truth_cloud, pred_cloud = create_wandb_point_cloud(pdb_path, y_true, y_pred_binary, threshold=0.5)
    cm_image = _create_confusion_matrix_image(tp, fp, fn, tn)

    if include_molecules:
        truth_mol, _ = create_wandb_molecule(pdb_path, y_true, y_pred_flat)
        table.add_data(
            pdb_name + f" (Epoch {epoch})",
            truth_mol,
            truth_cloud, pred_cloud,
            cm_image,
            round(precision, 4), round(recall, 4), round(f1, 4)
        )
    else:
        table.add_data(
            pdb_name + f" (Epoch {epoch})",
            truth_cloud, pred_cloud,
            cm_image,
            round(precision, 4), round(recall, 4), round(f1, 4)
        )

    return table


# =============================================================================
# Mesh/PLY Functions
# =============================================================================

def create_mesh_ply(
    pdb_path: str,
    labels: Union[np.ndarray, torch.Tensor],
    output_path: str,
    threshold: float = 0.5,
    binding_color: tuple = (255, 0, 0),
    non_binding_color: tuple = (180, 180, 180),
    radius: float = 1.5,
):
    """
    Create a PLY mesh file with colored spheres at CA positions.

    This can be logged to wandb via wandb.Object3D(output_path).
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels_binary = (labels.flatten() >= threshold).astype(int)

    coords = _get_ca_coordinates(pdb_path)
    n_residues = min(len(coords), len(labels_binary))

    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for i in range(n_residues):
        center = coords[i]
        color = binding_color if labels_binary[i] == 1 else non_binding_color

        verts, faces = _create_icosphere(center, radius, subdivisions=1)
        all_vertices.append(verts)
        all_faces.append(faces + vertex_offset)
        all_colors.extend([color] * len(verts))
        vertex_offset += len(verts)

    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)
    all_colors = np.array(all_colors, dtype=np.uint8)

    _write_ply(output_path, all_vertices, all_faces, all_colors)


# =============================================================================
# py3Dmol Visualization Functions
# =============================================================================

def compare_binding_predictions(
    pdb_path: str,
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    width: int = 800,
    height: int = 400,
) -> py3Dmol.view:
    """
    Visualize ground truth vs predicted binding sites side by side.

    Color scheme:
        - True Positive (TP): Green
        - False Positive (FP): Yellow
        - False Negative (FN): Red
        - True Negative (TN): Gray
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten().astype(int)
    y_pred_binary = (y_pred.flatten() >= threshold).astype(int)

    with open(pdb_path, 'r') as f:
        pdb_content = f.read()

    view = py3Dmol.view(width=width, height=height, viewergrid=(1, 2))

    view.addModel(pdb_content, 'pdb', viewer=(0, 0))
    view.addModel(pdb_content, 'pdb', viewer=(0, 1))

    view.setStyle({'cartoon': {'color': 'lightgray'}}, viewer=(0, 0))
    view.setStyle({'cartoon': {'color': 'lightgray'}}, viewer=(0, 1))

    residue_indices = _get_residue_indices(pdb_path)

    for i, (res_idx, true_label) in enumerate(zip(residue_indices, y_true)):
        color = 'red' if true_label == 1 else 'lightgray'
        view.addStyle({'resi': res_idx}, {'cartoon': {'color': color}}, viewer=(0, 0))

    for i, (res_idx, true_label, pred_label) in enumerate(zip(residue_indices, y_true, y_pred_binary)):
        if true_label == 1 and pred_label == 1:
            color = 'green'
        elif true_label == 0 and pred_label == 1:
            color = 'yellow'
        elif true_label == 1 and pred_label == 0:
            color = 'red'
        else:
            color = 'lightgray'
        view.addStyle({'resi': res_idx}, {'cartoon': {'color': color}}, viewer=(0, 1))

    view.zoomTo(viewer=(0, 0))
    view.zoomTo(viewer=(0, 1))

    return view


def visualize_binding_surface(
    pdb_path: str,
    labels: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    width: int = 600,
    height: int = 400,
    style: str = 'surface',
    binding_color: str = 'red',
    non_binding_color: str = 'lightgray',
) -> py3Dmol.view:
    """
    Visualize a single structure with binding site coloring.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    labels_binary = (labels.flatten() >= threshold).astype(int)

    with open(pdb_path, 'r') as f:
        pdb_content = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_content, 'pdb')

    residue_indices = _get_residue_indices(pdb_path)

    if style == 'surface':
        view.setStyle({'cartoon': {'color': non_binding_color}})
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': non_binding_color})
    elif style == 'cartoon':
        view.setStyle({'cartoon': {'color': non_binding_color}})
    else:
        view.setStyle({'stick': {'color': non_binding_color}})

    binding_resi = [res_idx for res_idx, label in zip(residue_indices, labels_binary) if label == 1]
    if binding_resi:
        sel = {'resi': binding_resi}
        if style == 'surface':
            view.addStyle(sel, {'cartoon': {'color': binding_color}})
            view.addSurface(py3Dmol.VDW, {'opacity': 0.9, 'color': binding_color}, sel)
        elif style == 'cartoon':
            view.addStyle(sel, {'cartoon': {'color': binding_color}})
        else:
            view.addStyle(sel, {'stick': {'color': binding_color}})

    view.zoomTo()
    return view


# =============================================================================
# Graph Visualization Functions
# =============================================================================

def edge_to_dense(data, num_nodes=None, edge_attr_channel=0):
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    if num_nodes is None:
        num_nodes = data.num_nodes

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    if edge_attr is not None:
        if edge_attr.dim() > 1:
            edge_attr = edge_attr[:, edge_attr_channel]
    else:
        edge_attr = torch.ones(edge_index.size(1))

    for i in range(edge_index.size(1)):
        if edge_attr[i] == 0:
            continue
        src = edge_index[0, i]
        dst = edge_index[1, i]
        A[src, dst] = edge_attr[i]

    return A / (A.max() + 1e-8)


def backbone_graph(data):
    node_xyz = data.pos

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45., azim=-45)

    ax.scatter(
        node_xyz[:, 0],
        node_xyz[:, 1],
        node_xyz[:, 2],
        c=data.y.numpy(),
        cmap='coolwarm',
        s=100
    )

    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Non-binding', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Binding', markerfacecolor='red', markersize=10)
    ])

    peptide_mask = data.edge_attr[:, 1] == 0
    peptide_edges = data.edge_index[:, peptide_mask]

    for i in range(peptide_edges.size(1)):
        u, v = peptide_edges[0, i].item(), peptide_edges[1, i].item()
        x = [node_xyz[u, 0], node_xyz[v, 0]]
        y = [node_xyz[u, 1], node_xyz[v, 1]]
        z = [node_xyz[u, 2], node_xyz[v, 2]]
        ax.plot(x, y, z, color="black", linewidth=2)

    ax.set_title("3D Protein Backbone (Peptide Bonds Only)")
    plt.show()


def visualize_graph(G, color):
    plt.figure(figsize=(15, 9))
    plt.xticks([])
    plt.yticks([])
    edge_values = [attr[2] for _, _, attr in G.edges(data="edge_attr")]
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap='coolwarm', edge_color=edge_values,
                     edge_cmap=plt.cm.viridis, edge_vmin=0, edge_vmax=1)
    plt.title("Protein Graph Visualization")
    plt.show()
